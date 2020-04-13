use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::fs;
use std::rc::Rc;

use crate::error::LispyError;
use crate::parser;
use crate::Result;

// Unfortunately I didn't find a better way to share the environment
// between `Lval`s and also have a reference to a parent environment.
// TODO. Revisit this later.
type SharedEnv = Rc<RefCell<LEnv>>;
type BuiltinFn = fn(Lval, &mut SharedEnv) -> Result<Lval>;

/// Represents Lispy's environment.
#[derive(Default, Clone, PartialEq)]
pub struct LEnv {
    map: HashMap<String, Lval>,
    parent: Option<SharedEnv>,
}

impl LEnv {
    /// Performs a symbol lookup in the environemnt and its parents and returns a matching `Lval`.
    ///
    /// # Arguments
    ///
    /// * `symbol` - A string slice that holds the name of the lookup symbol
    ///
    /// As a side-effect it will perform deep copy of `Lval`
    fn get(&self, symbol: &str) -> Result<Lval> {
        if let Some(val) = self.map.get(symbol) {
            // NOTE
            // We have to manually handle Fun::Lambda value, because it contains a reference counted pointer
            // to LEnv. When clone() is called on it - it will simply increase the counter but won't create
            // an actual copy. In this particular case it's not what we want. We want a new copy of 'val'.
            // Not doing this results in stack overflow as we may be able to substitute .parent for the
            // original LEnv.
            // FIXME
            // There must be a better way to do this.
            let mut val = (*val).clone();
            if let Lval::Fun(LvalFun::Lambda(ref mut lambda)) = &mut val {
                let env = lambda.env.borrow().clone().into_shared();
                lambda.env = env;
            }
            Ok(val)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().get(symbol)
        } else {
            Err(LispyError::UnboundSymbol(symbol.to_owned()))
        }
    }

    /// Bounds provided symbol to `Lval` in the current environment.
    ///
    /// # Arguments
    ///
    /// * `symbol` - A string slice that holds the name of the lookup symbol
    ///
    /// If the symbol was already bounded - updates its value.
    fn put(&mut self, symbol: &str, lval: Lval) {
        self.map.insert(symbol.to_owned(), lval);
    }

    /// Bounds provided symbol to `Lval` in the parent environment (recursively) if any.
    ///
    /// * `symbol` - A string slice that holds the name of the lookup symbol
    ///
    /// Otherwise bounds the symbol in the current environment.
    fn def(&mut self, key: &str, val: Lval) {
        if let Some(ref mut parent) = self.parent {
            return parent.borrow_mut().def(key, val);
        }

        self.put(key, val);
    }

    /// Registers provided `BuiltinFn` under name in the current environment.
    ///
    /// # Arguments
    ///
    /// * `name` - A string slice that holds the name of the builtin
    /// * `func` - A function of `BuiltinFn` type
    fn add_builtin(&mut self, name: &'static str, func: BuiltinFn) {
        self.put(name, Lval::builtin(name, func))
    }

    /// Creates a new environment that can be shared between multiple owners
    pub fn new_shared() -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(LEnv::default()))
    }

    /// Converts this environment into a shared one
    pub fn into_shared(self) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(self))
    }
}

/// This trait provides an interface for Lispy functions.
trait Fun {
    /// Calls this function with provided arguments and environment. Returns `Lval` as a result of evaluation.
    ///
    /// # Arguments
    ///
    /// * `sexpr` - An S-Expression that represents a function to be called
    /// * `env` - Callee's environment
    fn call(self, env: &mut SharedEnv, sexpr: Lval) -> Result<Lval>;
}

/// Respresents a built-in function.
#[derive(Clone)]
pub struct Builtin {
    name: &'static str,
    func: BuiltinFn,
}

impl PartialEq for Builtin {
    fn eq(&self, other: &Self) -> bool {
        // FIXME
        // For builtin functions we have to also compare function pointer. It's a bit tricky to do in
        // Rust for many reasons and I'm not even sure this trick works properly.
        self.name == other.name
            && self.func as *const BuiltinFn as usize == other.func as *const BuiltinFn as usize
    }
}

impl Fun for Builtin {
    fn call(self, env: &mut SharedEnv, expr: Lval) -> Result<Lval> {
        debug!("Calling {}", self.name);
        (self.func)(expr, env)
    }
}

/// Respresents a user-defined lambda function.
#[derive(Clone, PartialEq)]
pub struct Lambda {
    env: SharedEnv,
    formals: Box<Lval>,
    body: Box<Lval>,
}

impl Fun for Lambda {
    fn call(mut self, env: &mut SharedEnv, mut expr: Lval) -> Result<Lval> {
        debug!("Calling a lambda");
        let total = self.formals.len();
        let given = expr.len();

        while !expr.is_empty() {
            // If there are still arguments, but we exhausted all the formals
            // - return an error
            if self.formals.is_empty() {
                return Err(LispyError::InvalidArgNum("Fun", total, given));
            }

            let sym = self.formals.pop(0);

            // To handle variable arguments we have to convert all leftover arguments
            // into a list and store it in local environment under the name that follows
            // '&'. Then we can exit the loop.
            if sym.as_symbol() == "&" {
                if self.formals.len() != 1 {
                    return Err(LispyError::InvalidFormat(
                        "Fun format is invalid. Symbol '&' not followed by single symbol.",
                    ));
                }

                let next_sym = self.formals.pop(0);
                self.env
                    .borrow_mut()
                    .put(&next_sym.as_symbol(), expr.into_qexpr());
                break;
            }

            // Bind the argument into function's environment
            self.env.borrow_mut().put(sym.as_symbol(), expr.pop(0));
        }

        // In case function accepts variable arguments but they were not yet
        // specified we should bind an empty list instead.
        if !self.formals.is_empty() && self.formals.peek(0).as_symbol() == "&" {
            if self.formals.len() != 2 {
                return Err(LispyError::InvalidFormat(
                    "Fun format is invalid. Symbol '&' not followed by a single symbol.",
                ));
            }

            self.formals.pop(0);

            env.borrow_mut()
                .put(self.formals.pop(0).as_symbol(), Lval::qexpr());
        }

        // If we managed to bind all the arguments we can now evaluate this function.
        // If not we just return an updated function.
        if self.formals.is_empty() {
            self.env.borrow_mut().parent = Some(Rc::clone(env));
            self.body.into_sexpr().eval(&mut self.env)
        } else {
            Ok(Lval::Fun(LvalFun::Lambda(self)))
        }
    }
}

/// Assert-like macros similar to those defined in BuildYourOwnLisp book.
/// They would return `LispyError` in case assertion fails. This is C-ish
/// way if doing things, but I wanted to follow the book closely, to
/// begin with.

/// Checks if lval contains an expression with correct number of arguments
macro_rules! lassert_num {
    ($name:expr, $lval:ident, $num:expr) => {
        match &$lval {
            Lval::SExpr(cells) | Lval::QExpr(cells) => {
                if cells.len() != $num {
                    return Err(LispyError::InvalidArgNum($name, $num, cells.len()));
                }
            }
            _ => unreachable!(),
        }
    };
}

/// Checks if lval contains an non-empty expression
macro_rules! lassert_not_empty {
    ($name:expr, $lval:ident) => {
        match &$lval {
            Lval::SExpr(cells) | Lval::QExpr(cells) => {
                if cells.is_empty() {
                    return Err(LispyError::EmptyList($name));
                }
            }
            _ => unreachable!(),
        }
    };
}

/// Checks if lval is of expected type
macro_rules! lassert_type {
    ($name:expr, $lval:ident, $($type:pat)|+ ) => {
        match &$lval {
            $($type)|+ => {}
            _ => {
                return Err(LispyError::InvalidType(
                    $name,
                    stringify!($($type)|+),
                    $lval.to_string(),
                ))
            }
        }
    };
}

/// A wrapper type for Lispy function
#[derive(Clone, PartialEq)]
pub enum LvalFun {
    Builtin(Builtin),
    Lambda(Lambda),
}

/// Represents Lispy value. The main data structure our Lispy is built on.
#[derive(Clone, PartialEq)]
pub enum Lval {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Symbol(String),
    String(String),
    Fun(LvalFun),
    SExpr(Vec<Lval>),
    QExpr(Vec<Lval>),
    Exit,
}

impl fmt::Display for Lval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Lval::Boolean(x) => write!(f, "{}", x),
            Lval::Integer(x) => write!(f, "{}", x),
            Lval::Float(x) => write!(f, "{}", x),
            Lval::Symbol(sym) => write!(f, "{}", sym),
            Lval::String(string) => write!(f, "\"{}\"", string),
            Lval::Fun(LvalFun::Builtin(builtin)) => write!(f, "<builtin>: {}", builtin.name),
            Lval::Fun(LvalFun::Lambda(lambda)) => write!(f, "(lambda \nbody: {})", lambda.body),
            Lval::SExpr(v) => {
                write!(f, "(")?;
                for (index, lval) in v.iter().enumerate() {
                    write!(f, "{}", lval)?;
                    if index != v.len() - 1 {
                        write!(f, " ")?;
                    }
                }
                write!(f, ")")
            }
            Lval::QExpr(v) => {
                write!(f, "{{")?;
                for (index, lval) in v.iter().enumerate() {
                    write!(f, "{}", lval)?;
                    if index != v.len() - 1 {
                        write!(f, " ")?;
                    }
                }
                write!(f, "}}")
            }
            Lval::Exit => write!(f, "exit"),
        }
    }
}

impl fmt::Debug for Lval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Lval {
    /// Returns the underlying number representation if the type is
    /// `Lval::Integer`. Will panic otherwise.
    fn as_integer(&self) -> i64 {
        match self {
            Lval::Integer(num) => *num,
            _ => unreachable!(),
        }
    }

    /// Returns a reference to the underlying symbol representation if the type is
    /// `Lval::Symbol`. Will panic otherwise.
    fn as_symbol(&self) -> &str {
        match self {
            Lval::Symbol(ref sym) => sym,
            _ => unreachable!(),
        }
    }

    /// Converts this `Lval` into an S-Expression, preserving internal structure. In case `self` is not an expression
    /// - returns an empty S-Expression.
    fn into_sexpr(self) -> Lval {
        match self {
            Lval::QExpr(cells) => Lval::SExpr(cells),
            Lval::SExpr(_) => self,
            _ => Lval::SExpr(vec![]),
        }
    }

    /// Converts this `Lval` into a Q-Expression, preserving internal structure. In case `self` is not an expression
    /// - returns an empty Q-Expression.
    fn into_qexpr(self) -> Lval {
        match self {
            Lval::SExpr(cells) => Lval::QExpr(cells),
            Lval::QExpr(_) => self,
            _ => Lval::QExpr(vec![]),
        }
    }

    /// Returns a reference to a cell located at `index`. Panics in case `self` is not an expression or index is out
    /// of bounds.
    fn peek(&self, index: usize) -> &Lval {
        match self {
            Lval::SExpr(cells) | Lval::QExpr(cells) => &cells[index],
            _ => unreachable!(),
        }
    }

    /// Removes and returns a cell located at `index`. Panics in case `self` is not an expression or index is out
    /// of bounds.
    fn pop(&mut self, index: usize) -> Lval {
        match self {
            Lval::SExpr(cells) | Lval::QExpr(cells) => cells.remove(index),
            _ => unreachable!(),
        }
    }

    /// Inserts a cell at `index` and shift other cells to right. Panics in case `self` is not an expression.
    fn insert(&mut self, lval: Lval, index: usize) {
        match self {
            Lval::SExpr(cells) | Lval::QExpr(cells) => cells.insert(index, lval),
            _ => unreachable!(),
        }
    }

    /// Removes and returns a cell located at `index`. Consumes `self`.
    /// Panics in case `self` is not an expression or index is out of bounds.
    fn take(mut self, i: usize) -> Lval {
        self.pop(i)
    }

    /// Returns number of cells in `self`. Panics in case `self` is not an expression.
    fn len(&self) -> usize {
        match self {
            Lval::SExpr(cells) | Lval::QExpr(cells) => cells.len(),
            _ => unreachable!(),
        }
    }

    /// Returns `true` if `self` contains no cells. Panics in case `self` is not an expression.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Joins `self` and `other` expression together. Consumes `self`, `other` and returns new `Lval`.
    /// Panics is either argument is not an expression.
    fn join(self, mut other: Self) -> Lval {
        let mut lval = self;
        while !other.is_empty() {
            lval = lval.add(other.pop(0));
        }

        lval
    }

    /// Adds `val` to `self` expression. Consumes `self`, `val` and returns new `Lval`.
    /// Panics is either argument is not an expression.
    pub fn add(self, val: Self) -> Lval {
        let mut lval = self;
        match &mut lval {
            Lval::QExpr(ref mut cells) | Lval::SExpr(ref mut cells) => cells.push(val),
            _ => unreachable!(),
        }

        lval
    }

    /// Calls `self` as a function. Consumes `self` and returns result of the execution.
    ///
    /// # Arguments
    ///
    /// * `env` - Callee's environment
    /// * `args` - An S-Expression containing arguments, this function should be called with
    fn call(self, env: &mut SharedEnv, args: Lval) -> Result<Lval> {
        match self {
            Lval::Fun(LvalFun::Builtin(builtin)) => builtin.call(env, args),
            Lval::Fun(LvalFun::Lambda(lambda)) => lambda.call(env, args),
            _ => unreachable!(),
        }
    }

    /// Evaluates `self` and returns new `Lval` as a result. Consumes `self`.
    pub fn eval(self, env: &mut SharedEnv) -> Result<Lval> {
        debug!("eval: {}", self);

        match self {
            Lval::Symbol(sym) => env.borrow().get(&sym),
            Lval::SExpr(ref cells) if cells.is_empty() => Ok(self),
            Lval::SExpr(cells) => {
                let mut aux = vec![];
                for cell in cells {
                    aux.push(cell.eval(env)?);
                }
                let mut cells = aux;

                // If there's only one argument - evaluate it and return
                // the result of evaluation
                if cells.len() == 1 {
                    return cells.remove(0).eval(env);
                }

                // Otherwise it's an either builtin or a user-defined function
                let func = cells.remove(0);
                lassert_type!("S-Expression", func, Lval::Fun(_));
                func.call(env, Lval::SExpr(cells))
            }
            _ => Ok(self),
        }
    }

    /// Constructs a new `Lval` holding an S-Expression
    pub fn sexpr() -> Lval {
        Lval::SExpr(vec![])
    }

    /// Constructs a new `Lval` holding an Q-Expression
    pub fn qexpr() -> Lval {
        Lval::QExpr(vec![])
    }

    /// Constructs a new `Lval` holding a symbol
    pub fn symbol(sym: &str) -> Lval {
        Lval::Symbol(sym.to_owned())
    }

    /// Constructs a new `Lval` holding a number
    pub fn integer(num: i64) -> Lval {
        Lval::Integer(num)
    }

    /// Constructs a new `Lval` holding a built-in function
    pub fn builtin(name: &'static str, func: BuiltinFn) -> Lval {
        Lval::Fun(LvalFun::Builtin(Builtin { name, func }))
    }

    /// Constructs a new `Lval` holding a user-defined function
    pub fn lambda(formals: Lval, body: Lval) -> Lval {
        Lval::Fun(LvalFun::Lambda(Lambda {
            env: LEnv::new_shared(),
            formals: Box::new(formals),
            body: Box::new(body),
        }))
    }

    /// Constructs a new `Lval` holding a `bool` value
    pub fn boolean(val: bool) -> Lval {
        Lval::Boolean(val)
    }

    /// Returns a reference to the underlying `bool` representation if the type is
    /// `Lval::Boolean`. Will panic otherwise.
    fn as_boolean(&self) -> &bool {
        match self {
            Lval::Boolean(val) => val,
            _ => unreachable!(),
        }
    }

    /// Converts this `Lval` into a boolean, In case `self` is a boolean or a number
    /// - panics.
    fn into_boolean(self) -> Lval {
        match self {
            Lval::Boolean(_) => self,
            Lval::Integer(num) => Lval::boolean(num != 0),
            _ => unreachable!(),
        }
    }

    /// Constructs a new `Lval` holding a string
    pub fn string(string: &str) -> Lval {
        Lval::String(string.to_owned())
    }

    /// Returns a reference to the underlying `String` representation if the type
    /// is `Lval::String`. Will panic otherwise.
    fn as_string(&self) -> &String {
        match self {
            Lval::String(string) => string,
            _ => unreachable!(),
        }
    }

    /// Constructs a new `Lval` holding a `f64`
    pub fn float(val: f64) -> Lval {
        Lval::Float(val)
    }

    /// Returns the underlying number representation if the type is
    /// `Lval::Float`. Returns the underlying integer casted as float
    /// if the type is `Lval::Integer`. Will panic otherwise.
    fn as_float(&self) -> f64 {
        match self {
            Lval::Float(num) => *num,
            Lval::Integer(num) => *num as f64,
            _ => unreachable!(),
        }
    }

    /// Returns `true` if the underlying type is `Lval::Float`
    fn is_float(&self) -> bool {
        match self {
            Lval::Float(_) => true,
            _ => false,
        }
    }
}

/// Built-in functions.
/// All of those functions accept two arguments:
/// - lval. It should always be an S-Expression containing other expressions or operands,
///         the builtin function should work with;
/// - env. A mutable reference to the environment, in some cases may be used to look up
///        a symbol or to define a symbol.

/// Evaluates `lval` and returns `Lval` that holds the evaluation result
pub fn builtin_eval(lval: Lval, env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("eval", lval, 1);
    let expr = lval.take(0);

    lassert_type!("eval", expr, Lval::QExpr(_));

    expr.into_sexpr().eval(env)
}

/// Converts `lval` into a list (Q-Expression)
pub fn builtin_list(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    Ok(lval.into_qexpr())
}

/// Returns a Q-Expression that holds the head of the list, provided in `lval`
pub fn builtin_head(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("head", lval, 1);

    let list = lval.take(0);

    lassert_type!("head", list, Lval::QExpr(_));
    lassert_not_empty!("head", list);

    Ok(Lval::qexpr().add(list.take(0)))
}

/// Returns a Q-Expression that holds the tail of the list, provided in `lval`
pub fn builtin_tail(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("tail", lval, 1);

    let mut list = lval.take(0);

    lassert_type!("tail", list, Lval::QExpr(_));
    lassert_not_empty!("tail", list);

    list.pop(0);
    Ok(list)
}

/// Joins two Q-Expressions together and returns the new `Lval`
pub fn builtin_join(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    for index in 0..lval.len() {
        let expr = lval.peek(index);
        lassert_type!("join", expr, Lval::QExpr(_));
    }

    let mut result = lval.pop(0);
    while !lval.is_empty() {
        let other = lval.pop(0);
        result = result.join(other);
    }

    Ok(result)
}

/// Creates a new `Lval` that holds two Q-Expressions provided in `lval`
pub fn builtin_cons(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("cons", lval, 2);

    let val = lval.pop(0);
    let mut list = lval.pop(0);
    list.insert(val, 0);
    Ok(list)
}

/// Returns `lval` without the last element
pub fn builtin_init(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("init", lval, 1);

    let mut list = lval.pop(0);
    list.pop(list.len() - 1);
    Ok(list)
}

/// Performs checked integer operations where applicable and returns
/// the result in `Lval`
fn integer_op(x: i64, y: i64, op: &'static str) -> Result<Lval> {
    let result;

    match op {
        "+" => {
            result = x.checked_add(y).ok_or_else(|| LispyError::Overflow)?;
        }
        "-" => {
            result = x.checked_sub(y).ok_or_else(|| LispyError::Overflow)?;
        }
        "*" => {
            result = x.checked_mul(y).ok_or_else(|| LispyError::Overflow)?;
        }
        "^" => {
            let y = u32::try_from(y)?;
            result = i64::checked_pow(x, y).ok_or_else(|| LispyError::Overflow)?;
        }
        "min" => {
            result = x.min(y);
        }
        "max" => {
            result = x.max(y);
        }
        "%" => {
            if y == 0 {
                return Err(LispyError::DivisionByZero);
            }
            result = x.checked_rem(y).ok_or_else(|| LispyError::Overflow)?;
        }
        _ => unreachable!(),
    }

    Ok(Lval::integer(result))
}

/// Performs floating point operations where applicable and returns
/// the result in `Lval`
fn float_op(x: f64, y: f64, op: &'static str) -> Result<Lval> {
    let result;

    match op {
        "+" => {
            result = x + y;
        }
        "-" => {
            result = x - y;
        }
        "*" => {
            result = x * y;
        }
        "/" => {
            if y == 0.0f64 {
                return Err(LispyError::DivisionByZero);
            } else {
                result = x / y;
            }
        }
        "^" => {
            result = f64::powf(x, y.into());
        }
        "min" => result = x.min(y),
        "max" => result = x.max(y),
        "%" => {
            if y == 0.0f64 {
                return Err(LispyError::DivisionByZero);
            } else {
                result = x % y;
            }
        }
        _ => unreachable!(),
    }

    Ok(Lval::float(result))
}

fn builtin_op(mut lval: Lval, op: &'static str) -> Result<Lval> {
    // We can only perform integer operations if:
    // - it's not a division;
    // - all operands are integers.
    let mut int_op = if op == "/" { false } else { true };

    for index in 0..lval.len() {
        let number = lval.peek(index);
        lassert_type!(op, number, Lval::Integer(_) | Lval::Float(_));

        // If it least one argument is float - we should work with float.
        if number.is_float() {
            int_op = false;
        }
    }

    let mut result = lval.pop(0);

    if op == "-" && lval.is_empty() {
        if int_op {
            Lval::integer(-result.as_integer())
        } else {
            Lval::float(-result.as_float())
        };
    }

    while !lval.is_empty() {
        let next = lval.pop(0);

        result = if int_op {
            integer_op(result.as_integer(), next.as_integer(), op)?
        } else {
            float_op(result.as_float(), next.as_float(), op)?
        }
    }

    Ok(result)
}

/// Returns a new `Lval` that holds a sum of two numbers provided in `lval`
pub fn builtin_add(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "+")
}

/// Returns a new `Lval` that holds a difference of two numbers provided in `lval`
pub fn builtin_sub(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "-")
}

/// Returns a new `Lval` that holds a product of two numbers provided in `lval`
pub fn builtin_mul(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "*")
}

/// Returns a new `Lval` that holds a quotient of two numbers provided in `lval`
pub fn builtin_div(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "/")
}

/// Returns a new `Lval` that holds a pow(first,second) provided in `lval`
pub fn builtin_pow(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "^")
}

/// Returns a new `Lval` that holds a remainder of two numbers provided in `lval`
pub fn builtin_rem(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "%")
}

/// Returns a new `Lval` that holds a minimum of two numbers provided in `lval`
pub fn builtin_min(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "min")
}

/// Returns a new `Lval` that holds a maximum of two numbers provided in `lval`
pub fn builtin_max(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_op(lval, "max")
}

fn builtin_var(mut lval: Lval, env: &mut SharedEnv, func: &'static str) -> Result<Lval> {
    let mut symbols = lval.pop(0);
    lassert_type!(func, symbols, Lval::QExpr(_));

    for index in 0..symbols.len() {
        let sym = symbols.peek(index);
        lassert_type!(func, sym, Lval::Symbol(_));
    }

    lassert_num!(func, symbols, lval.len());

    while !symbols.is_empty() {
        let sym = symbols.pop(0);
        let arg = lval.pop(0);

        match func {
            "def" => env.borrow_mut().def(sym.as_symbol(), arg),
            "=" => env.borrow_mut().put(sym.as_symbol(), arg),
            _ => unreachable!(),
        }
    }

    Ok(Lval::sexpr())
}

/// Bounds a symbol, described in `lval`, into the global `env` environment. Returns and empty S-Expression.
pub fn builtin_def(lval: Lval, env: &mut SharedEnv) -> Result<Lval> {
    builtin_var(lval, env, "def")
}

/// Bounds a symbol, described in `lval`, into the local `env` environment. Returns and empty S-Expression.
pub fn builtin_put(lval: Lval, env: &mut SharedEnv) -> Result<Lval> {
    builtin_var(lval, env, "=")
}

/// Returns `Lval` that contains a lambda, described in `lval`.
pub fn builtin_lambda(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("\\", lval, 2);

    let formals = lval.pop(0);
    let body = lval.pop(0);

    lassert_type!("\\", formals, Lval::QExpr(_));
    lassert_type!("\\", body, Lval::QExpr(_));

    for index in 0..formals.len() {
        let sym = formals.peek(index);
        lassert_type!("\\", sym, Lval::Symbol(_));
    }

    Ok(Lval::lambda(formals, body))
}

/// Returns `Lval::Integer(1.0)` if the first number, provided in `lval` is bigger than the second.
pub fn builtin_gt(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_ord(lval, ">")
}

/// Returns `Lval::Integer(1.0)` if the first number, provided in `lval` is smaller than the second.
pub fn builtin_lt(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_ord(lval, "<")
}

/// Returns `Lval::Integer(1.0)` if the first number, provided in `lval` is bigger or equal than/to the second.
pub fn builtin_ge(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_ord(lval, ">=")
}

/// Returns `Lval::Integer(1.0)` if the first number, provided in `lval` is less or equal than/to the second.
pub fn builtin_le(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_ord(lval, "<=")
}

fn builtin_ord(mut lval: Lval, ord: &'static str) -> Result<Lval> {
    lassert_num!(ord, lval, 2);

    let first = lval.pop(0);
    lassert_type!(ord, first, Lval::Integer(_));
    let second = lval.pop(0);
    lassert_type!(ord, second, Lval::Integer(_));

    let first = first.as_integer();
    let second = second.as_integer();

    match ord {
        ">" => Ok(Lval::boolean(first > second)),
        "<" => Ok(Lval::boolean(first < second)),
        ">=" => Ok(Lval::boolean(first >= second)),
        "<=" => Ok(Lval::boolean(first <= second)),
        _ => unreachable!(),
    }
}

/// Returns `Lval::Integer(1.0)` if the first `Lval`, provided in `lval` is equal to the second.
pub fn builtin_eq(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_cmp(lval, "==")
}

/// Returns `Lval::Integer(1.0)` if the first `Lval`, provided in `lval` is not equal to the second.
pub fn builtin_ne(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    builtin_cmp(lval, "!=")
}

fn builtin_cmp(mut lval: Lval, op: &'static str) -> Result<Lval> {
    lassert_num!(op, lval, 2);

    let first = lval.pop(0);
    let second = lval.pop(0);

    match op {
        "==" => Ok(Lval::boolean(first == second)),
        "!=" => Ok(Lval::boolean(first != second)),
        _ => unreachable!(),
    }
}

/// Evaluates one of the two expressions, provided in `lval`, depending on the condition provided in `lval`.
pub fn builtin_if(mut lval: Lval, env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("if", lval, 3);

    let cond = lval.pop(0);
    let on_true = lval.pop(0);
    let on_false = lval.pop(0);

    lassert_type!("if", cond, Lval::Integer(_) | Lval::Boolean(_));
    lassert_type!("if", on_true, Lval::QExpr(_));
    lassert_type!("if", on_false, Lval::QExpr(_));

    let cond = cond.into_boolean();
    if *cond.as_boolean() {
        on_true.into_sexpr().eval(env)
    } else {
        on_false.into_sexpr().eval(env)
    }
}

/// Returns Lval::Integer(1.0) if at least one of the two operands, provided in `lval`, evaluates
/// to true
pub fn builtin_or(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("||", lval, 2);

    while !lval.is_empty() {
        let cond = lval.pop(0);
        lassert_type!("or", cond, Lval::Integer(_) | Lval::Boolean(_));

        let cond = cond.into_boolean();
        if *cond.as_boolean() {
            return Ok(cond);
        }
    }

    Ok(Lval::boolean(false))
}

/// Returns Lval::Integer(1.0) if both of the two operands, provided in `lval`, evaluate
/// to true
pub fn builtin_and(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("&&", lval, 2);

    while !lval.is_empty() {
        let cond = lval.pop(0);
        lassert_type!("and", cond, Lval::Integer(_) | Lval::Boolean(_));

        let cond = cond.into_boolean();
        if !*cond.as_boolean() {
            return Ok(cond);
        }
    }

    Ok(Lval::boolean(true))
}

/// Returns a new `Lval` that reverses the logical state of the operand, provided in `lval`
pub fn builtin_not(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("!", lval, 1);

    let cond = lval.pop(0);
    lassert_type!("!", cond, Lval::Integer(_) | Lval::Boolean(_));

    let cond = !cond.into_boolean().as_boolean();

    Ok(Lval::boolean(cond))
}

/// Loads and evaluates a file from the path, passed via `Lval`. Returns an empty S-Expression.
pub fn builtin_load(mut lval: Lval, env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("load", lval, 1);
    let path = lval.pop(0);
    lassert_type!("load", path, Lval::String(_));

    let input = fs::read_to_string(path.as_string())?;
    let mut lval = parser::parse(&input)?;

    for _ in 0..lval.len() {
        lval.pop(0).eval(env)?;
    }

    Ok(Lval::sexpr())
}

/// Outputs its arguments, passed via `lval` to the stdout, using `print!()`
pub fn builtin_print(lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    for index in 0..lval.len() {
        print!("{}", lval.peek(index));
        print!(" ");
    }
    println!("");

    Ok(Lval::sexpr())
}

/// Constructs an error from the first argument and returns it.
pub fn builtin_error(mut lval: Lval, _env: &mut SharedEnv) -> Result<Lval> {
    lassert_num!("error", lval, 1);
    let string = lval.pop(0);
    lassert_type!("error", string, Lval::String(_));

    Err(LispyError::BuiltinError(string.as_string().to_string()))
}

/// Registers all supported built-in functions and types into the provided environment.
pub fn add_builtins(lenv: &mut LEnv) {
    lenv.add_builtin("list", builtin_list);
    lenv.add_builtin("join", builtin_join);
    lenv.add_builtin("tail", builtin_tail);
    lenv.add_builtin("head", builtin_head);
    lenv.add_builtin("init", builtin_init);
    lenv.add_builtin("cons", builtin_cons);
    lenv.add_builtin("eval", builtin_eval);
    lenv.add_builtin("+", builtin_add);
    lenv.add_builtin("-", builtin_sub);
    lenv.add_builtin("/", builtin_div);
    lenv.add_builtin("*", builtin_mul);
    lenv.add_builtin("%", builtin_rem);
    lenv.add_builtin("^", builtin_pow);
    lenv.add_builtin("min", builtin_min);
    lenv.add_builtin("max", builtin_max);
    lenv.add_builtin("=", builtin_put);
    lenv.add_builtin("def", builtin_def);
    lenv.add_builtin("\\", builtin_lambda);
    lenv.add_builtin("==", builtin_eq);
    lenv.add_builtin("!=", builtin_ne);
    lenv.add_builtin(">", builtin_gt);
    lenv.add_builtin("<", builtin_lt);
    lenv.add_builtin(">=", builtin_ge);
    lenv.add_builtin("<=", builtin_le);
    lenv.add_builtin("if", builtin_if);
    lenv.add_builtin("||", builtin_or);
    lenv.add_builtin("&&", builtin_and);
    lenv.add_builtin("!", builtin_not);
    lenv.add_builtin("load", builtin_load);
    lenv.add_builtin("print", builtin_print);
    lenv.add_builtin("error", builtin_error);

    lenv.put("exit", Lval::Exit);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_eval() {
        let mut env = LEnv::new_shared();
        let sexpr = Lval::sexpr().add(Lval::qexpr().add(Lval::integer(1)).add(Lval::integer(2)));

        assert!(builtin_eval(sexpr, &mut env).is_err());

        env.borrow_mut().add_builtin("+", builtin_add);
        let sexpr = Lval::sexpr().add(
            Lval::qexpr()
                .add(Lval::symbol("+"))
                .add(Lval::integer(1))
                .add(Lval::integer(2)),
        );
        assert_eq!(builtin_eval(sexpr, &mut env), Ok(Lval::integer(3)));
    }

    #[test]
    fn test_builtin_list() {
        let mut env = LEnv::new_shared();
        let sexpr = Lval::sexpr().add(Lval::integer(1)).add(Lval::integer(2));

        // Test that an S-Expression gets converted into a QExpr and becomes a list
        assert_eq!(
            builtin_list(sexpr.clone(), &mut env),
            Ok(sexpr.into_qexpr())
        );

        // Test that only Lval other than S-Expression of Q-Expression becomes an
        // empty Q-Expression (list)
        assert_eq!(builtin_list(Lval::integer(42), &mut env), Ok(Lval::qexpr()));
    }

    #[test]
    fn test_builtin_head() {
        let mut env = LEnv::new_shared();

        // Test that 'head' return a list that contains the
        // first element
        let list = Lval::qexpr().add(Lval::integer(1)).add(Lval::integer(2));
        let sexpr = Lval::sexpr().add(list.clone());
        assert_eq!(
            builtin_head(sexpr, &mut env),
            Ok(Lval::qexpr().add(Lval::integer(1)))
        );

        // Test that 'head' return an error if the S-Expression
        // doesn't contain exactly one Q-Expression
        let sexpr = Lval::sexpr();
        assert!(matches!(
            builtin_head(sexpr, &mut env),
            Err(LispyError::InvalidArgNum("head", 1, _))
        ));
        let sexpr = Lval::sexpr().add(Lval::qexpr()).add(Lval::qexpr());
        assert!(matches!(
            builtin_head(sexpr, &mut env),
            Err(LispyError::InvalidArgNum("head", 1, _))
        ));

        // Test that 'head' returns an error if an empty list
        // was provided
        let sexpr = Lval::sexpr().add(Lval::qexpr());
        assert!(matches!(
            builtin_head(sexpr, &mut env),
            Err(LispyError::EmptyList("head"))
        ));
    }

    #[test]
    fn test_builtin_tail() {
        let mut env = LEnv::new_shared();
        let mut list = Lval::qexpr()
            .add(Lval::integer(1))
            .add(Lval::integer(2))
            .add(Lval::integer(3));
        let sexpr = Lval::sexpr().add(list.clone());

        list.pop(0);
        assert_eq!(builtin_tail(sexpr, &mut env), Ok(list));

        // Test that 'tail' return an error if the S-Expression
        // doesn't contain exactly one Q-Expression
        let sexpr = Lval::sexpr();
        assert!(matches!(
            builtin_tail(sexpr, &mut env),
            Err(LispyError::InvalidArgNum("tail", 1, _))
        ));
        let sexpr = Lval::sexpr().add(Lval::qexpr()).add(Lval::qexpr());
        assert!(matches!(
            builtin_tail(sexpr, &mut env),
            Err(LispyError::InvalidArgNum("tail", 1, _))
        ));

        // Test that 'tail' returns an error if an empty list
        // was provided
        let sexpr = Lval::sexpr().add(Lval::qexpr());
        assert!(matches!(
            builtin_tail(sexpr, &mut env),
            Err(LispyError::EmptyList("tail"))
        ));
    }

    #[test]
    fn test_builtin_join() {
        let mut env = LEnv::new_shared();
        let list1 = Lval::qexpr()
            .add(Lval::integer(1))
            .add(Lval::integer(2))
            .add(Lval::integer(3));
        let list2 = Lval::qexpr()
            .add(Lval::integer(4))
            .add(Lval::integer(5))
            .add(Lval::integer(6));
        let sexpr = Lval::sexpr().add(list1.clone()).add(list2.clone());

        let list = list1.join(list2);
        assert_eq!(builtin_join(sexpr, &mut env), Ok(list));

        let sexpr = Lval::sexpr().add(Lval::integer(42));
        assert!(matches!(
            builtin_join(sexpr, &mut env),
            Err(LispyError::InvalidType("join", _, _))
        ));
    }

    #[test]
    fn test_builtin_cons() {
        let mut env = LEnv::new_shared();
        let list = Lval::qexpr()
            .add(Lval::integer(1))
            .add(Lval::integer(2))
            .add(Lval::integer(3));
        let sexpr = Lval::sexpr().add(Lval::integer(1)).add(list.clone());

        let list = Lval::qexpr().add(Lval::integer(1)).join(list);
        assert_eq!(builtin_cons(sexpr, &mut env), Ok(list));

        let sexpr = Lval::sexpr().add(Lval::integer(1));
        assert!(matches!(
            builtin_cons(sexpr, &mut env),
            Err(LispyError::InvalidArgNum("cons", 2, _))
        ));
    }

    #[test]
    fn test_builtin_init() {
        let mut env = LEnv::new_shared();
        let mut list = Lval::qexpr()
            .add(Lval::integer(1))
            .add(Lval::integer(2))
            .add(Lval::integer(3));
        let sexpr = Lval::sexpr().add(list.clone());

        list.pop(list.len() - 1);
        assert_eq!(builtin_init(sexpr, &mut env), Ok(list));

        let sexpr = Lval::sexpr();
        assert!(matches!(
            builtin_init(sexpr, &mut env),
            Err(LispyError::InvalidArgNum("init", 1, _))
        ));
    }

    #[test]
    fn test_builtin_ops() {
        let mut env = LEnv::new_shared();
        let args = Lval::sexpr().add(Lval::integer(2)).add(Lval::integer(4));

        // integers
        assert_eq!(builtin_add(args.clone(), &mut env), Ok(Lval::integer(6)));
        assert_eq!(builtin_sub(args.clone(), &mut env), Ok(Lval::integer(-2)));
        assert_eq!(builtin_mul(args.clone(), &mut env), Ok(Lval::integer(8)));
        assert_eq!(builtin_div(args.clone(), &mut env), Ok(Lval::float(0.5)));
        assert_eq!(builtin_rem(args.clone(), &mut env), Ok(Lval::integer(2)));
        assert_eq!(builtin_pow(args.clone(), &mut env), Ok(Lval::integer(16)));
        assert_eq!(builtin_min(args.clone(), &mut env), Ok(Lval::integer(2)));
        assert_eq!(builtin_max(args.clone(), &mut env), Ok(Lval::integer(4)));

        // float
        let args = Lval::sexpr().add(Lval::integer(2)).add(Lval::float(4.0));
        assert_eq!(builtin_add(args.clone(), &mut env), Ok(Lval::float(6.0)));
        assert_eq!(builtin_sub(args.clone(), &mut env), Ok(Lval::float(-2.0)));
        assert_eq!(builtin_mul(args.clone(), &mut env), Ok(Lval::float(8.0)));
        assert_eq!(builtin_div(args.clone(), &mut env), Ok(Lval::float(0.5)));
        assert_eq!(builtin_rem(args.clone(), &mut env), Ok(Lval::float(2.0)));
        assert_eq!(builtin_pow(args.clone(), &mut env), Ok(Lval::float(16.0)));
        assert_eq!(builtin_min(args.clone(), &mut env), Ok(Lval::float(2.0)));
        assert_eq!(builtin_max(args.clone(), &mut env), Ok(Lval::float(4.0)));

        assert!(matches!(
            builtin_div(
                Lval::sexpr().add(Lval::integer(1)).add(Lval::integer(0)),
                &mut env
            ),
            Err(LispyError::DivisionByZero)
        ));
        assert!(matches!(
            builtin_rem(
                Lval::sexpr().add(Lval::integer(1)).add(Lval::integer(0)),
                &mut env
            ),
            Err(LispyError::DivisionByZero)
        ));

        assert!(matches!(
            builtin_pow(
                Lval::sexpr()
                    .add(Lval::integer(1))
                    .add(Lval::integer(std::i64::MAX)),
                &mut env
            ),
            Err(LispyError::ConversionError(_))
        ));

        assert!(matches!(
            builtin_pow(
                Lval::sexpr()
                    .add(Lval::integer(std::i64::MAX))
                    .add(Lval::integer(std::i32::MAX as i64)),
                &mut env
            ),
            Err(LispyError::Overflow)
        ));
    }

    #[test]
    fn test_builtin_def() {
        let mut env = LEnv::new_shared();
        let sexpr = Lval::sexpr()
            .add(Lval::qexpr().add(Lval::symbol("x")))
            .add(Lval::integer(42));

        assert_eq!(builtin_def(sexpr, &mut env), Ok(Lval::sexpr()));

        let sexpr = Lval::sexpr()
            .add(Lval::qexpr().add(Lval::integer(1)))
            .add(Lval::integer(42));

        assert!(matches!(
            builtin_def(sexpr, &mut env),
            Err(LispyError::InvalidType("def", _, _))
        ));
    }

    #[test]
    fn test_builtin_lambda() {
        let mut env = LEnv::new_shared();
        let formals = Lval::qexpr().add(Lval::symbol("x"));
        let body = Lval::qexpr().add(Lval::qexpr());

        assert!(matches!(
            builtin_lambda(Lval::sexpr(), &mut env),
            Err(LispyError::InvalidArgNum("\\", 2, _))
        ));

        // Formals should be a Q-Expression
        assert!(matches!(
            builtin_lambda(
                Lval::sexpr().add(Lval::integer(1)).add(body.clone()),
                &mut env
            ),
            Err(LispyError::InvalidType("\\", _, _))
        ));

        // Body should be a Q-Expression
        assert!(matches!(
            builtin_lambda(
                Lval::sexpr().add(formals.clone()).add(Lval::integer(1)),
                &mut env
            ),
            Err(LispyError::InvalidType("\\", _, _))
        ));

        let lambda = builtin_lambda(
            Lval::sexpr().add(formals.clone()).add(body.clone()),
            &mut env,
        );
        if let Ok(Lval::Fun(LvalFun::Lambda(lambda))) = lambda {
            assert_eq!(formals, *lambda.formals);
            assert_eq!(body, *lambda.body);
        }
    }

    #[test]
    fn test_builtin_ord() {
        let mut env = LEnv::new_shared();
        let args = Lval::qexpr().add(Lval::integer(2)).add(Lval::integer(1));

        assert_eq!(builtin_gt(args.clone(), &mut env), Ok(Lval::boolean(true)));
        assert_eq!(builtin_lt(args.clone(), &mut env), Ok(Lval::boolean(false)));
        assert_eq!(builtin_ge(args.clone(), &mut env), Ok(Lval::boolean(true)));
        assert_eq!(builtin_le(args.clone(), &mut env), Ok(Lval::boolean(false)));

        assert!(matches!(
            builtin_ge(
                Lval::qexpr()
                    .add(Lval::symbol("x"))
                    .add(Lval::boolean(true)),
                &mut env
            ),
            Err(LispyError::InvalidType(">=", _, _))
        ));
    }

    #[test]
    fn test_builtin_cmp() {
        let mut env = LEnv::new_shared();

        assert_eq!(
            builtin_eq(
                Lval::qexpr().add(Lval::qexpr()).add(Lval::qexpr()),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );
        assert_eq!(
            builtin_eq(
                Lval::qexpr().add(Lval::symbol("x")).add(Lval::symbol("x")),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );
        assert_eq!(
            builtin_eq(
                Lval::qexpr().add(Lval::integer(5)).add(Lval::integer(5)),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );

        assert_eq!(
            builtin_ne(
                Lval::qexpr().add(Lval::integer(1)).add(Lval::integer(2)),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );
        assert_eq!(
            builtin_ne(
                Lval::qexpr().add(Lval::symbol("x")).add(Lval::integer(2)),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );

        assert!(matches!(
            builtin_ne(Lval::qexpr().add(Lval::symbol("x")), &mut env),
            Err(LispyError::InvalidArgNum("!=", 2, _))
        ));
    }

    #[test]
    fn test_builtin_if() {
        let mut env = LEnv::new_shared();

        assert!(matches!(
            builtin_if(Lval::qexpr(), &mut env),
            Err(LispyError::InvalidArgNum("if", 3, _))
        ));

        assert_eq!(
            builtin_if(
                Lval::qexpr()
                    .add(Lval::integer(true as i64))
                    .add(Lval::qexpr().add(Lval::integer(42)))
                    .add(Lval::qexpr()),
                &mut env
            ),
            Ok(Lval::integer(42))
        );
    }

    #[test]
    fn test_builtin_or() {
        let mut env = LEnv::new_shared();

        assert!(matches!(
            builtin_or(Lval::qexpr(), &mut env),
            Err(LispyError::InvalidArgNum("||", 2, _))
        ));

        assert_eq!(
            builtin_or(
                Lval::qexpr()
                    .add(Lval::integer(false as i64))
                    .add(Lval::integer(true as i64)),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );

        assert_eq!(
            builtin_or(
                Lval::qexpr()
                    .add(Lval::boolean(false))
                    .add(Lval::boolean(false)),
                &mut env
            ),
            Ok(Lval::boolean(false))
        );
    }

    #[test]
    fn test_builtin_and() {
        let mut env = LEnv::new_shared();

        assert!(matches!(
            builtin_and(Lval::qexpr(), &mut env),
            Err(LispyError::InvalidArgNum("&&", 2, _))
        ));

        assert_eq!(
            builtin_and(
                Lval::qexpr()
                    .add(Lval::integer(true as i64))
                    .add(Lval::integer(true as i64)),
                &mut env
            ),
            Ok(Lval::boolean(true))
        );

        assert_eq!(
            builtin_and(
                Lval::qexpr()
                    .add(Lval::boolean(true))
                    .add(Lval::boolean(false)),
                &mut env
            ),
            Ok(Lval::boolean(false))
        );
    }

    #[test]
    fn test_builtin_not() {
        let mut env = LEnv::new_shared();

        assert!(matches!(
            builtin_not(Lval::qexpr(), &mut env),
            Err(LispyError::InvalidArgNum("!", 1, _))
        ));

        assert!(matches!(
            builtin_not(Lval::qexpr().add(Lval::symbol("x")), &mut env),
            Err(LispyError::InvalidType("!", _, _))
        ));

        assert_eq!(
            builtin_not(Lval::qexpr().add(Lval::boolean(true)), &mut env),
            Ok(Lval::boolean(false))
        );
    }
}
