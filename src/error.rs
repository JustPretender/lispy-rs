use rustyline::error::ReadlineError;
use std::error;
use std::fmt;
use std::num::ParseIntError;

/// The error type for Lispy errors that can arise from
/// parsing, input, and evaluation errors
#[derive(Debug, Clone, PartialEq)]
pub enum LispyError {
    /// Failed to parse an integer
    ParseInt(ParseIntError),
    /// Readline crate error
    Readline(String),
    /// Failed to tokenize input
    Grammar(String),
    /// Lispy built-in function was passed invalid number of arguments
    InvalidArgNum(&'static str, usize, usize),
    /// Lispy expression has invalid format
    InvalidFormat(&'static str),
    /// Lispy built-in function was passed invalid argument type(s)
    InvalidType(&'static str, &'static str, String),
    /// Lispy built-in was expecting a non-empty Q/S-expression
    EmptyList(&'static str),
    /// Lispy built-in attempted division by zero
    DivisionByZero,
    /// Lispy attempted to evaluate an unbound symbol
    UnboundSymbol(String),
}

impl fmt::Display for LispyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LispyError::ParseInt(e) => e.fmt(f),
            LispyError::Readline(e) => write!(f, "{}", e),
            LispyError::Grammar(e) => write!(f, "Grammar: {}", e),
            LispyError::InvalidArgNum(func, expected, given) => write!(
                f,
                "{} received too many arguments. Expected {}. Given {}!",
                func, expected, given
            ),
            LispyError::InvalidFormat(message) => write!(f, "Invalid function format. {}", message),
            LispyError::InvalidType(func, expected, given) => write!(
                f,
                "{} passed invalid type. Expected {}. Given {}!",
                func, expected, given
            ),
            LispyError::DivisionByZero => write!(f, "Attempted to divide by 0!"),
            LispyError::EmptyList(func) => write!(f, "{} passed an empty list!", func),
            LispyError::UnboundSymbol(symbol) => write!(f, "Unbound symbol: {}!", symbol),
        }
    }
}

impl error::Error for LispyError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
}

impl From<ParseIntError> for LispyError {
    fn from(err: ParseIntError) -> LispyError {
        LispyError::ParseInt(err)
    }
}

impl From<ReadlineError> for LispyError {
    fn from(err: ReadlineError) -> LispyError {
        LispyError::Readline(format!("{}", err))
    }
}
