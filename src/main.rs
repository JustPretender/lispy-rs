use std::cell::RefCell;
use std::rc::Rc;

extern crate rustyline;
use rustyline::error::ReadlineError;
use rustyline::Editor;

extern crate pest;
#[macro_use]
extern crate pest_derive;
use pest::iterators::{Pair, Pairs};
use pest::Parser;

#[macro_use]
extern crate log;

use ::lispy_rs::error::*;
use ::lispy_rs::lispy::*;
use ::lispy_rs::*;

#[derive(Parser)]
#[grammar = "lisp-rs.pest"]
struct LispyParser;

fn parse(line: &str) -> Result<Pairs<Rule>> {
    LispyParser::parse(Rule::lispy, line).map_err(|e| LispyError::Grammar(e.to_string()))
}

fn build_ast(pair: Pair<Rule>) -> Result<Lval> {
    let rule = pair.as_rule();
    match rule {
        Rule::number => Ok(Lval::Number(pair.as_str().parse::<i64>()?)),
        Rule::symbol => Ok(Lval::Symbol(pair.as_str().to_owned())),
        Rule::sexpr | Rule::qexpr | Rule::lispy => {
            let mut cells = vec![];
            for inner_pair in pair.into_inner() {
                cells.push(build_ast(inner_pair)?);
            }

            if rule == Rule::sexpr || rule == Rule::lispy {
                Ok(Lval::SExpr(cells))
            } else {
                Ok(Lval::QExpr(cells))
            }
        }
        _ => Err(LispyError::Grammar(format!("Unknown rule: {:?}", rule))),
    }
}

fn main() -> Result<()> {
    env_logger::init();

    info!("Lispy version 0.0.1");
    info!("Press Ctrl-C to Exit");

    let mut rl = Editor::<()>::new();

    if rl.load_history("history.txt").is_err() {
        info!("No previous history.");
    }

    // Initialize environment
    let mut env = Rc::new(RefCell::new(LEnv::default()));

    // Register built-in functions and types
    add_builtins(&mut env.borrow_mut());

    loop {
        // We should first check if were interrupted by the user
        // and treat it as a special non-error case
        let mut line = {
            let read = rl.readline("lispy> ");
            match read {
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
                _ => read,
            }
        }?;

        rl.add_history_entry(line.as_str());

        // Ignore empty inputs
        if line.is_empty() {
            continue;
        }

        // Remove '\n' from the multiline input. Don't think it's very
        // efficient because it creates a copy but we don't care about this
        // in repl.
        line = line.replace("\n", "");

        let result = parse(line.as_str())
            .and_then(|mut tree| {
                build_ast(
                    tree.next()
                        .ok_or_else(|| LispyError::Grammar("No next token".to_owned()))?,
                )
            })
            .and_then(|result| result.eval(&mut env));

        match result {
            Ok(result) => match result {
                Lval::Exit => break,
                _ => println!("{}", result),
            },
            Err(e) => error!("{}", e),
        }
    }

    rl.save_history("history.txt")?;

    Ok(())
}
