use std::cell::RefCell;
use std::rc::Rc;

extern crate rustyline;
use rustyline::error::ReadlineError;
use rustyline::Editor;

#[macro_use]
extern crate log;

use ::lispy_rs::lispy::*;
use ::lispy_rs::parser::*;
use ::lispy_rs::*;

/// Loads and evaluates files supplied in `files`
/// TODO think of a better name. This one is inspired by Emacs.
fn headless(files: Vec<String>, env: &mut Rc<RefCell<LEnv>>) -> Result<()> {
    for file in files.iter() {
        Lval::sexpr()
            .add(Lval::symbol("load"))
            .add(Lval::string(&file))
            .eval(env)
            .map(|_| ())?;
    }

    Ok(())
}

/// Enters a Lispy REPL
fn repl(env: &mut Rc<RefCell<LEnv>>) -> Result<()> {
    info!("Lispy version 0.0.1");
    info!("Press Ctrl-C to Exit");

    let mut rl = Editor::<()>::new();
    if rl.load_history("history.txt").is_err() {
        info!("No previous history.");
    }

    loop {
        // We should first check if were interrupted by the user
        // and treat it as a special non-error case
        let line = {
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

        let result = parse(line.as_str()).and_then(|result| result.eval(env));

        // We can't use ? operator because we don't want to treat evaluation errors as
        // fatal. We don't want to terminate the REPL, if eval() fails - we want to
        // print the error and continue.
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

fn main() -> Result<()> {
    env_logger::init();

    // Initialize environment
    let mut env = Rc::new(RefCell::new(LEnv::default()));
    // Register built-in functions and types
    add_builtins(&mut env.borrow_mut());

    // Check if user supplied any files to evaluate. If no - enter the REPL
    // if yes - evaluate them one by one and exit;
    let files = std::env::args().skip(1).collect::<Vec<String>>();
    if files.is_empty() {
        repl(&mut env)
    } else {
        headless(files, &mut env)
    }
}
