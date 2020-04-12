#![crate_name = "lispy_rs"]

#[macro_use]
extern crate log;

extern crate pest;
#[macro_use]
extern crate pest_derive;

pub mod error;
pub mod lispy;
pub mod parser;

pub type Result<T> = std::result::Result<T, error::LispyError>;
