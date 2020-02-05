#![crate_name = "lispy_rs"]

#[macro_use]
extern crate log;

pub mod error;
pub mod lispy;

pub type Result<T> = std::result::Result<T, error::LispyError>;
