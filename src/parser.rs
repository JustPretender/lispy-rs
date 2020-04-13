use pest::iterators::Pair;
use pest::Parser;

use crate::error::LispyError;
use crate::lispy::Lval;
use crate::Result;

#[derive(Parser)]
#[grammar = "lisp-rs.pest"]
struct LispyParser;

/// Tokenizes input `str` and builts an AST represented as an `Lval`.
pub fn parse(input: &str) -> Result<Lval> {
    let mut tree =
        LispyParser::parse(Rule::lispy, input).map_err(|e| LispyError::Grammar(e.to_string()))?;

    build_ast(
        tree.next()
            .ok_or_else(|| LispyError::Grammar("No next token".to_owned()))?,
    )
}

/// Builds an AST (`Lval`) from PEST `Pair`s.
fn build_ast(pair: Pair<Rule>) -> Result<Lval> {
    let rule = pair.as_rule();
    match rule {
        Rule::integer => Ok(Lval::integer(pair.as_str().parse::<i64>()?)),
        Rule::float => Ok(Lval::float(pair.as_str().parse::<f64>()?)),
        Rule::symbol => Ok(Lval::symbol(pair.as_str())),
        Rule::string => {
            let quoted = pair.as_str();
            // Strip "" and construct an Lval
            Ok(Lval::string(&quoted[1..quoted.len() - 1]))
        }
        Rule::sexpr | Rule::qexpr | Rule::lispy => {
            // Create a parent expression depending on the parser rule
            let mut expr = {
                if rule == Rule::sexpr || rule == Rule::lispy {
                    Lval::sexpr()
                } else {
                    Lval::qexpr()
                }
            };

            // Recurse into inner pairs and fill in the parent expression
            for inner_pair in pair.into_inner() {
                expr = expr.add(build_ast(inner_pair)?);
            }

            Ok(expr)
        }
        _ => Err(LispyError::Grammar(format!("Unknown rule: {:?}", rule))),
    }
}
