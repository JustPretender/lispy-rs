//! Here I just wanted to play with benchmarking different `Lval` implementations.
//! There is no actual value of doing this atm. This code may be removed later.

use criterion::{criterion_group, criterion_main, Criterion};

use ::lispy_rs::lispy::*;

fn bench_builtins(c: &mut Criterion) {
    let mut env = LEnv::new_shared();
    let list = vec![Lval::Number(1), Lval::Number(2), Lval::Number(3)];

    c.bench_function("builtin_list", |bench| {
        bench.iter(|| builtin_list(Lval::SExpr(list.clone()), &mut env))
    });

    c.bench_function("builtin_head", |bench| {
        bench.iter(|| builtin_head(Lval::SExpr(vec![Lval::QExpr(list.clone())]), &mut env))
    });

    c.bench_function("builtin_tail", |bench| {
        bench.iter(|| builtin_tail(Lval::SExpr(vec![Lval::QExpr(list.clone())]), &mut env))
    });

    c.bench_function("builtin_join", |bench| {
        bench.iter(|| {
            builtin_join(
                Lval::SExpr(vec![Lval::QExpr(list.clone()), Lval::QExpr(list.clone())]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_cons", |bench| {
        bench.iter(|| {
            builtin_cons(
                Lval::SExpr(vec![Lval::Number(42), Lval::QExpr(list.clone())]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_init", |bench| {
        bench.iter(|| builtin_init(Lval::SExpr(vec![Lval::QExpr(list.clone())]), &mut env))
    });

    c.bench_function("builtin_add", |bench| {
        bench.iter(|| builtin_add(Lval::SExpr(list.clone()), &mut env))
    });

    c.bench_function("builtin_def", |bench| {
        bench.iter(|| {
            builtin_def(
                Lval::SExpr(vec![
                    Lval::QExpr(vec![Lval::Symbol("x".to_owned())]),
                    Lval::Number(42),
                ]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_lambda", |bench| {
        bench.iter(|| {
            builtin_lambda(
                Lval::SExpr(vec![
                    Lval::QExpr(vec![Lval::Symbol("x".to_owned())]),
                    Lval::QExpr(vec![]),
                ]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_gt", |bench| {
        bench.iter(|| {
            builtin_gt(
                Lval::SExpr(vec![Lval::QExpr(vec![Lval::Number(1), Lval::Number(2)])]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_eq", |bench| {
        bench.iter(|| {
            builtin_eq(
                Lval::QExpr(vec![Lval::Number(1), Lval::Number(2)]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_if", |bench| {
        bench.iter(|| {
            builtin_if(
                Lval::QExpr(vec![
                    Lval::boolean(true),
                    Lval::QExpr(vec![Lval::Number(42)]),
                    Lval::QExpr(vec![]),
                ]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_or", |bench| {
        bench.iter(|| {
            builtin_or(
                Lval::QExpr(vec![Lval::boolean(false), Lval::boolean(true)]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_and", |bench| {
        bench.iter(|| {
            builtin_and(
                Lval::QExpr(vec![Lval::boolean(false), Lval::boolean(true)]),
                &mut env,
            )
        })
    });

    c.bench_function("builtin_not", |bench| {
        bench.iter(|| builtin_not(Lval::QExpr(vec![Lval::boolean(true)]), &mut env))
    });
}

criterion_group!(benches, bench_builtins);
criterion_main!(benches);
