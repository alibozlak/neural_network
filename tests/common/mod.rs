#![allow(dead_code)]

pub const EPSILON: f64 = 1e-12;

/// Panics with a readable message when two floats are not close enough.
pub fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < EPSILON,
        "expected {expected}, got {actual}"
    );
}

/// Reference implementation of the sigmoid, used to check the crate's version.
pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

/// Reference implementation of the linear combination, used to check the crate's version.
pub fn linear(input_array: &[f64], weights: &[f64], bias: f64) -> f64 {
    let mut output = bias;
    for i in 0..weights.len() {
        output += input_array[i] * weights[i];
    }
    output
}
