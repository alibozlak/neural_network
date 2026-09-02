#![allow(dead_code)]

use neural_network::activation::Activation;
use neural_network::layer_request_infos::LayerRequestInfo;

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

/// Builds one sigmoid layer request per given unit count.
pub fn layer_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(Activation::Sigmoid, unit_count))
        .collect()
}
