#![allow(dead_code)]

use ndarray::Array2;
use neural_network::activation::Activation;
use neural_network::layer::Layer;
use neural_network::layer_request_infos::LayerRequestInfo;
use neural_network::sequential_model::SequentialModel;

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

/// Reference binary cross entropy, used to check the crate's loss.
pub fn binary_cross_entropy(predict: f64, real_output: f64) -> f64 {
    -(real_output * predict.ln() + (1. - real_output) * (1. - predict).ln())
}

/// Builds one sigmoid layer request per given unit count.
pub fn layer_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(Activation::Sigmoid, unit_count))
        .collect()
}

/// Builds a model from hand written layer matrices, all of them sigmoid.
///
/// `new` fills every weight with zero, which hides any arithmetic mistake behind
/// `sigmoid(0) = 0.5`; the tests that check real numbers need layers they choose themselves.
pub fn model_of(matrices: Vec<Array2<f64>>, sample_feature_size: usize) -> SequentialModel {
    model_of_activation(matrices, sample_feature_size, Activation::Sigmoid)
}

/// The same thing as `model_of`, for an activation the caller picks.
///
/// `cost` and `loss` read the activation of the *output* layer to choose their formula, so
/// a test of anything but the cross entropy needs a model that is not sigmoid.
pub fn model_of_activation(
    matrices: Vec<Array2<f64>>,
    sample_feature_size: usize,
    activation: Activation,
) -> SequentialModel {
    let layers = matrices
        .into_iter()
        .map(|matrix| Layer::new(matrix, activation))
        .collect();

    SequentialModel::generate_sequential_model_with_layers(layers, sample_feature_size)
}
