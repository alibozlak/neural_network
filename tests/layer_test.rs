use ndarray::{array, Array2};
use neural_network::activation::Activation;
use neural_network::layer::Layer;

mod common;
use common::assert_close;

#[test]
fn new_stores_the_given_matrix_without_changing_it() {
    let matrix: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let layer = Layer::new(matrix.clone(), Activation::Sigmoid);

    assert_eq!(layer.get_matrix(), &matrix);
}

#[test]
fn new_keeps_the_matrix_shape() {
    let layer = Layer::new(Array2::zeros((5, 3)), Activation::Sigmoid);

    assert_eq!(layer.get_matrix().nrows(), 5);
    assert_eq!(layer.get_matrix().ncols(), 3);
}

#[test]
fn new_stores_the_given_activation() {
    let layer = Layer::new(Array2::zeros((2, 2)), Activation::Sigmoid);

    assert!(layer.get_activation_function() == Activation::Sigmoid);
}

#[test]
fn the_stored_activation_can_be_applied_to_a_whole_matrix() {
    let layer = Layer::new(array![[0.0, 0.0], [0.0, 0.0]], Activation::Sigmoid);

    let activated = layer
        .get_matrix()
        .mapv(|z| layer.get_activation_function().apply(z));

    for output in activated.iter() {
        assert_close(*output, 0.5);
    }
}

#[test]
fn get_matrix_gives_access_to_single_weights_by_row_and_column() {
    let layer = Layer::new(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], Activation::Sigmoid);

    assert_close(layer.get_matrix()[[0, 0]], 1.0);
    assert_close(layer.get_matrix()[[2, 1]], 6.0);
}

#[test]
fn summary_reports_the_raw_matrix_shape_and_the_activation() {
    let summary = Layer::new(Array2::zeros((3, 4)), Activation::Sigmoid).summary();

    assert_eq!(summary, "Matrix shape = 3x4, activation_func = Sigmoid");
}

#[test]
fn summary_prints_the_row_count_before_the_column_count() {
    let summary = Layer::new(Array2::zeros((7, 2)), Activation::Sigmoid).summary();

    assert!(summary.contains("7x2"), "got: {summary}");
}

/// The shape is reported raw, bias row and bias column included. A layer of 2 incoming
/// features and 3 units is a `(3, 4)` matrix and prints as `3x4`, so a reader has to know
/// the convention to get back to "2 features, 3 units".
#[test]
fn summary_does_not_subtract_the_bias_row_and_column() {
    let layer = Layer::new(Array2::zeros((2 + 1, 3 + 1)), Activation::Sigmoid);

    assert!(layer.summary().contains("3x4"), "got: {}", layer.summary());
}
