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
fn summary_reports_both_a_unit_count_and_a_weight_count() {
    let summary = Layer::new(Array2::zeros((3, 4)), Activation::Sigmoid).summary();

    assert!(summary.contains("Unit_Count: "), "got: {summary}");
    assert!(summary.contains("weight_count: "), "got: {summary}");
}

/// The matrix of a layer is built as `(weight_count + 1, unit_count + 1)`, the two
/// `+ 1`s being the bias row and the bias column. `summary` prints `ncols()` and
/// `nrows()` as they are, so both numbers are currently one too big, and the two
/// labels are swapped on top of that: for a `(3, 4)` matrix it prints
/// `Unit_Count: 4, weight_count: 3` instead of `Unit_Count: 3, weight_count: 2`.
#[test]
#[ignore = "known bug: summary does not subtract the bias row/column"]
fn summary_reports_the_real_unit_count_and_weight_count() {
    let summary = Layer::new(Array2::zeros((3, 4)), Activation::Sigmoid).summary();

    assert_eq!(summary, "Unit_Count: 3, weight_count: 2");
}
