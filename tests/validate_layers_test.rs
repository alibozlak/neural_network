//! `generate_sequential_model_with_layers` hands the layers to the model as they come, so
//! it is the only place that can still catch a stack whose matrices cannot be multiplied.
//!
//! The rule it enforces: layer 0 needs `sample_feature_size + 1` rows (the features plus
//! the bias), and every later layer needs as many rows as the previous layer has columns.
//! A model that breaks either one would otherwise fail much later, inside `predict`, with
//! ndarray's own shape error instead of a message naming the layer at fault.

use ndarray::{array, Array2};

mod common;
use common::model_of;

#[test]
fn a_chain_whose_layers_multiply_is_accepted() {
    let model = model_of(vec![Array2::zeros((3, 4)), Array2::zeros((4, 2))], 2);

    assert_eq!(model.get_layers().len(), 2);
}

#[test]
fn a_long_chain_is_checked_layer_by_layer() {
    let model = model_of(
        vec![
            Array2::zeros((3, 5)),
            Array2::zeros((5, 3)),
            Array2::zeros((3, 2)),
        ],
        2,
    );

    assert_eq!(model.get_layers().len(), 3);
}

#[test]
#[should_panic(expected = "Layer_2 row size and its previous column size mismatch")]
fn a_second_layer_that_cannot_consume_the_first_one_is_rejected() {
    // Layer 1 outputs 4 columns, so layer 2 needs 4 rows and gets 3.
    model_of(vec![Array2::zeros((3, 4)), Array2::zeros((3, 2))], 2);
}

#[test]
#[should_panic(expected = "Layer_3 row size and its previous column size mismatch")]
fn a_break_deep_in_the_chain_names_the_layer_at_fault() {
    model_of(
        vec![
            Array2::zeros((3, 5)),
            Array2::zeros((5, 3)),
            Array2::zeros((7, 2)),
        ],
        2,
    );
}

#[test]
#[should_panic(expected = "first layer row count mismatch")]
fn a_first_layer_that_does_not_match_the_sample_features_is_rejected() {
    // 2 features need 3 rows, not 9.
    model_of(vec![Array2::zeros((9, 4))], 2);
}

#[test]
#[should_panic(expected = "first layer row count mismatch")]
fn a_first_layer_without_a_bias_row_is_rejected() {
    // 2 rows would hold the two features but leave no room for the bias.
    model_of(vec![Array2::zeros((2, 4))], 2);
}

#[test]
fn a_single_layer_model_only_has_its_first_layer_checked() {
    let model = model_of(vec![Array2::zeros((3, 2))], 2);

    assert_eq!(model.get_layers().len(), 1);
}

#[test]
fn a_model_without_any_sample_feature_needs_a_single_bias_row() {
    let model = model_of(vec![Array2::zeros((1, 2))], 0);

    assert_eq!(model.get_layers()[0].get_matrix().dim(), (1, 2));
}

#[test]
fn the_layers_reach_the_model_in_the_given_order_and_unchanged() {
    let first = array![[1.5, -2.0], [0.25, 3.0]];
    let second = array![[0.5, 1.0], [4.0, -1.0]];

    let model = model_of(vec![first.clone(), second.clone()], 1);

    assert_eq!(model.get_layers()[0].get_matrix(), &first);
    assert_eq!(model.get_layers()[1].get_matrix(), &second);
}

/// The shape that `new` produces must pass the validator, or the two constructors disagree
/// about what a well formed model is.
#[test]
fn the_shapes_that_new_produces_are_accepted_by_the_validator() {
    use common::layer_requests;
    use neural_network::sequential_model::SequentialModel;

    let built = SequentialModel::new(5, &layer_requests(&[8, 4, 2, 1]));
    let matrices: Vec<Array2<f64>> = built
        .get_layers()
        .iter()
        .map(|layer| layer.get_matrix().clone())
        .collect();

    let rebuilt = model_of(matrices, 5);

    assert_eq!(rebuilt.get_layers().len(), 4);
}
