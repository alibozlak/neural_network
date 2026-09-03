//! Checks the arithmetic `predict` actually performs.
//!
//! Every other predict test runs on a model built by `SequentialModel::new`, whose weights
//! are all zero; there `z` is zero everywhere and the answer is 0.5 whatever the matrices
//! do. These tests hand-build the layers instead, so a wrong multiplication shows up.
//!
//! The layout under test: a layer matrix is `(weight_count + 1, unit_count + 1)`, column
//! `j` holds the weights of unit `j` with the bias in the last row, and the input reaches
//! the first layer as `[x0, .., xn-1, 1]`.

use ndarray::{array, Array1};

mod common;
use common::{assert_close, model_of, sigmoid};

#[test]
fn predict_multiplies_every_feature_with_the_weight_of_its_own_row() {
    // One unit of 2 features: column 0 is [w0, w1, bias].
    let model = model_of(vec![array![[2.0, 0.0], [-3.0, 0.0], [0.5, 0.0]]], 2);

    // z = 1.0 * 2.0 + 2.0 * (-3.0) + 1.0 * 0.5 = -3.5
    assert_close(model.predict(&array![1.0, 2.0]), sigmoid(-3.5));
}

#[test]
fn predict_does_not_swap_the_features() {
    // Weights differ by an order of magnitude, so reading the rows in the wrong order
    // lands on sigmoid(10.0) instead of sigmoid(1.0).
    let model = model_of(vec![array![[10.0, 0.0], [1.0, 0.0], [0.0, 0.0]]], 2);

    // z = 0.0 * 10.0 + 1.0 * 1.0 + 1.0 * 0.0 = 1.0
    assert_close(model.predict(&array![0.0, 1.0]), sigmoid(1.0));
}

#[test]
fn predict_adds_the_bias_of_the_last_row_exactly_once() {
    // Both weights are zero, so only the bias row can reach z. The input value is 3.0,
    // so a bias slot filled with the input instead of 1.0 would give sigmoid(15.0).
    let model = model_of(vec![array![[0.0, 0.0], [5.0, 0.0]]], 1);

    // z = 3.0 * 0.0 + 1.0 * 5.0 = 5.0
    assert_close(model.predict(&array![3.0]), sigmoid(5.0));
}

#[test]
fn predict_reads_the_units_from_the_columns() {
    // Unit 0 looks at feature 0, unit 1 at feature 1. predict returns unit 0, so a layer
    // read row wise instead of column wise would answer with the other unit.
    let model = model_of(
        vec![array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        2,
    );

    // z of unit 0 = 2.0 * 1.0 + 7.0 * 0.0 + 1.0 * 0.0 = 2.0  (unit 1 would be 7.0)
    assert_close(model.predict(&array![2.0, 7.0]), sigmoid(2.0));
}

#[test]
fn predict_feeds_the_activations_of_a_layer_into_the_next_one() {
    let model = model_of(
        vec![
            array![[3.0, 0.0], [-1.0, 0.0]], // layer 1: one unit of one feature
            array![[2.0, 0.0], [0.5, 0.0]],  // layer 2: one unit of one feature
        ],
        1,
    );

    // z1 = 1.0 * 3.0 + 1.0 * (-1.0) = 2.0        -> a1 = sigmoid(2.0)
    // z2 = a1 * 2.0 + 1.0 * 0.5
    let expected = sigmoid(2.0 * sigmoid(2.0) + 0.5);

    assert_close(model.predict(&array![1.0]), expected);
}

#[test]
fn predict_gives_the_next_layer_a_bias_of_one_not_the_previous_activation() {
    let model = model_of(
        vec![
            array![[0.0, 0.0], [0.0, 0.0]], // layer 1: z = 0, so a1 = 0.5
            array![[0.0, 0.0], [7.0, 0.0]], // layer 2: only the bias row can reach z
        ],
        1,
    );

    // z2 = 0.5 * 0.0 + 1.0 * 7.0 = 7.0; a bias of 0.5 would give sigmoid(3.5)
    assert_close(model.predict(&array![4.0]), sigmoid(7.0));
}

#[test]
fn predict_of_a_deep_model_chains_every_layer() {
    // Three layers, each an identity on z: weight 1.0, bias 0.0.
    let model = model_of(
        vec![
            array![[1.0, 0.0], [0.0, 0.0]],
            array![[1.0, 0.0], [0.0, 0.0]],
            array![[1.0, 0.0], [0.0, 0.0]],
        ],
        1,
    );

    let expected = sigmoid(sigmoid(sigmoid(2.0)));

    assert_close(model.predict(&array![2.0]), expected);
}

#[test]
fn predict_drops_the_units_after_the_first_one_of_the_output_layer() {
    // The output layer holds two units with very different weights; predict answers with
    // unit 0 and never mentions unit 1.
    let model = model_of(vec![array![[1.0, 9.0, 0.0], [0.0, 0.0, 0.0]]], 1);

    assert_close(model.predict(&array![1.0]), sigmoid(1.0));
}

#[test]
#[should_panic(expected = "first layer row count mismatch")]
fn predict_rejects_an_input_that_does_not_match_the_hand_built_first_layer() {
    let model = model_of(vec![array![[1.0, 0.0], [0.0, 0.0]]], 1);

    // The layer expects 1 feature; 3 of them must not be accepted.
    model.predict(&array![1.0, 2.0, 3.0]);
}

#[test]
fn predict_of_a_model_without_features_still_uses_the_bias() {
    // No feature at all: the input is empty and only the bias row feeds z.
    let model = model_of(vec![array![[2.5, 0.0]]], 0);

    assert_close(model.predict(&Array1::zeros(0)), sigmoid(2.5));
}

#[test]
fn predict_carries_a_wide_hidden_layer_into_a_single_output_unit() {
    // Two hidden units with different weights, so a layer that collapses them or reads
    // the wrong column cannot reach the same answer.
    let hidden = array![
        [1.0, 0.0, 0.0],  // feature 0 -> unit 0
        [0.0, 2.0, 0.0],  // feature 1 -> unit 1
        [0.5, -1.0, 0.0], // bias
    ];
    let output = array![[3.0, 0.0], [-2.0, 0.0], [1.0, 0.0]];
    let model = model_of(vec![hidden, output], 2);

    // z_h0 = 1.0 * 1.0 + 1.0 * 0.0 + 1.0 * 0.5  = 1.5
    // z_h1 = 1.0 * 0.0 + 1.0 * 2.0 + 1.0 * -1.0 = 1.0
    // z_out = a_h0 * 3.0 + a_h1 * (-2.0) + 1.0 * 1.0
    let expected = sigmoid(sigmoid(1.5) * 3.0 + sigmoid(1.0) * -2.0 + 1.0);

    assert_close(model.predict(&array![1.0, 1.0]), expected);
}
