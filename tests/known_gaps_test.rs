//! Edges that the crate does not handle yet.
//!
//! Two kinds of test live here. The plain ones pin down what happens today, so that a
//! change of behaviour has to be deliberate. The `#[ignore]`d ones describe what *should*
//! happen and fail on purpose until the gap is closed; run them with
//! `cargo test -- --ignored` to see the list.

use ndarray::{array, Array1, Array2};

mod common;
use common::model_of;

/// One unit of two features: z = 2*x0 - 3*x1 + 0.5
fn two_feature_model() -> neural_network::sequential_model::SequentialModel {
    model_of(vec![array![[2.0, 0.0], [-3.0, 0.0], [0.5, 0.0]]], 2)
}

// ------------------------------------------------- a model without any layer

/// A model with no layer at all is refused, which is the useful half of the behaviour.
/// The message is not one of ours though: `validate_layers` reaches `layers[0]` and gets
/// an index out of bounds from the standard library.
#[test]
#[should_panic]
fn an_empty_layer_list_does_not_build_a_model() {
    model_of(vec![], 2);
}

#[test]
#[ignore = "an empty layer list panics with an index out of bounds, not a message of ours"]
#[should_panic(expected = "layer")]
fn an_empty_layer_list_is_refused_with_a_message_that_names_the_mistake() {
    model_of(vec![], 2);
}

// ------------------------------------------------- outputs that do not match the batch

/// Fewer outputs than samples is caught, again by an index out of bounds inside
/// `get_mean_loss` rather than by a check of its own.
#[test]
#[should_panic]
fn cost_does_not_accept_fewer_outputs_than_the_batch_holds() {
    let a0: Array2<f64> = array![[1.0, 2.0], [0.0, 1.0], [1.0, 1.0]];

    two_feature_model().cost(&a0, &array![1.0]);
}

/// More outputs than samples is the dangerous one: nothing complains, the extra outputs
/// are dropped and the mean is taken over the rows that happen to exist.
#[test]
#[ignore = "extra outputs are silently ignored instead of being rejected"]
#[should_panic]
fn cost_does_not_accept_more_outputs_than_the_batch_holds() {
    let a0: Array2<f64> = array![[1.0, 2.0]];

    two_feature_model().cost(&a0, &array![1.0, 0.0, 1.0, 0.0]);
}

// ------------------------------------------------- a saturated sigmoid

/// `Activation::apply` returns exactly 1.0 once z passes about 37, and the loss then takes
/// `ln(0)`. A confidently wrong sample gives `inf`, and a confidently *right* one gives
/// `NaN`, because `(1 - 1) * ln(0)` is `0 * -inf`. Either one poisons the mean of the whole
/// batch, so a single saturated unit is enough to lose a training run.
#[test]
#[ignore = "a saturated sigmoid makes the loss inf or NaN; the prediction needs clamping"]
fn cost_of_a_saturated_prediction_stays_finite() {
    // Only the bias row feeds z, so z = 100 and the prediction is exactly 1.0.
    let model = model_of(vec![array![[0.0, 0.0], [100.0, 0.0]]], 1);
    let a0: Array2<f64> = array![[1.0]];

    let against = model.cost(&a0, &array![0.0]);
    let agreeing = model.cost(&a0, &array![1.0]);

    assert!(against.is_finite(), "a wrong confident prediction gave {against}");
    assert!(agreeing.is_finite(), "a right confident prediction gave {agreeing}");
}

/// The same edge reached through `loss`, one sample at a time.
#[test]
#[ignore = "a saturated sigmoid makes the loss inf or NaN; the prediction needs clamping"]
fn loss_of_a_saturated_prediction_stays_finite() {
    let model = model_of(vec![array![[0.0, 0.0], [100.0, 0.0]]], 1);

    assert!(model.loss(&array![1.0], 0.0).is_finite());
    assert!(model.loss(&array![1.0], 1.0).is_finite());
}

/// The last z that still leaves room for `ln(1 - p)`. Kept as a plain test because it is
/// the boundary the clamping has to move, and it should not drift unnoticed.
#[test]
fn a_prediction_below_the_saturation_point_still_gives_a_finite_loss() {
    let model = model_of(vec![array![[0.0, 0.0], [36.0, 0.0]]], 1);

    let loss = model.loss(&array![1.0], 0.0);

    assert!(loss.is_finite(), "z = 36 already saturates: {loss}");
}

// ------------------------------------------------- a batch without any sample

/// An empty batch divides zero by zero and hands back `NaN` without a word.
#[test]
#[ignore = "an empty batch returns NaN instead of being rejected or defined"]
fn cost_of_an_empty_batch_is_not_a_silent_nan() {
    let a0: Array2<f64> = Array2::zeros((0, 2));

    let cost = two_feature_model().cost(&a0, &Array1::zeros(0));

    assert!(!cost.is_nan(), "an empty batch gave {cost}");
}
