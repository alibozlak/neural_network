//! `loss` scores one sample, `cost` scores a whole batch.
//!
//! Both pick their formula from the activation of the output layer: `Sigmoid` gives the
//! binary cross entropy `-[y * ln(p) + (1 - y) * ln(1 - p)]`, `Linear` and `ReLU` give the
//! squared error `(p - y)^2`. Whichever formula is picked, `cost` has to stay the mean of
//! the `loss` of every row of the batch; the sections below pin that down for the cross
//! entropy first and for the squared error last.
//!
//! `cost` takes the batch as an `(sample_count, feature_count)` matrix, so these tests also
//! pin down that every row travels through the network on its own: a batch must never mix
//! two samples, and it must agree with calling `loss` on each row separately.

use ndarray::{array, Array1, Array2};
use neural_network::activation::Activation;

mod common;
use common::{assert_close, binary_cross_entropy, model_of, model_of_activation, sigmoid, EPSILON};

/// One unit of two features: z = 2*x0 - 3*x1 + 0.5
fn two_feature_model() -> neural_network::sequential_model::SequentialModel {
    model_of(vec![array![[2.0, 0.0], [-3.0, 0.0], [0.5, 0.0]]], 2)
}

// ---------------------------------------------------------------- loss

#[test]
fn loss_of_one_sample_is_the_binary_cross_entropy_of_its_prediction() {
    let model = two_feature_model();
    let sample = array![1.0, 2.0];

    let predict = model.predict(&sample);

    assert_close(model.loss(&sample, 1.0), binary_cross_entropy(predict, 1.0));
    assert_close(model.loss(&sample, 0.0), binary_cross_entropy(predict, 0.0));
}

#[test]
fn loss_is_smaller_when_the_prediction_agrees_with_the_real_output() {
    // z = 5.0, so the model is confident that the answer is 1.
    let model = model_of(vec![array![[0.0, 0.0], [5.0, 0.0]]], 1);
    let sample = array![1.0];

    assert!(
        model.loss(&sample, 1.0) < model.loss(&sample, 0.0),
        "agreeing with the model must cost less than contradicting it"
    );
}

#[test]
fn loss_of_a_half_prediction_is_the_logarithm_of_two() {
    // Every weight is zero, so p = 0.5 whatever the input is.
    let model = model_of(vec![array![[0.0, 0.0], [0.0, 0.0]]], 1);

    assert_close(model.loss(&array![4.0], 1.0), 2.0_f64.ln());
    assert_close(model.loss(&array![4.0], 0.0), 2.0_f64.ln());
}

#[test]
fn loss_is_never_negative() {
    let model = two_feature_model();

    for sample in [array![1.0, 2.0], array![-3.0, 0.5], array![0.0, 0.0]] {
        for output in [0.0, 1.0] {
            let loss = model.loss(&sample, output);
            assert!(loss >= 0.0, "got {loss} for {sample} -> {output}");
        }
    }
}

#[test]
fn loss_does_not_change_the_model() {
    let model = two_feature_model();
    let before = model.summary();

    model.loss(&array![1.0, 2.0], 1.0);

    assert_eq!(model.summary(), before);
}

// ---------------------------------------------------------------- cost

#[test]
fn cost_of_a_single_row_batch_is_the_loss_of_that_row() {
    let model = two_feature_model();
    let a0: Array2<f64> = array![[1.0, 2.0]];

    assert_close(
        model.cost(&a0, &array![1.0]),
        model.loss(&array![1.0, 2.0], 1.0),
    );
}

#[test]
fn cost_is_the_mean_of_the_losses_of_every_row() {
    let model = two_feature_model();
    let samples = [array![1.0, 2.0], array![0.0, 1.0], array![-2.0, 4.0]];
    let outputs: Array1<f64> = array![1.0, 0.0, 1.0];

    let mut a0: Array2<f64> = Array2::zeros((3, 2));
    for (row_index, sample) in samples.iter().enumerate() {
        a0.row_mut(row_index).assign(sample);
    }

    let expected: f64 = samples
        .iter()
        .zip(outputs.iter())
        .map(|(sample, &output)| model.loss(sample, output))
        .sum::<f64>()
        / 3.0;

    assert_close(model.cost(&a0, &outputs), expected);
}

#[test]
fn cost_divides_by_the_sample_count_not_by_something_else() {
    // Every row is the same sample with the same output, so the mean must equal the
    // single loss no matter how many rows the batch holds.
    let model = two_feature_model();
    let single = model.loss(&array![1.0, 2.0], 1.0);

    for sample_count in [1, 2, 5, 17] {
        let mut a0: Array2<f64> = Array2::zeros((sample_count, 2));
        for row_index in 0..sample_count {
            a0.row_mut(row_index).assign(&array![1.0, 2.0]);
        }

        assert_close(model.cost(&a0, &Array1::ones(sample_count)), single);
    }
}

#[test]
fn every_row_of_a_batch_keeps_its_own_bias_of_one() {
    // All weights are zero, so only the bias row can reach z and every row must land on
    // sigmoid(7.0). A row whose bias slot held the sample value instead would differ.
    let model = model_of(vec![array![[0.0, 0.0], [7.0, 0.0]]], 1);
    let a0: Array2<f64> = array![[3.0], [99.0], [-40.0]];

    assert_close(
        model.cost(&a0, &array![1.0, 1.0, 1.0]),
        binary_cross_entropy(sigmoid(7.0), 1.0),
    );
}

#[test]
fn a_batch_does_not_let_one_row_change_another() {
    // Rows spread far apart, but not far enough to saturate the sigmoid: if the batch
    // mixed them, splitting it apart would not give the same numbers back.
    let model = two_feature_model();
    let rows = [array![3.0, -2.0], array![0.0, 0.0], array![-2.0, 2.0]];
    let outputs: Array1<f64> = array![1.0, 0.0, 1.0];

    let mut a0: Array2<f64> = Array2::zeros((3, 2));
    for (row_index, sample) in rows.iter().enumerate() {
        a0.row_mut(row_index).assign(sample);
    }
    let together = model.cost(&a0, &outputs);

    let apart: f64 = rows
        .iter()
        .zip(outputs.iter())
        .map(|(sample, &output)| {
            let mut single: Array2<f64> = Array2::zeros((1, 2));
            single.row_mut(0).assign(sample);
            model.cost(&single, &array![output])
        })
        .sum::<f64>()
        / 3.0;

    assert_close(together, apart);
}

#[test]
fn cost_of_a_multi_layer_model_uses_the_output_of_the_last_layer() {
    let hidden = array![
        [1.0, 0.0, 0.0],  // feature 0 -> unit 0
        [0.0, 2.0, 0.0],  // feature 1 -> unit 1
        [0.5, -1.0, 0.0], // bias
    ];
    let output = array![[3.0, 0.0], [-2.0, 0.0], [1.0, 0.0]];
    let model = model_of(vec![hidden, output], 2);

    let a0: Array2<f64> = array![[1.0, 1.0]];
    let predict = sigmoid(sigmoid(1.5) * 3.0 + sigmoid(1.0) * -2.0 + 1.0);

    assert_close(model.cost(&a0, &array![1.0]), binary_cross_entropy(predict, 1.0));
}

#[test]
#[should_panic(expected = "first layer row count mismatch")]
fn cost_rejects_a_batch_whose_columns_are_not_the_sample_features() {
    let model = two_feature_model();
    // The model was built for 2 features, this batch carries 4 of them.
    let a0: Array2<f64> = Array2::zeros((3, 4));

    model.cost(&a0, &array![1.0, 0.0, 1.0]);
}

#[test]
fn cost_does_not_change_the_model() {
    let model = two_feature_model();
    let before = model.summary();

    model.cost(&array![[1.0, 2.0], [0.0, 1.0]], &array![1.0, 0.0]);

    assert_eq!(model.summary(), before);
}

// ------------------------------------------- squared error: Linear and ReLU

/// The activations that share the squared error formula. Every test of this section runs
/// against both of them, because `cost` and `loss` answer them from the same match arm and
/// a mistake in that arm has to fail for either one.
const SQUARED_ERROR: [Activation; 2] = [Activation::Linear, Activation::ReLU];

/// One unit of two features, z = 2*x0 - 3*x1 + 0.5, with the given output activation.
fn squared_error_model(activation: Activation) -> neural_network::sequential_model::SequentialModel {
    model_of_activation(
        vec![array![[2.0, 0.0], [-3.0, 0.0], [0.5, 0.0]]],
        2,
        activation,
    )
}

/// `assert_close` again, naming the activation that failed: the assertions below run once
/// per activation and the line number alone would not say which pass gave the number.
fn assert_close_for(activation: Activation, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < EPSILON,
        "{activation}: expected {expected}, got {actual}"
    );
}

/// The anchor of the whole section: a prediction that lands exactly on the real output
/// costs nothing. `(p + y).abs()`, the shape the ReLU arm was first written in, gives twice
/// the prediction here instead of zero.
#[test]
fn loss_of_a_prediction_that_hits_the_real_output_is_zero() {
    // z = 6.0 - 1.5 + 0.5 = 5.0, positive, so the ReLU does not clamp this one to zero and
    // both activations really do predict a number of their own.
    let sample = array![3.0, 0.5];

    for activation in SQUARED_ERROR {
        let model = squared_error_model(activation);
        let predict = model.predict(&sample);

        assert_close_for(activation, model.loss(&sample, predict), 0.0);
    }
}

/// The contract that `cost` and `loss` have to keep between them, on the smallest batch
/// there is. A `cost` of squared errors next to a `loss` of absolute ones passes the test
/// above and fails this one.
#[test]
fn cost_of_a_single_row_batch_is_the_loss_of_that_row_for_the_squared_error() {
    let sample = array![3.0, 0.5];
    let a0: Array2<f64> = array![[3.0, 0.5]];

    for activation in SQUARED_ERROR {
        let model = squared_error_model(activation);

        assert_close_for(
            activation,
            model.cost(&a0, &array![1.0]),
            model.loss(&sample, 1.0),
        );
    }
}

/// The same contract over a batch that holds more than one row, so that the division by the
/// sample count is checked too.
#[test]
fn cost_is_the_mean_of_the_losses_of_every_row_for_the_squared_error() {
    // The first two rows give a positive z, the last one gives z = -5.5: the ReLU model has
    // to clamp that prediction to zero while the linear one keeps it negative, so the two
    // activations do not walk over the same numbers.
    let samples = [array![3.0, 0.5], array![1.0, 0.0], array![0.0, 2.0]];
    let outputs: Array1<f64> = array![5.0, 2.0, -1.0];

    let mut a0: Array2<f64> = Array2::zeros((3, 2));
    for (row_index, sample) in samples.iter().enumerate() {
        a0.row_mut(row_index).assign(sample);
    }

    for activation in SQUARED_ERROR {
        let model = squared_error_model(activation);

        let expected: f64 = samples
            .iter()
            .zip(outputs.iter())
            .map(|(sample, &output)| model.loss(sample, output))
            .sum::<f64>()
            / 3.0;

        assert_close_for(activation, model.cost(&a0, &outputs), expected);
    }
}
