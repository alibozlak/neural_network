//! `loss` scores one sample, `cost` scores a whole batch.
//!
//! Both pick their formula from the activation of the output layer; with only `Sigmoid`
//! implemented that formula is the binary cross entropy
//! `-[y * ln(p) + (1 - y) * ln(1 - p)]`.
//!
//! `cost` takes the batch as an `(sample_count, feature_count)` matrix, so these tests also
//! pin down that every row travels through the network on its own: a batch must never mix
//! two samples, and it must agree with calling `loss` on each row separately.

use ndarray::{array, Array1, Array2};

mod common;
use common::{assert_close, binary_cross_entropy, model_of, sigmoid};

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
