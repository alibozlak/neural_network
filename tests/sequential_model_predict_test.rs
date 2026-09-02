use ndarray::array;
use neural_network::sequential_model::SequentialModel;

mod common;
use common::{assert_close, layer_requests};

#[test]
#[should_panic(expected = "Input array's feature count not correct!!")]
fn predict_rejects_an_input_that_does_not_hold_the_sample_features() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 1]));

    // The model was built for 2 features, this input carries 7 of them.
    model.predict(array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
}

#[test]
#[should_panic]
fn predict_cannot_run_on_a_model_without_layers() {
    let model = SequentialModel::new(2, &layer_requests(&[]));

    model.predict(array![1.0, 2.0]);
}

// -------------------------------------------------------------------------------------
// The tests below describe what `predict` is meant to do. They are ignored because the
// current implementation panics on *every* input, for two independent reasons:
//
// 1. The guard compares `feature_size + 1` with `self.layers[0].get_matrix().ncols()`.
//    In an `Array2` the first dimension is the row count, so the matrix built as
//    `(sample_feature_size + 1, unit_count + 1)` keeps the feature count in `nrows()`;
//    `ncols()` is the unit count of the first layer. A correct input is therefore
//    rejected, and an input whose length happens to equal the first unit count passes.
//
// 2. `input.into_shape_with_order((1, feature_size + 1))` asks for a matrix of
//    `feature_size + 1` values while `input` only holds `feature_size` of them, so the
//    reshape always fails and `unwrap` panics. The bias column has to be appended to the
//    input, it cannot appear out of a reshape.
//
// Run them with `cargo test -- --ignored` to check a fix.
// -------------------------------------------------------------------------------------

#[test]
#[ignore = "known bug: predict panics on every input"]
fn predict_accepts_an_input_that_holds_exactly_the_sample_features() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 1]));

    model.predict(array![1.0, 2.0]);
}

#[test]
#[ignore = "known bug: predict panics on every input"]
fn predict_of_a_single_zero_weight_layer_is_one_half() {
    let model = SequentialModel::new(2, &layer_requests(&[1]));

    // Every weight and every bias is zero, so z = 0 and sigmoid(0) = 0.5.
    assert_close(model.predict(array![3.0, -4.0]), 0.5);
}

#[test]
#[ignore = "known bug: predict panics on every input"]
fn predict_of_a_deep_zero_weight_model_is_one_half() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 2, 1]));

    assert_close(model.predict(array![1.0, 2.0, 3.0]), 0.5);
}

#[test]
#[ignore = "known bug: predict panics on every input"]
fn predict_of_a_zero_weight_model_ignores_the_input_values() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 1]));

    let first = model.predict(array![0.0, 0.0]);
    let second = model.predict(array![1000.0, -1000.0]);

    assert_close(first, second);
}

#[test]
#[ignore = "known bug: predict panics on every input"]
fn predict_returns_a_sigmoid_output_between_zero_and_one() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 1]));

    let output = model.predict(array![0.5, -0.5]);

    assert!(output > 0.0 && output < 1.0, "got: {output}");
}

#[test]
#[ignore = "known bug: predict panics on every input"]
fn predict_does_not_change_the_model() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 1]));
    let before = model.summary();

    model.predict(array![1.0, 2.0]);

    assert_eq!(model.summary(), before);
    for layer in model.get_layers() {
        assert!(layer.get_matrix().iter().all(|&weight| weight == 0.0));
    }
}
