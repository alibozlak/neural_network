use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::layer_request_infos::LayerRequestInfo;
use neural_network::sequential_model::SequentialModel;

mod common;
use common::{assert_close, sigmoid};

fn layer_requests(
    unit_counts: &[usize],
    activation_function_type: fn() -> ActivationFunctionType,
) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(activation_function_type(), unit_count))
        .collect()
}

fn sigmoid_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    layer_requests(unit_counts, || ActivationFunctionType::Sigmoid)
}

fn linear_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    layer_requests(unit_counts, || ActivationFunctionType::Linear)
}

/// Walks the model layer by layer, the way a forward pass is supposed to work,
/// so the result can be compared with the one of predict_function.
fn forward_pass(model: &SequentialModel, input_sample: &Vec<f64>) -> f64 {
    let mut inputs = input_sample.clone();

    for layer in model.get_layers() {
        inputs = layer
            .get_units()
            .iter()
            .map(|unit| {
                (unit.get_activation_function())(&inputs, unit.get_weights(), unit.get_bias())
            })
            .collect();
    }

    inputs[0]
}

#[test]
fn predicts_the_bias_of_a_single_linear_unit() {
    let model = SequentialModel::new(3, &linear_requests(&[1]));

    // Every weight and bias of a fresh model is zero.
    assert_close(model.predict_function(&vec![1.0, 2.0, 3.0]), 0.0);
}

#[test]
fn predicts_one_half_for_a_single_sigmoid_unit() {
    let model = SequentialModel::new(3, &sigmoid_requests(&[1]));

    assert_close(model.predict_function(&vec![1.0, 2.0, 3.0]), sigmoid(0.0));
}

#[test]
fn predicts_zero_when_the_output_layer_of_a_fresh_model_is_linear() {
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 4),
        LayerRequestInfo::new(ActivationFunctionType::Linear, 1),
    ];

    let model = SequentialModel::new(3, &layer_request_infos);

    // The hidden layer answers 0.5 everywhere, but the output weights are zero.
    assert_close(model.predict_function(&vec![1.0, 2.0, 3.0]), 0.0);
}

#[test]
fn predicts_one_half_when_the_output_layer_of_a_fresh_model_is_sigmoid() {
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Linear, 4),
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 1),
    ];

    let model = SequentialModel::new(3, &layer_request_infos);

    assert_close(model.predict_function(&vec![7.0, 8.0, 9.0]), 0.5);
}

#[test]
fn propagates_through_more_than_two_layers() {
    let model = SequentialModel::new(3, &sigmoid_requests(&[5, 4, 3, 2, 1]));

    assert_close(model.predict_function(&vec![1.0, 2.0, 3.0]), 0.5);
}

#[test]
fn feeds_the_outputs_of_a_layer_as_the_inputs_of_the_next_one() {
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 3),
        LayerRequestInfo::new(ActivationFunctionType::Linear, 2),
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 1),
    ];
    let model = SequentialModel::new(4, &layer_request_infos);
    let input_sample = vec![1.0, -2.0, 3.0, -4.0];

    // A layer only ever sees as many inputs as it has weights, so a wrong chaining
    // would make the prediction differ from the hand written forward pass.
    assert_close(
        model.predict_function(&input_sample),
        forward_pass(&model, &input_sample),
    );
}

#[test]
fn matches_a_manual_forward_pass_for_every_layer_shape() {
    for unit_counts in [
        vec![1],
        vec![2, 1],
        vec![4, 2, 1],
        vec![3, 3, 3, 1],
    ] {
        let model = SequentialModel::new(2, &sigmoid_requests(&unit_counts));
        let input_sample = vec![0.5, -1.5];

        assert_close(
            model.predict_function(&input_sample),
            forward_pass(&model, &input_sample),
        );
    }
}

#[test]
fn returns_the_value_of_the_first_unit_of_the_last_layer() {
    // The last layer holds two units here, and only the first one is returned.
    let model = SequentialModel::new(2, &sigmoid_requests(&[3, 2]));
    let input_sample = vec![1.0, 2.0];

    assert_close(
        model.predict_function(&input_sample),
        forward_pass(&model, &input_sample),
    );
}

#[test]
fn a_sigmoid_output_layer_keeps_the_prediction_between_zero_and_one() {
    let model = SequentialModel::new(1, &sigmoid_requests(&[2, 1]));

    for feature in [-100.0, -1.0, 0.0, 1.0, 100.0] {
        let prediction = model.predict_function(&vec![feature]);
        assert!(
            prediction > 0.0 && prediction < 1.0,
            "prediction {prediction} is out of the (0, 1) range"
        );
    }
}

#[test]
fn every_sample_of_a_data_set_can_be_predicted() {
    let samples = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let model = SequentialModel::new(3, &sigmoid_requests(&[2, 1]));

    let predictions: Vec<f64> = samples
        .iter()
        .map(|sample| model.predict_function(sample))
        .collect();

    // A fresh model has no weight at all, so it answers the same value everywhere.
    assert_eq!(predictions.len(), 3);
    for prediction in predictions {
        assert_close(prediction, 0.5);
    }
}

#[test]
fn accepts_an_input_sample_without_any_feature_when_the_model_expects_none() {
    let model = SequentialModel::new(0, &linear_requests(&[1]));

    assert_close(model.predict_function(&vec![]), 0.0);
}

#[test]
fn repeated_calls_return_the_same_value_and_leave_the_input_sample_untouched() {
    let input_sample = vec![1.0, 2.0, 3.0];
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 2),
        LayerRequestInfo::new(ActivationFunctionType::Linear, 1),
    ];
    let model = SequentialModel::new(3, &layer_request_infos);

    let first = model.predict_function(&input_sample);
    let second = model.predict_function(&input_sample);

    assert_close(second, first);
    assert_eq!(input_sample, vec![1.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "Input sample feature size must be 3 count !! Yours = 2")]
fn panics_when_the_input_sample_has_too_few_features() {
    let model = SequentialModel::new(3, &sigmoid_requests(&[2, 1]));

    model.predict_function(&vec![1.0, 2.0]);
}

#[test]
#[should_panic(expected = "Input sample feature size must be 3 count !! Yours = 4")]
fn panics_when_the_input_sample_has_too_many_features() {
    let model = SequentialModel::new(3, &sigmoid_requests(&[2, 1]));

    model.predict_function(&vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic(expected = "Input sample feature size must be 0 count !! Yours = 1")]
fn panics_when_a_feature_is_given_to_a_model_that_expects_none() {
    let model = SequentialModel::new(0, &linear_requests(&[1]));

    model.predict_function(&vec![1.0]);
}

#[test]
#[should_panic]
fn panics_when_the_first_layer_has_no_unit() {
    // The expected feature count is read from the first unit of the first layer,
    // so a model whose first layer is empty cannot predict.
    let model = SequentialModel::new(2, &sigmoid_requests(&[0, 1]));

    model.predict_function(&vec![1.0, 2.0]);
}

#[test]
#[should_panic]
fn panics_when_the_model_has_no_layer_at_all() {
    let model = SequentialModel::new(2, &sigmoid_requests(&[]));

    model.predict_function(&vec![1.0, 2.0]);
}
