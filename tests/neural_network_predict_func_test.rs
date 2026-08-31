use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::layer::Layer;
use neural_network::layer_request_infos::LayerRequestInfo;
use neural_network::neural_network::NeuralNetwork;
use neural_network::unit::Unit;

mod common;
use common::{assert_close, linear, sigmoid};

fn linear_unit(weights: Vec<f64>, bias: f64) -> Unit {
    Unit::new(&ActivationFunctionType::Linear, weights, bias)
}

fn sigmoid_unit(weights: Vec<f64>, bias: f64) -> Unit {
    Unit::new(&ActivationFunctionType::Sigmoid, weights, bias)
}

/// Builds a network straight from its units, so every weight and bias is known.
fn network(layers: Vec<Vec<Unit>>) -> NeuralNetwork {
    NeuralNetwork {
        layers: layers.into_iter().map(Layer::new).collect(),
    }
}

fn samples() -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let y = vec![0.0, 1.0, 0.0];

    (x, y)
}

fn layer_requests(
    unit_counts: &[usize],
    activation_function_type: fn() -> ActivationFunctionType,
) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(activation_function_type(), unit_count))
        .collect()
}

#[test]
fn predicts_the_output_of_a_single_linear_unit() {
    let network = network(vec![vec![linear_unit(vec![0.5, -1.0, 2.0], 0.25)]]);

    // 1 * 0.5 + 2 * -1 + 3 * 2 + 0.25
    assert_close(network.network_union_predict_function(&vec![1.0, 2.0, 3.0]), 4.75);
}

#[test]
fn predicts_the_output_of_a_single_sigmoid_unit() {
    let input_sample = vec![1.0, 2.0, 3.0];
    let weights = vec![0.5, -1.0, 2.0];
    let bias = 0.25;
    let network = network(vec![vec![sigmoid_unit(weights.clone(), bias)]]);

    assert_close(
        network.network_union_predict_function(&input_sample),
        sigmoid(linear(&input_sample, &weights, bias)),
    );
}

#[test]
fn feeds_the_whole_input_sample_to_every_unit_of_the_first_layer() {
    let input_sample = vec![1.0, 2.0];
    let hidden_weights = [vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, -1.0]];
    let hidden_biases = [0.5, -0.5, 1.0];
    let output_weights = vec![2.0, 3.0, 4.0];
    let output_bias = -1.0;

    let network = network(vec![
        vec![
            linear_unit(hidden_weights[0].clone(), hidden_biases[0]),
            linear_unit(hidden_weights[1].clone(), hidden_biases[1]),
            linear_unit(hidden_weights[2].clone(), hidden_biases[2]),
        ],
        vec![linear_unit(output_weights.clone(), output_bias)],
    ]);

    let hidden_outputs: Vec<f64> = (0..3)
        .map(|u| linear(&input_sample, &hidden_weights[u], hidden_biases[u]))
        .collect();
    let expected = linear(&hidden_outputs, &output_weights, output_bias);

    // hidden = [1.5, 1.5, -2.0], output = 1.5 * 2 + 1.5 * 3 + -2 * 4 + -1
    assert_close(expected, -1.5);
    assert_close(network.network_union_predict_function(&input_sample), expected);
}

#[test]
fn feeds_the_outputs_of_a_layer_as_the_inputs_of_the_next_one() {
    let input_sample = vec![2.0, -3.0];
    // The single hidden unit doubles the first feature, the output unit adds ten to it.
    let network = network(vec![
        vec![linear_unit(vec![2.0, 0.0], 0.0)],
        vec![linear_unit(vec![1.0], 10.0)],
    ]);

    assert_close(network.network_union_predict_function(&input_sample), 14.0);
}

#[test]
fn chains_layers_that_use_different_activation_function_types() {
    let input_sample = vec![1.0, -2.0];
    let hidden_weights = [vec![0.5, 1.5], vec![-2.0, 0.25]];
    let hidden_biases = [0.75, -1.25];
    let output_weights = vec![1.5, -0.5];
    let output_bias = 0.125;

    let network = network(vec![
        vec![
            sigmoid_unit(hidden_weights[0].clone(), hidden_biases[0]),
            sigmoid_unit(hidden_weights[1].clone(), hidden_biases[1]),
        ],
        vec![linear_unit(output_weights.clone(), output_bias)],
    ]);

    let hidden_outputs: Vec<f64> = (0..2)
        .map(|u| sigmoid(linear(&input_sample, &hidden_weights[u], hidden_biases[u])))
        .collect();

    assert_close(
        network.network_union_predict_function(&input_sample),
        linear(&hidden_outputs, &output_weights, output_bias),
    );
}

#[test]
fn propagates_through_more_than_two_layers() {
    let input_sample = vec![1.0, 1.0];
    // Each layer only forwards a sum, so the value grows in a predictable way.
    let network = network(vec![
        vec![linear_unit(vec![1.0, 1.0], 1.0), linear_unit(vec![1.0, 1.0], 2.0)],
        vec![linear_unit(vec![1.0, 1.0], 0.0)],
        vec![linear_unit(vec![2.0], -3.0)],
    ]);

    // layer 1 = [3.0, 4.0], layer 2 = [7.0], layer 3 = 7 * 2 - 3
    assert_close(network.network_union_predict_function(&input_sample), 11.0);
}

#[test]
fn returns_the_value_of_the_first_unit_of_the_last_layer() {
    let network = network(vec![vec![
        linear_unit(vec![1.0], 0.0),
        linear_unit(vec![100.0], 100.0),
    ]]);

    assert_close(network.network_union_predict_function(&vec![3.0]), 3.0);
}

#[test]
fn a_sigmoid_output_layer_keeps_the_prediction_between_zero_and_one() {
    let network = network(vec![vec![sigmoid_unit(vec![0.5], 0.0)]]);

    let big = network.network_union_predict_function(&vec![20.0]);
    let small = network.network_union_predict_function(&vec![-20.0]);

    assert!(big > 0.0 && big < 1.0, "sigmoid output out of range: {big}");
    assert!(small > 0.0 && small < 1.0, "sigmoid output out of range: {small}");
    assert!(big > small);
    assert_close(big, sigmoid(10.0));
    assert_close(small, sigmoid(-10.0));
}

#[test]
fn predicts_zero_for_a_freshly_created_network_whose_output_layer_is_linear() {
    let (x, y) = samples();
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 4),
        LayerRequestInfo::new(ActivationFunctionType::Linear, 1),
    ];

    let network = NeuralNetwork::new(&x, &y, &layer_request_infos);

    // All weights and biases are zero, so the linear output unit answers with its bias.
    assert_close(network.network_union_predict_function(&x[0]), 0.0);
}

#[test]
fn predicts_one_half_for_a_freshly_created_network_whose_output_layer_is_sigmoid() {
    let (x, y) = samples();

    let network = NeuralNetwork::new(&x, &y, &layer_requests(&[4, 2, 1], || {
        ActivationFunctionType::Sigmoid
    }));

    assert_close(network.network_union_predict_function(&x[2]), 0.5);
}

#[test]
fn predicts_with_the_weights_and_biases_set_on_a_created_network() {
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let y = vec![0.0, 1.0];
    let mut network = NeuralNetwork::new(&x, &y, &layer_requests(&[2, 1], || {
        ActivationFunctionType::Linear
    }));

    network.layers[0].units[0].weights = vec![1.0, 0.0];
    network.layers[0].units[0].bias = 1.0;
    network.layers[0].units[1].weights = vec![0.0, 2.0];
    network.layers[0].units[1].bias = -1.0;
    network.layers[1].units[0].weights = vec![3.0, 1.0];
    network.layers[1].units[0].bias = 0.5;

    // hidden = [2.0, 3.0], output = 2 * 3 + 3 * 1 + 0.5
    assert_close(network.network_union_predict_function(&x[0]), 9.5);
}

#[test]
fn every_sample_of_the_training_set_can_be_predicted() {
    let (x, y) = samples();
    let mut network = NeuralNetwork::new(&x, &y, &layer_requests(&[1], || {
        ActivationFunctionType::Linear
    }));
    network.layers[0].units[0].weights = vec![1.0, 1.0, 1.0];

    let predictions: Vec<f64> = x
        .iter()
        .map(|sample| network.network_union_predict_function(sample))
        .collect();

    assert_close(predictions[0], 6.0);
    assert_close(predictions[1], 15.0);
    assert_close(predictions[2], 24.0);
}

#[test]
fn accepts_an_input_sample_without_any_feature_when_the_first_layer_expects_none() {
    let network = network(vec![vec![linear_unit(vec![], 4.5)]]);

    assert_close(network.network_union_predict_function(&vec![]), 4.5);
}

#[test]
fn repeated_calls_return_the_same_value_and_leave_the_input_sample_untouched() {
    let input_sample = vec![1.0, 2.0, 3.0];
    let network = network(vec![
        vec![sigmoid_unit(vec![0.1, 0.2, 0.3], 0.4), sigmoid_unit(vec![-0.5, 0.6, -0.7], 0.8)],
        vec![linear_unit(vec![1.0, -1.0], 0.5)],
    ]);

    let first = network.network_union_predict_function(&input_sample);
    let second = network.network_union_predict_function(&input_sample);

    assert_close(second, first);
    assert_eq!(input_sample, vec![1.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "Input sample feature size must be 3 count !! Yours = 2")]
fn panics_when_the_input_sample_has_too_few_features() {
    let (x, y) = samples();
    let network = NeuralNetwork::new(&x, &y, &layer_requests(&[2, 1], || {
        ActivationFunctionType::Sigmoid
    }));

    network.network_union_predict_function(&vec![1.0, 2.0]);
}

#[test]
#[should_panic(expected = "Input sample feature size must be 3 count !! Yours = 4")]
fn panics_when_the_input_sample_has_too_many_features() {
    let (x, y) = samples();
    let network = NeuralNetwork::new(&x, &y, &layer_requests(&[2, 1], || {
        ActivationFunctionType::Sigmoid
    }));

    network.network_union_predict_function(&vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic(expected = "Input sample feature size must be 0 count !! Yours = 1")]
fn panics_when_a_feature_is_given_to_a_first_layer_that_expects_none() {
    let network = network(vec![vec![linear_unit(vec![], 0.0)]]);

    network.network_union_predict_function(&vec![1.0]);
}

#[test]
#[should_panic]
fn panics_when_the_first_layer_has_no_unit() {
    // The expected feature count is read from the first unit of the first layer,
    // so an empty first layer cannot be used to predict.
    let network = network(vec![vec![], vec![linear_unit(vec![], 0.0)]]);

    network.network_union_predict_function(&vec![1.0]);
}
