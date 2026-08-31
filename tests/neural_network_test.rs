use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::functions::{linear_func, sigmoid_func_using_linear_func};
use neural_network::layer_request_infos::LayerRequestInfo;
use neural_network::neural_network::NeuralNetwork;

mod common;
use common::assert_close;

fn samples() -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let y = vec![0.0, 1.0, 0.0];

    (x, y)
}

fn layer_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(ActivationFunctionType::Sigmoid, unit_count))
        .collect()
}

#[test]
fn validate_returns_layer_count_feature_size_and_sample_size() {
    let (x, y) = samples();

    let (layer_count, feature_size, sample_size) =
        NeuralNetwork::validate(&x, &y, &layer_requests(&[4, 2, 1]));

    assert_eq!(layer_count, 3);
    assert_eq!(feature_size, 3);
    assert_eq!(sample_size, 3);
}

#[test]
fn validate_accepts_a_network_with_a_single_output_layer() {
    let x = vec![vec![1.0]];
    let y = vec![1.0];

    let (layer_count, feature_size, sample_size) =
        NeuralNetwork::validate(&x, &y, &layer_requests(&[1]));

    assert_eq!((layer_count, feature_size, sample_size), (1, 1, 1));
}

#[test]
#[should_panic(expected = "Network must has at least one layer!!")]
fn validate_panics_without_any_layer() {
    let (x, y) = samples();

    NeuralNetwork::validate(&x, &y, &layer_requests(&[]));
}

#[test]
#[should_panic(expected = "Output layer must have only one unit!!")]
fn validate_panics_when_the_output_layer_has_more_than_one_unit() {
    let (x, y) = samples();

    NeuralNetwork::validate(&x, &y, &layer_requests(&[3, 2]));
}

#[test]
#[should_panic(expected = "y and X must have the same sample count!!")]
fn validate_panics_when_x_and_y_sample_counts_differ() {
    let (x, _) = samples();
    let y = vec![0.0, 1.0];

    NeuralNetwork::validate(&x, &y, &layer_requests(&[2, 1]));
}

#[test]
#[should_panic(expected = "X's each sample feature count should be equal!!")]
fn validate_panics_when_samples_have_different_feature_counts() {
    let x = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0, 8.0]];
    let y = vec![0.0, 1.0, 0.0];

    NeuralNetwork::validate(&x, &y, &layer_requests(&[2, 1]));
}

#[test]
#[should_panic]
fn validate_panics_when_x_is_empty() {
    // The feature count is read from the first sample, so an empty X cannot be validated.
    NeuralNetwork::validate(&vec![], &vec![], &layer_requests(&[1]));
}

#[test]
fn new_creates_one_layer_per_request_with_the_requested_unit_counts() {
    let (x, y) = samples();

    let network = NeuralNetwork::new(x, y, layer_requests(&[4, 2, 1]));

    assert_eq!(network.layers.len(), 3);
    assert_eq!(network.layers[0].units.len(), 4);
    assert_eq!(network.layers[1].units.len(), 2);
    assert_eq!(network.layers[2].units.len(), 1);
}

#[test]
fn new_gives_the_first_layer_one_weight_per_feature() {
    let (x, y) = samples();

    let network = NeuralNetwork::new(x, y, layer_requests(&[4, 1]));

    for unit in &network.layers[0].units {
        assert_eq!(unit.weights.len(), 3);
    }
}

#[test]
fn new_gives_every_other_layer_one_weight_per_previous_layer_unit() {
    let (x, y) = samples();

    let network = NeuralNetwork::new(x, y, layer_requests(&[4, 2, 1]));

    for unit in &network.layers[1].units {
        assert_eq!(unit.weights.len(), 4);
    }
    for unit in &network.layers[2].units {
        assert_eq!(unit.weights.len(), 2);
    }
}

#[test]
fn new_initializes_all_weights_and_biases_to_zero() {
    let (x, y) = samples();

    let network = NeuralNetwork::new(x, y, layer_requests(&[3, 1]));

    for layer in &network.layers {
        for unit in &layer.units {
            assert!(unit.weights.iter().all(|&w| w == 0.0));
            assert_close(unit.bias, 0.0);
        }
    }
}

#[test]
fn new_uses_the_activation_function_type_requested_for_each_layer() {
    let (x, y) = samples();
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Linear, 2),
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 1),
    ];

    let network = NeuralNetwork::new(x, y, layer_request_infos);

    for unit in &network.layers[0].units {
        assert_eq!(
            unit.activation_function as *const () as usize,
            linear_func as *const () as usize
        );
    }
    for unit in &network.layers[1].units {
        assert_eq!(
            unit.activation_function as *const () as usize,
            sigmoid_func_using_linear_func as *const () as usize
        );
    }
}

#[test]
fn units_of_a_freshly_created_network_output_their_neutral_value() {
    let (x, y) = samples();
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 2),
        LayerRequestInfo::new(ActivationFunctionType::Linear, 1),
    ];

    let network = NeuralNetwork::new(x, y, layer_request_infos);
    let input_array = [1.0, 2.0, 3.0];

    // All weights and biases are zero: sigmoid units output 0.5, linear units output 0.
    for unit in &network.layers[0].units {
        assert_close(
            (unit.activation_function)(&input_array, &unit.weights, unit.bias),
            0.5,
        );
    }

    let hidden_output = [0.5, 0.5];
    let output_unit = &network.layers[1].units[0];
    assert_close(
        (output_unit.activation_function)(&hidden_output, &output_unit.weights, output_unit.bias),
        0.0,
    );
}

#[test]
fn new_accepts_a_network_made_of_a_single_output_layer() {
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let y = vec![1.0, 0.0];

    let network = NeuralNetwork::new(x, y, layer_requests(&[1]));

    assert_eq!(network.layers.len(), 1);
    assert_eq!(network.layers[0].units.len(), 1);
    assert_eq!(network.layers[0].units[0].weights.len(), 2);
}

#[test]
fn new_creates_an_empty_layer_when_zero_units_are_requested() {
    let (x, y) = samples();

    let network = NeuralNetwork::new(x, y, layer_requests(&[0, 1]));

    assert!(network.layers[0].units.is_empty());
    assert!(network.layers[1].units[0].weights.is_empty());
}
