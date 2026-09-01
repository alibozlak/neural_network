use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::functions::{linear_func, sigmoid_func_using_linear_func};
use neural_network::layer_request_infos::LayerRequestInfo;
use neural_network::sequential_model::SequentialModel;
use neural_network::unit::Unit;

mod common;
use common::{assert_close, sigmoid};

fn layer_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(ActivationFunctionType::Sigmoid, unit_count))
        .collect()
}

fn activation_function_address(unit: &Unit) -> usize {
    *unit.get_activation_function() as *const () as usize
}

#[test]
fn new_creates_one_layer_per_request_with_the_requested_unit_counts() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 2, 1]));

    let layers = model.get_layers();
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].get_units().len(), 4);
    assert_eq!(layers[1].get_units().len(), 2);
    assert_eq!(layers[2].get_units().len(), 1);
}

#[test]
fn new_gives_the_first_layer_one_weight_per_sample_feature() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 1]));

    for unit in model.get_layers()[0].get_units() {
        assert_eq!(unit.get_weights().len(), 3);
    }
}

#[test]
fn new_gives_every_other_layer_one_weight_per_previous_layer_unit() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 2, 1]));

    for unit in model.get_layers()[1].get_units() {
        assert_eq!(unit.get_weights().len(), 4);
    }
    for unit in model.get_layers()[2].get_units() {
        assert_eq!(unit.get_weights().len(), 2);
    }
}

#[test]
fn new_initializes_all_weights_and_biases_to_zero() {
    let model = SequentialModel::new(3, &layer_requests(&[3, 1]));

    for layer in model.get_layers() {
        for unit in layer.get_units() {
            assert!(unit.get_weights().iter().all(|&weight| weight == 0.0));
            assert_close(unit.get_bias(), 0.0);
        }
    }
}

#[test]
fn new_uses_the_activation_function_type_requested_for_each_layer() {
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Linear, 2),
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 1),
    ];

    let model = SequentialModel::new(3, &layer_request_infos);

    for unit in model.get_layers()[0].get_units() {
        assert_eq!(
            activation_function_address(unit),
            linear_func as *const () as usize
        );
    }
    for unit in model.get_layers()[1].get_units() {
        assert_eq!(
            activation_function_address(unit),
            sigmoid_func_using_linear_func as *const () as usize
        );
    }
}

#[test]
fn units_of_a_freshly_created_model_output_their_neutral_value() {
    let layer_request_infos = vec![
        LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 2),
        LayerRequestInfo::new(ActivationFunctionType::Linear, 1),
    ];

    let model = SequentialModel::new(3, &layer_request_infos);
    let input_array = [1.0, 2.0, 3.0];

    // All weights and biases are zero: sigmoid units output 0.5, linear units output 0.
    for unit in model.get_layers()[0].get_units() {
        let output = (unit.get_activation_function())(&input_array, unit.get_weights(), unit.get_bias());
        assert_close(output, sigmoid(0.0));
    }

    let hidden_output = [0.5, 0.5];
    let output_unit = &model.get_layers()[1].get_units()[0];
    let output = (output_unit.get_activation_function())(
        &hidden_output,
        output_unit.get_weights(),
        output_unit.get_bias(),
    );
    assert_close(output, 0.0);
}

#[test]
fn new_accepts_a_model_made_of_a_single_output_layer() {
    let model = SequentialModel::new(2, &layer_requests(&[1]));

    let layers = model.get_layers();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].get_units().len(), 1);
    assert_eq!(layers[0].get_units()[0].get_weights().len(), 2);
}

#[test]
fn new_creates_an_empty_layer_when_zero_units_are_requested() {
    let model = SequentialModel::new(3, &layer_requests(&[0, 1]));

    assert!(model.get_layers()[0].get_units().is_empty());
    // The next layer takes one weight per unit of the previous one, so it gets none.
    assert!(model.get_layers()[1].get_units()[0].get_weights().is_empty());
}

#[test]
fn new_accepts_a_sample_feature_size_of_zero() {
    let model = SequentialModel::new(0, &layer_requests(&[2, 1]));

    for unit in model.get_layers()[0].get_units() {
        assert!(unit.get_weights().is_empty());
    }
    assert_eq!(model.get_layers()[1].get_units()[0].get_weights().len(), 2);
}

#[test]
fn new_without_any_layer_request_creates_a_model_without_any_layer() {
    // SequentialModel::new does not validate: the check belongs to validates::validate.
    let model = SequentialModel::new(3, &layer_requests(&[]));

    assert!(model.get_layers().is_empty());
}

#[test]
fn new_does_not_require_the_output_layer_to_have_a_single_unit() {
    // Again, that rule is enforced by validates::validate, not by the constructor.
    let model = SequentialModel::new(3, &layer_requests(&[2, 5]));

    assert_eq!(model.get_layers()[1].get_units().len(), 5);
}

#[test]
fn every_unit_of_a_layer_gets_its_own_copy_of_the_weights() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 1]));

    let units = model.get_layers()[0].get_units();
    assert_eq!(units.len(), 3);
    for unit in units {
        assert_eq!(unit.get_weights(), &vec![0.0, 0.0]);
    }
}

#[test]
fn summary_reports_the_layer_count_and_the_unit_count_of_every_layer() {
    let model = SequentialModel::new(2, &layer_requests(&[3, 2, 1]));

    // FixMe: "Unit_Count" shows up twice on each line, because SequentialModel::summary
    // already writes it and Layer::summary writes it again.
    assert_eq!(
        model.summary(),
        "Layer_Count: 3\n\
         Layer_1_Unit_Count: Unit_Count: 3\n\
         Layer_2_Unit_Count: Unit_Count: 2\n\
         Layer_3_Unit_Count: Unit_Count: 1\n"
    );
}

#[test]
fn summary_numbers_the_layers_starting_from_one() {
    let model = SequentialModel::new(1, &layer_requests(&[4, 1]));
    let summary = model.summary();

    assert!(summary.contains("Layer_1_"), "missing first layer: {summary}");
    assert!(summary.contains("Layer_2_"), "missing second layer: {summary}");
    assert!(!summary.contains("Layer_0_"), "layers start at 0: {summary}");
    assert!(!summary.contains("Layer_3_"), "one layer too many: {summary}");
}

#[test]
fn summary_of_a_model_without_any_layer_only_reports_the_layer_count() {
    let model = SequentialModel::new(3, &layer_requests(&[]));

    assert_eq!(model.summary(), "Layer_Count: 0\n");
}

#[test]
fn summary_counts_a_layer_without_any_unit_too() {
    let model = SequentialModel::new(3, &layer_requests(&[0, 1]));

    assert_eq!(
        model.summary(),
        "Layer_Count: 2\n\
         Layer_1_Unit_Count: Unit_Count: 0\n\
         Layer_2_Unit_Count: Unit_Count: 1\n"
    );
}

#[test]
fn get_layers_returns_the_layers_in_the_requested_order() {
    let model = SequentialModel::new(5, &layer_requests(&[7, 3, 1]));

    let unit_counts: Vec<usize> = model
        .get_layers()
        .iter()
        .map(|layer| layer.get_units().len())
        .collect();

    assert_eq!(unit_counts, vec![7, 3, 1]);
}
