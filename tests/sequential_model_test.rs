use neural_network::activation::Activation;
use neural_network::sequential_model::SequentialModel;

mod common;
use common::layer_requests;

#[test]
fn new_creates_one_layer_per_request() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 2, 1]));

    assert_eq!(model.get_layers().len(), 3);
}

#[test]
fn new_accepts_an_empty_layer_request_list() {
    let model = SequentialModel::new(3, &layer_requests(&[]));

    assert!(model.get_layers().is_empty());
}

#[test]
fn new_shapes_the_first_layer_from_the_sample_feature_size() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 2, 1]));

    // (sample_feature_size + 1) rows for the weights plus the bias,
    // (unit_count + 1) columns for the units plus the bias column of the output.
    assert_eq!(model.get_layers()[0].get_matrix().dim(), (4, 5));
}

#[test]
fn new_shapes_every_other_layer_from_the_previous_unit_count() {
    let model = SequentialModel::new(3, &layer_requests(&[4, 2, 1]));

    assert_eq!(model.get_layers()[1].get_matrix().dim(), (5, 3));
    assert_eq!(model.get_layers()[2].get_matrix().dim(), (3, 2));
}

#[test]
fn every_layer_matrix_can_be_multiplied_with_the_output_of_the_previous_one() {
    let model = SequentialModel::new(5, &layer_requests(&[8, 4, 2, 1]));

    let layers = model.get_layers();
    for layer_index in 1..layers.len() {
        assert_eq!(
            layers[layer_index].get_matrix().nrows(),
            layers[layer_index - 1].get_matrix().ncols(),
            "layer {layer_index} cannot consume the output of layer {}",
            layer_index - 1
        );
    }
}

#[test]
fn new_initializes_every_weight_to_zero() {
    let model = SequentialModel::new(3, &layer_requests(&[3, 1]));

    for layer in model.get_layers() {
        assert!(
            layer.get_matrix().iter().all(|&weight| weight == 0.0),
            "a freshly built layer holds a non zero weight"
        );
    }
}

#[test]
fn new_keeps_the_requested_activation_of_every_layer() {
    let model = SequentialModel::new(3, &layer_requests(&[3, 2, 1]));

    for layer in model.get_layers() {
        assert_eq!(layer.get_activation_function(), Activation::Sigmoid);
    }
}

#[test]
fn new_accepts_a_model_with_a_single_layer() {
    let model = SequentialModel::new(2, &layer_requests(&[1]));

    assert_eq!(model.get_layers().len(), 1);
    assert_eq!(model.get_layers()[0].get_matrix().dim(), (3, 2));
}

#[test]
fn new_accepts_a_model_without_any_sample_feature() {
    let model = SequentialModel::new(0, &layer_requests(&[1]));

    // Only the bias row is left.
    assert_eq!(model.get_layers()[0].get_matrix().dim(), (1, 2));
}

#[test]
fn new_accepts_a_layer_without_any_unit() {
    let model = SequentialModel::new(2, &layer_requests(&[0, 1]));

    // Only the bias column is left, so the next layer gets a single row.
    assert_eq!(model.get_layers()[0].get_matrix().dim(), (3, 1));
    assert_eq!(model.get_layers()[1].get_matrix().dim(), (1, 2));
}

#[test]
fn summary_starts_with_the_layer_count() {
    let summary = SequentialModel::new(2, &layer_requests(&[3, 1])).summary();

    assert!(summary.starts_with("Layer_Count: 2\n"), "got: {summary}");
}

#[test]
fn summary_holds_one_line_per_layer_after_the_header() {
    let summary = SequentialModel::new(2, &layer_requests(&[3, 2, 1])).summary();

    assert_eq!(summary.lines().count(), 4);
}

#[test]
fn summary_numbers_the_layers_starting_from_one_in_request_order() {
    let summary = SequentialModel::new(2, &layer_requests(&[3, 2, 1])).summary();

    let layer_lines: Vec<&str> = summary.lines().skip(1).collect();
    for (line_index, line) in layer_lines.iter().enumerate() {
        assert!(
            line.starts_with(&format!("Layer_{}: ", line_index + 1)),
            "got: {line}"
        );
    }
}

#[test]
fn summary_reports_the_shape_and_the_activation_of_every_layer() {
    let summary = SequentialModel::new(2, &layer_requests(&[3, 1])).summary();

    assert_eq!(
        summary,
        "Layer_Count: 2\n\
         Layer_1: Matrix shape = 3x4, activation_func = Sigmoid\n\
         Layer_2: Matrix shape = 4x2, activation_func = Sigmoid\n"
    );
}

#[test]
fn summary_of_a_model_without_layers_only_reports_the_layer_count() {
    let summary = SequentialModel::new(2, &layer_requests(&[])).summary();

    assert_eq!(summary, "Layer_Count: 0\n");
}
