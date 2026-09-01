use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::layer_request_infos::LayerRequestInfo;
use neural_network::validates::validate;

fn samples() -> (Vec<Vec<f64>>, Vec<f64>) {
    let samples = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let y = vec![0.0, 1.0, 0.0];

    (samples, y)
}

fn layer_requests(unit_counts: &[usize]) -> Vec<LayerRequestInfo> {
    unit_counts
        .iter()
        .map(|&unit_count| LayerRequestInfo::new(ActivationFunctionType::Sigmoid, unit_count))
        .collect()
}

#[test]
fn validate_returns_the_layer_count_and_the_sample_feature_size() {
    let (samples, y) = samples();

    let (layer_count, sample_feature_size) = validate(&samples, &y, &layer_requests(&[4, 2, 1]));

    assert_eq!(layer_count, 3);
    assert_eq!(sample_feature_size, 3);
}

#[test]
fn validate_accepts_a_network_made_of_a_single_output_layer() {
    let samples = vec![vec![1.0]];
    let y = vec![1.0];

    assert_eq!(validate(&samples, &y, &layer_requests(&[1])), (1, 1));
}

#[test]
fn validate_accepts_a_hidden_layer_without_any_unit() {
    let (samples, y) = samples();

    assert_eq!(validate(&samples, &y, &layer_requests(&[0, 1])), (2, 3));
}

#[test]
fn validate_accepts_samples_without_any_feature() {
    let samples = vec![vec![], vec![]];
    let y = vec![0.0, 1.0];

    assert_eq!(validate(&samples, &y, &layer_requests(&[1])), (1, 0));
}

#[test]
#[should_panic(expected = "Network must has at least one layer!!")]
fn validate_panics_without_any_layer() {
    let (samples, y) = samples();

    validate(&samples, &y, &layer_requests(&[]));
}

#[test]
#[should_panic(expected = "Output layer must have only one unit!!")]
fn validate_panics_when_the_output_layer_has_more_than_one_unit() {
    let (samples, y) = samples();

    validate(&samples, &y, &layer_requests(&[3, 2]));
}

#[test]
#[should_panic(expected = "Output layer must have only one unit!!")]
fn validate_panics_when_the_output_layer_has_no_unit() {
    let (samples, y) = samples();

    validate(&samples, &y, &layer_requests(&[3, 0]));
}

#[test]
#[should_panic(expected = "y and samples (X) must have the same sample count!!")]
fn validate_panics_when_the_sample_and_y_counts_differ() {
    let (samples, _) = samples();
    let y = vec![0.0, 1.0];

    validate(&samples, &y, &layer_requests(&[2, 1]));
}

#[test]
#[should_panic(expected = "samples's each sample feature count should be equal!!")]
fn validate_panics_when_samples_have_different_feature_counts() {
    let samples = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0, 7.0, 8.0]];
    let y = vec![0.0, 1.0, 0.0];

    validate(&samples, &y, &layer_requests(&[2, 1]));
}

#[test]
#[should_panic(expected = "samples's each sample feature count should be equal!!")]
fn validate_checks_the_feature_count_of_the_last_sample_too() {
    let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0]];
    let y = vec![0.0, 1.0, 0.0];

    validate(&samples, &y, &layer_requests(&[1]));
}

#[test]
#[should_panic]
fn validate_panics_when_there_is_no_sample_at_all() {
    // The feature count is read from the first sample, so an empty sample list
    // cannot be validated.
    validate(&vec![], &vec![], &layer_requests(&[1]));
}
