use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::layer_request_infos::LayerRequestInfo;

#[test]
fn new_keeps_the_given_unit_count() {
    let info = LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 4);

    assert_eq!(info.unit_count, 4);
}

#[test]
fn new_keeps_the_given_activation_function_type() {
    let sigmoid_info = LayerRequestInfo::new(ActivationFunctionType::Sigmoid, 2);
    let linear_info = LayerRequestInfo::new(ActivationFunctionType::Linear, 2);

    assert!(matches!(
        sigmoid_info.activation_function_type,
        ActivationFunctionType::Sigmoid
    ));
    assert!(matches!(
        linear_info.activation_function_type,
        ActivationFunctionType::Linear
    ));
}

#[test]
fn a_layer_can_be_requested_without_any_unit() {
    let info = LayerRequestInfo::new(ActivationFunctionType::Linear, 0);

    assert_eq!(info.unit_count, 0);
}
