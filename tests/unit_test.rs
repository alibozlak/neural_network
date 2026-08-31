use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::functions::{linear_func, sigmoid_func_using_linear_func};
use neural_network::unit::Unit;

mod common;
use common::{assert_close, linear, sigmoid};

#[test]
fn new_keeps_the_given_weights_and_bias() {
    let weights = vec![0.1, 0.2, 0.3];
    let unit = Unit::new(&ActivationFunctionType::Linear, weights.clone(), 1.5);

    assert_eq!(unit.weights, weights);
    assert_close(unit.bias, 1.5);
}

#[test]
fn linear_type_selects_the_linear_activation_function() {
    let unit = Unit::new(&ActivationFunctionType::Linear, vec![1.0, 2.0], 0.5);

    let input_array = [3.0, 4.0];
    let expected = linear(&input_array, &unit.weights, unit.bias);

    assert_close(
        (unit.activation_function)(&input_array, &unit.weights, unit.bias),
        expected,
    );
    assert_eq!(
        unit.activation_function as *const () as usize,
        linear_func as *const () as usize
    );
}

#[test]
fn sigmoid_type_selects_the_sigmoid_activation_function() {
    let unit = Unit::new(&ActivationFunctionType::Sigmoid, vec![1.0, 2.0], 0.5);

    let input_array = [3.0, 4.0];
    let expected = sigmoid(linear(&input_array, &unit.weights, unit.bias));

    assert_close(
        (unit.activation_function)(&input_array, &unit.weights, unit.bias),
        expected,
    );
    assert_eq!(
        unit.activation_function as *const () as usize,
        sigmoid_func_using_linear_func as *const () as usize
    );
}

#[test]
fn activation_functions_of_the_two_types_are_different() {
    let linear_unit = Unit::new(&ActivationFunctionType::Linear, vec![1.0], 0.0);
    let sigmoid_unit = Unit::new(&ActivationFunctionType::Sigmoid, vec![1.0], 0.0);

    assert_ne!(
        linear_unit.activation_function as *const () as usize,
        sigmoid_unit.activation_function as *const () as usize
    );
}

#[test]
fn a_unit_can_be_created_without_weights() {
    let unit = Unit::new(&ActivationFunctionType::Linear, vec![], -2.0);

    assert!(unit.weights.is_empty());
    assert_close(
        (unit.activation_function)(&[], &unit.weights, unit.bias),
        -2.0,
    );
}
