use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::functions::{linear_func, sigmoid_func_using_linear_func};
use neural_network::unit::Unit;

mod common;
use common::{assert_close, linear, sigmoid};

/// Evaluates a unit the way the model does: with its own weights and bias.
fn evaluate(unit: &Unit, input_array: &[f64]) -> f64 {
    (unit.get_activation_function())(input_array, unit.get_weights(), unit.get_bias())
}

#[test]
fn new_keeps_the_given_weights_and_bias() {
    let weights = vec![0.1, 0.2, 0.3];
    let unit = Unit::new(&ActivationFunctionType::Linear, weights.clone(), 1.5);

    assert_eq!(unit.get_weights(), &weights);
    assert_close(unit.get_bias(), 1.5);
}

#[test]
fn linear_type_selects_the_linear_activation_function() {
    let unit = Unit::new(&ActivationFunctionType::Linear, vec![1.0, 2.0], 0.5);

    let input_array = [3.0, 4.0];
    let expected = linear(&input_array, unit.get_weights(), unit.get_bias());

    assert_close(evaluate(&unit, &input_array), expected);
    assert_eq!(
        *unit.get_activation_function() as *const () as usize,
        linear_func as *const () as usize
    );
}

#[test]
fn sigmoid_type_selects_the_sigmoid_activation_function() {
    let unit = Unit::new(&ActivationFunctionType::Sigmoid, vec![1.0, 2.0], 0.5);

    let input_array = [3.0, 4.0];
    let expected = sigmoid(linear(&input_array, unit.get_weights(), unit.get_bias()));

    assert_close(evaluate(&unit, &input_array), expected);
    assert_eq!(
        *unit.get_activation_function() as *const () as usize,
        sigmoid_func_using_linear_func as *const () as usize
    );
}

#[test]
fn activation_functions_of_the_two_types_are_different() {
    let linear_unit = Unit::new(&ActivationFunctionType::Linear, vec![1.0], 0.0);
    let sigmoid_unit = Unit::new(&ActivationFunctionType::Sigmoid, vec![1.0], 0.0);

    let input_array = [2.0];
    assert_close(evaluate(&linear_unit, &input_array), 2.0);
    assert_close(evaluate(&sigmoid_unit, &input_array), sigmoid(2.0));

    assert_ne!(
        *linear_unit.get_activation_function() as *const () as usize,
        *sigmoid_unit.get_activation_function() as *const () as usize
    );
}

#[test]
fn a_unit_can_be_created_without_weights() {
    let unit = Unit::new(&ActivationFunctionType::Linear, vec![], -2.0);

    assert!(unit.get_weights().is_empty());
    assert_close(evaluate(&unit, &[]), -2.0);
}

#[test]
fn getters_expose_the_stored_values_without_changing_them() {
    let unit = Unit::new(&ActivationFunctionType::Sigmoid, vec![1.5, -2.5], 0.75);

    // Reading twice must give the same values: the getters only borrow.
    assert_eq!(unit.get_weights(), &vec![1.5, -2.5]);
    assert_eq!(unit.get_weights(), &vec![1.5, -2.5]);
    assert_close(unit.get_bias(), 0.75);
    assert_close(unit.get_bias(), 0.75);
}
