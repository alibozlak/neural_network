use neural_network::functions::{linear_func, sigmoid_func, sigmoid_func_using_linear_func};

mod common;
use common::{assert_close, linear, sigmoid};

#[test]
fn linear_func_computes_weighted_sum_plus_bias() {
    let input_array = [1.0, 2.0, 3.0];
    let weights = [0.5, -1.0, 2.0];
    let bias = 0.25;

    assert_close(linear_func(&input_array, &weights, bias), 4.75);
}

#[test]
fn linear_func_with_zero_weights_returns_bias() {
    let input_array = [7.0, -3.5, 100.0];
    let weights = [0.0, 0.0, 0.0];

    assert_close(linear_func(&input_array, &weights, 1.5), 1.5);
    assert_close(linear_func(&input_array, &weights, 0.0), 0.0);
}

#[test]
fn linear_func_with_empty_weights_returns_bias() {
    assert_close(linear_func(&[], &[], -2.0), -2.0);
}

#[test]
fn linear_func_only_consumes_as_many_inputs_as_weights() {
    let input_array = [1.0, 2.0, 3.0, 4.0];
    let weights = [1.0, 1.0];

    assert_close(linear_func(&input_array, &weights, 0.0), 3.0);
}

#[test]
fn linear_func_matches_reference_implementation() {
    let input_array = [0.1, -0.2, 0.3, -0.4];
    let weights = [1.5, 2.5, -3.5, 4.5];
    let bias = -0.75;

    assert_close(
        linear_func(&input_array, &weights, bias),
        linear(&input_array, &weights, bias),
    );
}

#[test]
#[should_panic]
fn linear_func_panics_when_there_are_more_weights_than_inputs() {
    linear_func(&[1.0], &[1.0, 2.0], 0.0);
}

#[test]
fn sigmoid_func_applies_sigmoid_over_the_inner_function() {
    let input_array = [1.0, 2.0];
    let weights = [1.0, -0.5];
    let bias = 0.5;

    // linear part: 1*1 + 2*(-0.5) + 0.5 = 0.5
    assert_close(
        sigmoid_func(linear_func, &input_array, &weights, bias),
        sigmoid(0.5),
    );
}

#[test]
fn sigmoid_func_returns_one_half_at_zero() {
    assert_close(sigmoid_func(linear_func, &[3.0], &[0.0], 0.0), 0.5);
}

#[test]
fn sigmoid_func_output_stays_between_zero_and_one() {
    for x in [-50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0] {
        let output = sigmoid_func(linear_func, &[x], &[1.0], 0.0);
        assert!(
            (0.0..=1.0).contains(&output),
            "sigmoid({x}) = {output} is out of the [0, 1] range"
        );
    }

    for x in [-5.0, -1.0, 0.0, 1.0, 5.0] {
        let output = sigmoid_func(linear_func, &[x], &[1.0], 0.0);
        assert!(
            output > 0.0 && output < 1.0,
            "sigmoid({x}) = {output} is out of the (0, 1) range"
        );
    }
}

#[test]
fn sigmoid_func_saturates_for_large_inputs() {
    assert_close(sigmoid_func(linear_func, &[40.0], &[1.0], 0.0), 1.0);
    assert_close(sigmoid_func(linear_func, &[-40.0], &[1.0], 0.0), 0.0);
}

#[test]
fn sigmoid_func_is_monotonically_increasing() {
    let inputs = [-3.0, -1.0, 0.0, 1.0, 3.0];

    let mut previous = f64::NEG_INFINITY;
    for x in inputs {
        let current = sigmoid_func(linear_func, &[x], &[1.0], 0.0);
        assert!(current > previous, "sigmoid is not increasing at x = {x}");
        previous = current;
    }
}

#[test]
fn sigmoid_func_is_symmetric_around_one_half() {
    let positive = sigmoid_func(linear_func, &[2.0], &[1.0], 0.0);
    let negative = sigmoid_func(linear_func, &[-2.0], &[1.0], 0.0);

    assert_close(positive + negative, 1.0);
}

#[test]
fn sigmoid_func_can_be_used_with_a_custom_inner_function() {
    fn constant_two(_input_array: &[f64], _weights: &[f64], _bias: f64) -> f64 {
        2.0
    }

    assert_close(sigmoid_func(constant_two, &[], &[], 0.0), sigmoid(2.0));
}

#[test]
fn sigmoid_func_using_linear_func_is_the_composition_of_both() {
    let input_array = [0.5, -1.5, 2.0];
    let weights = [2.0, 1.0, -0.5];
    let bias = 0.25;

    let expected = sigmoid(linear(&input_array, &weights, bias));

    assert_close(
        sigmoid_func_using_linear_func(&input_array, &weights, bias),
        expected,
    );
    assert_close(
        sigmoid_func_using_linear_func(&input_array, &weights, bias),
        sigmoid_func(linear_func, &input_array, &weights, bias),
    );
}
