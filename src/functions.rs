use crate::{validates};

pub fn sigmoid_func(
    inside_func: fn(&[f64], &[f64], f64) -> f64,
    input_array : &[f64],
    weights : &[f64],
    bias : f64
) -> f64 {
    1. / (1. + (-inside_func(input_array, weights, bias)).exp())
}

pub fn sigmoid_func_using_linear_func(input_array : &[f64], weights : &[f64], bias : f64) -> f64 {

    sigmoid_func(linear_func, input_array, weights, bias)
}

pub fn linear_func(input_array : &[f64], weights : &[f64], bias : f64) -> f64 {
    let n = validates::validate_one_sample_array_and_weight_array_same_size(input_array, weights);

    let mut output : f64 = 0.0;
    for i in 0..n { output += input_array[i] * weights[i] }

    output + bias
}