use crate::activation_function_type::ActivationFunctionType;
use crate::functions;

pub struct Unit {
    pub activation_function: fn(&[f64], &[f64], f64) -> f64,
    pub weights : Vec<f64>,
    pub bias : f64,
}

impl Unit {
    pub fn new(
        activation_function_type: ActivationFunctionType,
        weights: Vec<f64>,
        bias: f64,
    ) -> Self {

        let activation_function: fn(&[f64], &[f64], f64) -> f64 = match activation_function_type {
            ActivationFunctionType::Linear  => functions::linear_func,
            ActivationFunctionType::Sigmoid => functions::sigmoid_func_using_linear_func,
        };

        Self { activation_function, weights, bias }
    }
}