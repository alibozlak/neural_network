use crate::activation_function_type::ActivationFunctionType;
use crate::functions;

pub struct Unit {
    activation_function: fn(&[f64], &[f64], f64) -> f64,
    weights : Vec<f64>,
    bias : f64,
}

impl Unit {
    pub fn new(
        activation_function_type: &ActivationFunctionType,
        weights: Vec<f64>,
        bias: f64,
    ) -> Self {

        let activation_function: fn(&[f64], &[f64], f64) -> f64 = match activation_function_type {
            ActivationFunctionType::Linear  => functions::linear_func,
            ActivationFunctionType::Sigmoid => functions::sigmoid_func_using_linear_func,
        };

        Self { activation_function, weights, bias }
    }


    pub fn get_activation_function(&self) -> &fn(&[f64], &[f64], f64) -> f64 {
        &self.activation_function
    }

    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }


}