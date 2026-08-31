use crate::activation_function_type::ActivationFunctionType;

pub struct LayerRequestInfo {
    pub activation_function_type: ActivationFunctionType,
    pub unit_count: usize,
}

impl LayerRequestInfo {
    pub fn new(activation_function_type: ActivationFunctionType, unit_count: usize) -> Self {
        Self { activation_function_type, unit_count }
    }
}