use crate::layer::Layer;

pub struct NeuralNetwork {
    pub layers : Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
}