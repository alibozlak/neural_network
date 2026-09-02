use ndarray::{ Array2 };
use crate::activation::Activation;

pub struct Layer {

    /// Weights fill column step column.
    /// matrix shape = (weight_count + 1, unit_count + 1)
    /// +1s for matrix multiplication :
    /// weight_count + 1 : each unit's weights + bias,
    /// unit_count + 1 : for a_next (layer output)
    matrix: Array2<f64>,
    
    activation: Activation,
}

impl Layer {
    pub fn new(
        matrix: Array2<f64>,
        activation: Activation,
    ) -> Self {
        Self { matrix, activation }
    }

    pub fn get_matrix(&self) -> &Array2<f64> {
        &self.matrix
    }
    
    pub fn get_activation_function(&self) -> Activation {
        self.activation
    }

    pub fn summary(&self) -> String {
        format!("Unit_Count: {}, weight_count: {}", self.matrix.ncols(), self.matrix.nrows())
    }


}