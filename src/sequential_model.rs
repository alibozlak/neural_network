use ndarray::{s, Array1, Array2};
use crate::layer::Layer;
use crate::layer_request_infos::LayerRequestInfo;

pub struct SequentialModel {
    layers : Vec<Layer>,
    layer_count : usize,
}

impl SequentialModel {
    pub fn new(
        sample_feature_size: usize,
        layer_request_infos: &Vec<LayerRequestInfo>,
    ) -> Self {
        let layer_count = layer_request_infos.len();
        let mut layers : Vec<Layer> = Vec::with_capacity(layer_count);

        let mut column_size: usize = sample_feature_size + 1;
        for layer_index in 0..layer_count {
            let unit_count = layer_request_infos[layer_index].unit_count;
            let array2: Array2<f64> = Array2::zeros(
                (column_size, unit_count + 1)
            );

            layers.push(
                Layer::new(array2, layer_request_infos[layer_index].activation)
            );

            column_size = unit_count + 1;
        }

        Self { layers, layer_count }
    }

    pub fn predict(&self, input: Array1<f64>) -> f64 {
        let feature_size = input.len();
        let first_layer_row_size = self.layers[0].get_matrix().nrows();
        if feature_size + 1 != first_layer_row_size {
            panic!("Input array's feature count not correct!!");
        }

        let mut a_previous_matrix: Array2<f64> = Array2::ones((1, feature_size+1));
        a_previous_matrix.slice_mut(s![0, ..feature_size]).assign(&input);
        for layer_index in 0..self.layer_count {
            a_previous_matrix = Self::build_a_next(&self.layers[layer_index], a_previous_matrix);
        }

        a_previous_matrix[[0, 0]]
    }

    fn build_a_next(layer: &Layer, a_previous: Array2<f64>) -> Array2<f64> {
        let z_matrix_linear_output = &a_previous.dot(layer.get_matrix());
        let mut a_next: Array2<f64>
            = z_matrix_linear_output.mapv(|z_ij| layer.get_activation_function().apply(z_ij));

        let column_size = a_next.ncols();
        a_next.column_mut(column_size - 1).fill(1.);
        a_next
    }

    pub fn summary(&self) -> String {
        let mut summary: String = format!("Layer_Count: {}\n", self.layer_count);
        for layer_index in 0..self.layer_count {
            summary.push_str(&format!(
                "Layer_{}: {}\n", layer_index + 1, self.layers[layer_index].summary())
            );
        }

        summary
    }

    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }





}


