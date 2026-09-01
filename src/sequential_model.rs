use crate::layer::Layer;
use crate::layer_request_infos::LayerRequestInfo;
use crate::unit::Unit;

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

        for layer_index in 0..layer_count {
            let unit_count = layer_request_infos[layer_index].unit_count;
            let activation_function_type
                = &layer_request_infos[layer_index].activation_function_type;

            let mut weights: Vec<f64> = vec![0.0; sample_feature_size];
            if layer_index != 0 {
                weights = vec![0.0; layer_request_infos[layer_index - 1].unit_count];
            }

            let mut units: Vec<Unit> = Vec::with_capacity(unit_count);
            for _ in 0..unit_count {
                units.push(Unit::new(activation_function_type, weights.clone(), 0.));
            }
            let layer: Layer = Layer::new(units);

            layers.push(layer);
        }

        Self { layers, layer_count }
    }

    pub fn predict_function(&self, input_sample: &Vec<f64>) -> f64 {
        let layer_1_weight_size: usize = self.layers[0].get_units()[0].get_weights().len();

        if layer_1_weight_size != input_sample.len() {
            panic!("Input sample feature size must be {} count !! Yours = {}", layer_1_weight_size, input_sample.len());
        }

        let mut a_before: Vec<f64> = input_sample.clone();
        for layer_index in 0..self.layer_count {
            let unit_count = self.layers[layer_index].get_units().len();
            let mut a_after: Vec<f64> = vec![0.; unit_count];

            for u in 0..unit_count {
                let activation_func = self.layers[layer_index].get_units()[u].get_activation_function();
                let weights = &self.layers[layer_index].get_units()[u].get_weights();
                let bias = self.layers[layer_index].get_units()[u].get_bias();
                a_after[u] = activation_func(&a_before, weights, bias);
            }
            a_before = a_after;
        }

        a_before[0]
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


