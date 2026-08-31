use crate::layer::Layer;
use crate::layer_request_infos::LayerRequestInfo;
use crate::unit::Unit;

#[allow(non_snake_case)]
pub struct NeuralNetwork {
    pub layers : Vec<Layer>,
}

#[allow(non_snake_case)]
impl NeuralNetwork {
    pub fn new(
        X: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        layer_request_infos: &Vec<LayerRequestInfo>,
    ) -> Self {
        let (layer_count, X_feature_size) = Self::validate(&X, &y, &layer_request_infos);

        let mut layers : Vec<Layer> = Vec::with_capacity(layer_count);

        for layer_index in 0..layer_count {
            let unit_count = layer_request_infos[layer_index].unit_count;
            let activation_function_type = &layer_request_infos[layer_index].activation_function_type;

            let mut weights: Vec<f64> = vec![0.0; X_feature_size];
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

        Self { layers }
    }

    pub fn network_union_predict_function(&self, input_sample: &Vec<f64>) -> f64 {
        let layer_1_weight_size: usize = self.layers[0].units[0].weights.len();

        if layer_1_weight_size != input_sample.len() {
            panic!("Input sample feature size must be {} count !! Yours = {}", layer_1_weight_size, input_sample.len());
        }

        let layer_count = self.layers.len();

        let mut a_before: Vec<f64> = input_sample.clone();
        for layer_index in 0..layer_count {
            let unit_count = self.layers[layer_index].units.len();
            let mut a_after: Vec<f64> = vec![0.; unit_count];

            for u in 0..unit_count {
                let activation_func = self.layers[layer_index].units[u].activation_function;
                let weights = &self.layers[layer_index].units[u].weights;
                let bias = self.layers[layer_index].units[u].bias;
                let a_before2 = &a_before.clone();
                a_after[u] = (activation_func)(a_before2, weights, bias);
            }
            a_before = a_after;
        }

        a_before[0]
    }

    pub fn validate(
        X: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        layer_request_infos: &Vec<LayerRequestInfo>
    ) -> (usize, usize) {
        let layer_count = layer_request_infos.len();
        if layer_count == 0 {
            panic!("Network must has at least one layer!!");
        }

        if layer_request_infos[layer_count-1].unit_count != 1 {
            panic!("Output layer must have only one unit!!");
        }

        let X_feature_size: usize = X[0].len();
        let sample_size: usize = X.len();

        if sample_size != y.len() {
            panic!("y and X must have the same sample count!!");
        }

        for i in 1..sample_size {
            if X_feature_size != X[i].len() {
                panic!("X's each sample feature count should be equal!!");
            }
        }

        (layer_count, X_feature_size)
    }

}


