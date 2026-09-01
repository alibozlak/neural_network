use crate::layer_request_infos::LayerRequestInfo;


// FixMe: I should convert from panic! to Result
pub fn validate(
    samples: &Vec<Vec<f64>>,
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

    let sample_feature_size: usize = samples[0].len();
    let sample_size: usize = samples.len();

    if sample_size != y.len() {
        panic!("y and samples (X) must have the same sample count!!");
    }

    for i in 1..sample_size {
        if sample_feature_size != samples[i].len() {
            panic!("samples's each sample feature count should be equal!!");
        }
    }

    (layer_count, sample_feature_size)
}