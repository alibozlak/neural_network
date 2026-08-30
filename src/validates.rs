

pub fn validate_one_sample_array_and_weight_array_same_size(
    one_sample_array : &[f64], weights : &[f64]
) -> usize {
    let n = weights.len();

    if n != one_sample_array.len() {
        panic!("The weights array and one sample array must be same length!! \
        weights length: {}, one_sample_array length: {}", n, one_sample_array.len()
        );
    }

    n
}