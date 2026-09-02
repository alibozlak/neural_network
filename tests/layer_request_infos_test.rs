use neural_network::activation::Activation;
use neural_network::layer_request_infos::LayerRequestInfo;

#[test]
fn new_stores_the_given_activation_and_unit_count() {
    let info = LayerRequestInfo::new(Activation::Sigmoid, 7);

    assert!(info.activation == Activation::Sigmoid);
    assert_eq!(info.unit_count, 7);
}

#[test]
fn new_accepts_a_request_without_any_unit() {
    let info = LayerRequestInfo::new(Activation::Sigmoid, 0);

    assert_eq!(info.unit_count, 0);
}

#[test]
fn every_request_keeps_its_own_unit_count() {
    let infos = vec![
        LayerRequestInfo::new(Activation::Sigmoid, 4),
        LayerRequestInfo::new(Activation::Sigmoid, 2),
        LayerRequestInfo::new(Activation::Sigmoid, 1),
    ];

    let unit_counts: Vec<usize> = infos.iter().map(|info| info.unit_count).collect();

    assert_eq!(unit_counts, vec![4, 2, 1]);
}
