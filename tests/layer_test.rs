use neural_network::activation_function_type::ActivationFunctionType;
use neural_network::layer::Layer;
use neural_network::unit::Unit;

mod common;
use common::assert_close;

fn unit(weights: Vec<f64>, bias: f64) -> Unit {
    Unit::new(&ActivationFunctionType::Linear, weights, bias)
}

#[test]
fn new_stores_every_given_unit_in_order() {
    let layer = Layer::new(vec![
        unit(vec![1.0], 0.1),
        unit(vec![2.0], 0.2),
        unit(vec![3.0], 0.3),
    ]);

    assert_eq!(layer.get_units().len(), 3);
    assert_eq!(layer.get_units()[0].get_weights(), &vec![1.0]);
    assert_eq!(layer.get_units()[1].get_weights(), &vec![2.0]);
    assert_eq!(layer.get_units()[2].get_weights(), &vec![3.0]);
    assert_close(layer.get_units()[2].get_bias(), 0.3);
}

#[test]
fn new_accepts_an_empty_unit_list() {
    let layer = Layer::new(vec![]);

    assert!(layer.get_units().is_empty());
}

#[test]
fn units_of_a_layer_can_be_evaluated_one_by_one() {
    let layer = Layer::new(vec![unit(vec![1.0, 1.0], 0.0), unit(vec![2.0, 0.0], 1.0)]);
    let input_array = [3.0, 4.0];

    let outputs: Vec<f64> = layer
        .get_units()
        .iter()
        .map(|u| (u.get_activation_function())(&input_array, &u.get_weights(), u.get_bias()))
        .collect();

    assert_close(outputs[0], 7.0);
    assert_close(outputs[1], 7.0);
}
