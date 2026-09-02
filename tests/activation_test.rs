use neural_network::activation::Activation;

mod common;
use common::{assert_close, sigmoid};

#[test]
fn sigmoid_maps_zero_to_one_half() {
    assert_close(Activation::Sigmoid.apply(0.0), 0.5);
}

#[test]
fn sigmoid_matches_the_reference_implementation() {
    for z in [-10.0, -3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5, 10.0] {
        assert_close(Activation::Sigmoid.apply(z), sigmoid(z));
    }
}

#[test]
fn sigmoid_output_always_stays_between_zero_and_one() {
    for z in [-30.0, -7.0, -0.5, 0.0, 0.5, 7.0, 30.0] {
        let output = Activation::Sigmoid.apply(z);
        assert!(output > 0.0, "sigmoid({z}) = {output} is not greater than 0");
        assert!(output < 1.0, "sigmoid({z}) = {output} is not smaller than 1");
    }
}

/// Not a bug of the activation, only the limit of `f64`: from about 37 upwards
/// `exp(-z)` is too small to change the `1. + exp(-z)` sum, so the result rounds to
/// exactly 1. Anything that later relies on "the output is never exactly 1" (a
/// logarithm of `1 - output`, for instance) has to keep this in mind.
#[test]
fn sigmoid_rounds_to_exactly_one_for_a_large_input() {
    assert_close(Activation::Sigmoid.apply(40.0), 1.0);
}

#[test]
fn sigmoid_is_symmetric_around_one_half() {
    for z in [0.3, 1.0, 2.75, 6.0] {
        assert_close(
            Activation::Sigmoid.apply(-z),
            1.0 - Activation::Sigmoid.apply(z),
        );
    }
}

#[test]
fn sigmoid_is_strictly_increasing() {
    let inputs = [-8.0, -4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0, 8.0];

    for pair in inputs.windows(2) {
        let (smaller, bigger) = (pair[0], pair[1]);
        assert!(
            Activation::Sigmoid.apply(smaller) < Activation::Sigmoid.apply(bigger),
            "sigmoid({smaller}) is not smaller than sigmoid({bigger})"
        );
    }
}

#[test]
fn sigmoid_saturates_at_the_infinities() {
    assert_close(Activation::Sigmoid.apply(f64::INFINITY), 1.0);
    assert_close(Activation::Sigmoid.apply(f64::NEG_INFINITY), 0.0);
}

#[test]
fn sigmoid_of_not_a_number_is_not_a_number() {
    assert!(Activation::Sigmoid.apply(f64::NAN).is_nan());
}

#[test]
fn an_activation_is_copied_instead_of_being_moved() {
    let activation = Activation::Sigmoid;
    let copy = activation;

    assert!(activation == copy);
    assert_close(activation.apply(1.0), copy.apply(1.0));
}

#[test]
fn display_prints_the_variant_name() {
    assert_eq!(format!("{}", Activation::Sigmoid), "Sigmoid");
}

#[test]
fn to_string_comes_from_the_display_implementation() {
    assert_eq!(Activation::Sigmoid.to_string(), "Sigmoid");
}

#[test]
fn from_str_parses_back_the_name_that_display_prints() {
    let printed = Activation::Sigmoid.to_string();

    let parsed: Activation = printed.parse().expect("Display output must parse back");

    assert!(parsed == Activation::Sigmoid);
}

#[test]
fn from_str_rejects_an_unknown_name_and_names_it_in_the_error() {
    // `expect_err` would need `Activation: Debug`, which the enum does not derive yet.
    let error = match "relu".parse::<Activation>() {
        Ok(_) => panic!("relu must not parse into an activation"),
        Err(message) => message,
    };

    assert!(error.contains("relu"), "got: {error}");
}

#[test]
fn from_str_is_case_sensitive() {
    assert!("sigmoid".parse::<Activation>().is_err());
    assert!("SIGMOID".parse::<Activation>().is_err());
}

#[test]
fn from_str_rejects_an_empty_name() {
    assert!("".parse::<Activation>().is_err());
}
