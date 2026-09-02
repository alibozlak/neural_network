

#[derive(Clone, Copy, PartialEq)]
pub enum Activation {
    Sigmoid,
    // Linear,
}

impl Activation {
    pub fn apply(self, z: f64) -> f64 {
        let mut result: f64 = 1.0;

        if self == Self::Sigmoid {
            result = 1. / (1. + (-z).exp());
        }

        result
    }

    // pub fn derivative(self, x: f64) -> f64 {
    // }
}