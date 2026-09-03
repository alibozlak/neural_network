use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Sigmoid,
    Linear,
}

impl Activation {
    pub fn apply(self, z: f64) -> f64 {
        let mut result: f64 = 1.0;

        if self == Self::Sigmoid {
            result = 1. / (1. + (-z).exp());
        } else if self == Self::Linear {
            result = z;
        }

        result
    }

    // pub fn derivative(self, x: f64) -> f64 {
    // }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Sigmoid => "Sigmoid",
            Self::Linear => "Linear",
        };
        write!(f, "{name}")
    }
}

impl FromStr for Activation {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "Sigmoid" => Ok(Self::Sigmoid),
            // "Linear" => Ok(Self::Linear),
            other => Err(format!("unknown activation: {other}")),
        }
    }
}