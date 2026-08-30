use crate::unit::Unit;

pub struct Layer {
    pub units: Vec<Unit>,
}

impl Layer {
    pub fn new(units: Vec<Unit>, ) -> Self {
        Self { units }
    }
}