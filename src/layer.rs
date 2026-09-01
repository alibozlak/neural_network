use crate::unit::Unit;

pub struct Layer {
    units: Vec<Unit>,
}

impl Layer {
    pub fn new(units: Vec<Unit>, ) -> Self {
        Self { units }
    }

    pub fn summary(&self) -> String {
        format!("Unit_Count: {}", self.units.len())
    }

    pub fn get_units(&self) -> &Vec<Unit> {
        &self.units
    }
}