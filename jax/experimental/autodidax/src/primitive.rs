#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Primitive {
    Add,
    Mul,
    Neg,
    Sin,
    Cos,
    ReduceSum,
    Greater,
    Less,
}

impl Primitive {
    pub fn name(&self) -> &'static str {
        match self {
            Primitive::Add => "add",
            Primitive::Mul => "mul",
            Primitive::Neg => "neg",
            Primitive::Sin => "sin",
            Primitive::Cos => "cos",
            Primitive::ReduceSum => "reduce_sum",
            Primitive::Greater => "greater",
            Primitive::Less => "less",
        }
    }
}
