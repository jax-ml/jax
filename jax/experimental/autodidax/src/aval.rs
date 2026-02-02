use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float32 => write!(f, "float32"),
            DType::Float64 => write!(f, "float64"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShapedArray {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl ShapedArray {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        ShapedArray { shape, dtype }
    }

    pub fn scalar(dtype: DType) -> Self {
        ShapedArray {
            shape: vec![],
            dtype,
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn str_short(&self) -> String {
        let shape_str = self
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("{}[{}]", self.dtype, shape_str)
    }
}

#[derive(Debug, Clone)]
pub struct ConcreteArray {
    pub val: f64,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl ConcreteArray {
    pub fn from_f64(val: f64) -> Self {
        ConcreteArray {
            val,
            shape: vec![],
            dtype: DType::Float64,
        }
    }

    pub fn to_shaped(&self) -> ShapedArray {
        ShapedArray {
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AbstractValue {
    Shaped(ShapedArray),
    Concrete(ConcreteArray),
}

impl AbstractValue {
    pub fn shape(&self) -> &[usize] {
        match self {
            AbstractValue::Shaped(s) => &s.shape,
            AbstractValue::Concrete(c) => &c.shape,
        }
    }

    pub fn dtype(&self) -> &DType {
        match self {
            AbstractValue::Shaped(s) => &s.dtype,
            AbstractValue::Concrete(c) => &c.dtype,
        }
    }
}

pub fn get_aval(x: f64) -> AbstractValue {
    AbstractValue::Concrete(ConcreteArray::from_f64(x))
}

pub fn raise_to_shaped(aval: &AbstractValue) -> ShapedArray {
    match aval {
        AbstractValue::Shaped(s) => s.clone(),
        AbstractValue::Concrete(c) => c.to_shaped(),
    }
}
