use crate::primitive::Primitive;
use crate::tracer::Value;
use std::collections::HashMap;

pub fn eval_primitive(prim: Primitive, args: Vec<Value>, _params: &HashMap<String, String>) -> Vec<Value> {
    let floats: Vec<f64> = args
        .iter()
        .map(|v| v.as_f64().expect("EvalTrace requires concrete values"))
        .collect();

    let result = match prim {
        Primitive::Add => {
            assert_eq!(floats.len(), 2);
            floats[0] + floats[1]
        }
        Primitive::Mul => {
            assert_eq!(floats.len(), 2);
            floats[0] * floats[1]
        }
        Primitive::Neg => {
            assert_eq!(floats.len(), 1);
            -floats[0]
        }
        Primitive::Sin => {
            assert_eq!(floats.len(), 1);
            floats[0].sin()
        }
        Primitive::Cos => {
            assert_eq!(floats.len(), 1);
            floats[0].cos()
        }
        Primitive::ReduceSum => {
            floats.iter().sum()
        }
        Primitive::Greater => {
            assert_eq!(floats.len(), 2);
            if floats[0] > floats[1] { 1.0 } else { 0.0 }
        }
        Primitive::Less => {
            assert_eq!(floats.len(), 2);
            if floats[0] < floats[1] { 1.0 } else { 0.0 }
        }
    };

    vec![Value::Float(result)]
}

pub fn bind(prim: Primitive, args: Vec<Value>, params: HashMap<String, String>) -> Vec<Value> {
    eval_primitive(prim, args, &params)
}

pub fn bind1(prim: Primitive, args: Vec<Value>, params: HashMap<String, String>) -> Value {
    let mut results = bind(prim, args, params);
    assert_eq!(results.len(), 1);
    results.pop().unwrap()
}
