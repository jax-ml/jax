use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::conversion::ToPyObject;
use std::collections::HashMap;

use crate::eval_trace::{bind1 as rust_bind1, bind as rust_bind};
use crate::jaxpr::{self, Jaxpr as JaxprIR, JaxprValue};
use crate::jvp_trace::{self, JvpValue};
use crate::primitive::Primitive;
use crate::tracer::Value;
use crate::stablehlo;

fn str_to_primitive(name: &str) -> PyResult<Primitive> {
    match name {
        "add" => Ok(Primitive::Add),
        "mul" => Ok(Primitive::Mul),
        "neg" => Ok(Primitive::Neg),
        "sin" => Ok(Primitive::Sin),
        "cos" => Ok(Primitive::Cos),
        "reduce_sum" => Ok(Primitive::ReduceSum),
        "greater" => Ok(Primitive::Greater),
        "less" => Ok(Primitive::Less),
        _ => Err(PyValueError::new_err(format!("Unknown primitive: {}", name))),
    }
}

#[pyfunction]
fn bind(prim_name: &str, args: Vec<f64>) -> PyResult<Vec<f64>> {
    let prim = str_to_primitive(prim_name)?;
    let values: Vec<Value> = args.into_iter().map(Value::Float).collect();
    let results = rust_bind(prim, values, HashMap::new());
    Ok(results.into_iter().map(|v| v.as_f64().unwrap()).collect())
}

#[pyfunction]
fn bind1(prim_name: &str, args: Vec<f64>) -> PyResult<f64> {
    let prim = str_to_primitive(prim_name)?;
    let values: Vec<Value> = args.into_iter().map(Value::Float).collect();
    let result = rust_bind1(prim, values, HashMap::new());
    Ok(result.as_f64().unwrap())
}

#[pyclass]
#[derive(Clone)]
pub struct JvpTracer {
    inner: JvpValue,
}

impl JvpTracer {
    fn from_jvp_value(val: JvpValue) -> Self {
        JvpTracer { inner: val }
    }
}

#[pymethods]
impl JvpTracer {
    #[new]
    fn new(primal: &Bound<'_, PyAny>, tangent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p = py_to_jvp_value(primal)?;
        let t = py_to_jvp_value(tangent)?;
        Ok(JvpTracer {
            inner: JvpValue::tracer(p, t),
        })
    }

    #[getter]
    fn primal(&self, py: Python<'_>) -> PyResult<PyObject> {
        jvp_value_to_py(&self.inner.primal(), py)
    }

    #[getter]
    fn tangent(&self, py: Python<'_>) -> PyResult<PyObject> {
        jvp_value_to_py(&self.inner.tangent(), py)
    }

    fn get_concrete(&self) -> f64 {
        self.inner.get_concrete()
    }

    fn __repr__(&self) -> String {
        format!("JvpTracer(primal={:?}, tangent={:?})", 
                self.inner.primal(), self.inner.tangent())
    }

    fn __neg__(&self) -> JvpTracer {
        JvpTracer::from_jvp_value(jvp_trace::jvp_neg(self.inner.clone()))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
        let other_val = py_to_jvp_value(other)?;
        Ok(JvpTracer::from_jvp_value(jvp_trace::jvp_add(self.inner.clone(), other_val)))
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
        let other_val = py_to_jvp_value(other)?;
        Ok(JvpTracer::from_jvp_value(jvp_trace::jvp_add(other_val, self.inner.clone())))
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
        let other_val = py_to_jvp_value(other)?;
        let neg_other = jvp_trace::jvp_neg(other_val);
        Ok(JvpTracer::from_jvp_value(jvp_trace::jvp_add(self.inner.clone(), neg_other)))
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
        let other_val = py_to_jvp_value(other)?;
        let neg_self = jvp_trace::jvp_neg(self.inner.clone());
        Ok(JvpTracer::from_jvp_value(jvp_trace::jvp_add(other_val, neg_self)))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
        let other_val = py_to_jvp_value(other)?;
        Ok(JvpTracer::from_jvp_value(jvp_trace::jvp_mul(self.inner.clone(), other_val)))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
        let other_val = py_to_jvp_value(other)?;
        Ok(JvpTracer::from_jvp_value(jvp_trace::jvp_mul(other_val, self.inner.clone())))
    }
}

fn py_to_jvp_value(obj: &Bound<'_, PyAny>) -> PyResult<JvpValue> {
    if let Ok(tracer) = obj.extract::<JvpTracer>() {
        return Ok(tracer.inner);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(JvpValue::Concrete(f));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert {} to JvpValue",
        obj.get_type().name()?
    )))
}

fn jvp_value_to_py(val: &JvpValue, py: Python<'_>) -> PyResult<PyObject> {
    match val {
        JvpValue::Concrete(f) => Ok(f.to_object(py)),
        JvpValue::Tracer(_) => {
            let tracer = JvpTracer::from_jvp_value(val.clone());
            Ok(tracer.into_py(py))
        }
    }
}

#[pyfunction]
fn jvp_bind1(prim_name: &str, args: Vec<JvpTracer>) -> PyResult<JvpTracer> {
    let prim = str_to_primitive(prim_name)?;
    let jvp_values: Vec<JvpValue> = args.into_iter().map(|t| t.inner).collect();
    let result = jvp_trace::jvp_primitive(prim, jvp_values);
    Ok(JvpTracer::from_jvp_value(result))
}

#[pyfunction]
fn make_jvp_tracer(primal: &Bound<'_, PyAny>, tangent: &Bound<'_, PyAny>) -> PyResult<JvpTracer> {
    let p = py_to_jvp_value(primal)?;
    let t = py_to_jvp_value(tangent)?;
    Ok(JvpTracer::from_jvp_value(JvpValue::tracer(p, t)))
}

#[pyclass]
#[derive(Clone)]
pub struct JaxprTracer {
    inner: JaxprValue,
}

impl JaxprTracer {
    fn from_jaxpr_value(val: JaxprValue) -> Self {
        JaxprTracer { inner: val }
    }
}

#[pymethods]
impl JaxprTracer {
    fn __repr__(&self) -> String {
        match &self.inner {
            JaxprValue::Concrete(f) => format!("{}", f),
            JaxprValue::Tracer(t) => t.var.display_name(),
        }
    }

    fn __neg__(&self) -> JaxprTracer {
        let result = jaxpr::jaxpr_bind(Primitive::Neg, vec![self.inner.clone()]);
        JaxprTracer::from_jaxpr_value(result)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<JaxprTracer> {
        let other_val = py_to_jaxpr_value(other)?;
        let result = jaxpr::jaxpr_bind(Primitive::Add, vec![self.inner.clone(), other_val]);
        Ok(JaxprTracer::from_jaxpr_value(result))
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<JaxprTracer> {
        let other_val = py_to_jaxpr_value(other)?;
        let result = jaxpr::jaxpr_bind(Primitive::Add, vec![other_val, self.inner.clone()]);
        Ok(JaxprTracer::from_jaxpr_value(result))
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<JaxprTracer> {
        let other_val = py_to_jaxpr_value(other)?;
        let result = jaxpr::jaxpr_bind(Primitive::Mul, vec![self.inner.clone(), other_val]);
        Ok(JaxprTracer::from_jaxpr_value(result))
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<JaxprTracer> {
        let other_val = py_to_jaxpr_value(other)?;
        let result = jaxpr::jaxpr_bind(Primitive::Mul, vec![other_val, self.inner.clone()]);
        Ok(JaxprTracer::from_jaxpr_value(result))
    }
}

fn py_to_jaxpr_value(obj: &Bound<'_, PyAny>) -> PyResult<JaxprValue> {
    if let Ok(tracer) = obj.extract::<JaxprTracer>() {
        return Ok(tracer.inner);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(JaxprValue::Concrete(f));
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert {} to JaxprValue",
        obj.get_type().name()?
    )))
}

#[pyclass]
pub struct Jaxpr {
    inner: JaxprIR,
}

#[pymethods]
impl Jaxpr {
    fn __repr__(&self) -> String {
        self.inner.pretty_print()
    }

    fn __str__(&self) -> String {
        self.inner.pretty_print()
    }
}

#[pyfunction]
fn start_jaxpr_trace() {
    jaxpr::start_jaxpr_trace();
}

#[pyfunction]
fn is_jaxpr_tracing() -> bool {
    jaxpr::is_jaxpr_tracing()
}

#[pyfunction]
fn make_jaxpr_tracers(num_inputs: usize) -> Vec<JaxprTracer> {
    let tracers = jaxpr::make_jaxpr_inputs(num_inputs);
    tracers.into_iter()
        .map(|v| JaxprTracer::from_jaxpr_value(v))
        .collect()
}

#[pyfunction]
fn finalize_jaxpr(in_tracers: Vec<JaxprTracer>, out_tracers: Vec<JaxprTracer>) -> Jaxpr {
    let in_values: Vec<JaxprValue> = in_tracers.into_iter().map(|t| t.inner).collect();
    let out_values: Vec<JaxprValue> = out_tracers.into_iter().map(|t| t.inner).collect();
    let jaxpr_ir = jaxpr::finalize_jaxpr(in_values, out_values);
    Jaxpr { inner: jaxpr_ir }
}

#[pyfunction]
fn jaxpr_bind1(prim_name: &str, args: Vec<Bound<'_, PyAny>>) -> PyResult<JaxprTracer> {
    let prim = str_to_primitive(prim_name)?;
    let mut jaxpr_values = Vec::new();
    for arg in args {
        if let Ok(tracer) = arg.extract::<JaxprTracer>() {
            jaxpr_values.push(tracer.inner);
        } else if let Ok(f) = arg.extract::<f64>() {
            jaxpr_values.push(JaxprValue::Concrete(f));
        } else {
            return Err(PyValueError::new_err(format!(
                "Cannot convert {} to JaxprValue",
                arg.get_type().name()?
            )));
        }
    }
    let result = jaxpr::jaxpr_bind(prim, jaxpr_values);
    Ok(JaxprTracer::from_jaxpr_value(result))
}

#[pyfunction]
fn zeros_like(_val: f64) -> f64 {
    0.0
}

#[pyfunction]
fn lower_jaxpr_to_stablehlo(jaxpr: &Jaxpr) -> String {
    stablehlo::lower_to_stablehlo(&jaxpr.inner)
}

#[pymodule]
fn autodidax_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bind, m)?)?;
    m.add_function(wrap_pyfunction!(bind1, m)?)?;
    m.add_function(wrap_pyfunction!(jvp_bind1, m)?)?;
    m.add_function(wrap_pyfunction!(make_jvp_tracer, m)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_class::<JvpTracer>()?;
    m.add_function(wrap_pyfunction!(start_jaxpr_trace, m)?)?;
    m.add_function(wrap_pyfunction!(is_jaxpr_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(make_jaxpr_tracers, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_jaxpr, m)?)?;
    m.add_function(wrap_pyfunction!(jaxpr_bind1, m)?)?;
    m.add_function(wrap_pyfunction!(lower_jaxpr_to_stablehlo, m)?)?;
    m.add_class::<JaxprTracer>()?;
    m.add_class::<Jaxpr>()?;
    Ok(())
}
