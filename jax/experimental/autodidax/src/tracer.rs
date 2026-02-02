use crate::aval::AbstractValue;
use std::cell::RefCell;

pub type TraceLevel = usize;

#[derive(Debug, Clone)]
pub struct MainTrace {
    pub level: TraceLevel,
    pub trace_type: TraceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceType {
    Eval,
}

thread_local! {
    static TRACE_STACK: RefCell<Vec<MainTrace>> = RefCell::new(vec![
        MainTrace {
            level: 0,
            trace_type: TraceType::Eval,
        }
    ]);
}

pub fn push_trace(trace_type: TraceType) -> MainTrace {
    TRACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let level = stack.len();
        let main = MainTrace { level, trace_type };
        stack.push(main.clone());
        main
    })
}

pub fn pop_trace() {
    TRACE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.len() > 1 {
            stack.pop();
        }
    })
}

pub fn current_trace() -> MainTrace {
    TRACE_STACK.with(|stack| stack.borrow().last().unwrap().clone())
}

#[derive(Debug, Clone)]
pub enum Value {
    Float(f64),
    Tracer(Box<TracerValue>),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn is_tracer(&self) -> bool {
        matches!(self, Value::Tracer(_))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

#[derive(Debug, Clone)]
pub struct TracerValue {
    pub trace_level: TraceLevel,
    pub aval: AbstractValue,
}

pub fn find_top_trace(args: &[Value]) -> MainTrace {
    let mut max_level = 0;

    for arg in args {
        if let Value::Tracer(t) = arg {
            if t.trace_level > max_level {
                max_level = t.trace_level;
            }
        }
    }

    TRACE_STACK.with(|stack| {
        let stack = stack.borrow();
        if max_level < stack.len() {
            stack[max_level].clone()
        } else {
            stack[0].clone()
        }
    })
}

pub fn full_lower(val: Value) -> Value {
    val
}
