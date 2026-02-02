use crate::primitive::Primitive;
use std::sync::atomic::{AtomicUsize, Ordering};

static VAR_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn fresh_var_id() -> usize {
    VAR_COUNTER.fetch_add(1, Ordering::SeqCst)
}

pub fn reset_var_counter() {
    VAR_COUNTER.store(0, Ordering::SeqCst);
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Var {
    pub id: usize,
    pub name: Option<String>,
}

impl Var {
    pub fn new() -> Self {
        Var {
            id: fresh_var_id(),
            name: None,
        }
    }

    pub fn with_name(name: &str) -> Self {
        Var {
            id: fresh_var_id(),
            name: Some(name.to_string()),
        }
    }

    pub fn display_name(&self) -> String {
        match &self.name {
            Some(n) => n.clone(),
            None => format!("_{}", self.id),
        }
    }
}

impl Default for Var {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum Atom {
    Var(Var),
    Lit(f64),
}

impl Atom {
    pub fn is_var(&self) -> bool {
        matches!(self, Atom::Var(_))
    }

    pub fn is_lit(&self) -> bool {
        matches!(self, Atom::Lit(_))
    }
}

#[derive(Debug, Clone)]
pub struct JaxprEqn {
    pub primitive: Primitive,
    pub inputs: Vec<Atom>,
    pub outputs: Vec<Var>,
}

impl JaxprEqn {
    pub fn new(primitive: Primitive, inputs: Vec<Atom>, outputs: Vec<Var>) -> Self {
        JaxprEqn {
            primitive,
            inputs,
            outputs,
        }
    }

    pub fn to_string(&self) -> String {
        let ins: Vec<String> = self.inputs.iter().map(|a| match a {
            Atom::Var(v) => v.display_name(),
            Atom::Lit(f) => format!("{}", f),
        }).collect();
        let outs: Vec<String> = self.outputs.iter().map(|v| v.display_name()).collect();
        format!("{} = {:?}({})", outs.join(", "), self.primitive, ins.join(", "))
    }
}

#[derive(Debug, Clone)]
pub struct Jaxpr {
    pub in_binders: Vec<Var>,
    pub eqns: Vec<JaxprEqn>,
    pub out_atoms: Vec<Atom>,
}

impl Jaxpr {
    pub fn new(in_binders: Vec<Var>, eqns: Vec<JaxprEqn>, out_atoms: Vec<Atom>) -> Self {
        Jaxpr {
            in_binders,
            eqns,
            out_atoms,
        }
    }

    pub fn pretty_print(&self) -> String {
        let mut lines = Vec::new();
        
        let in_names: Vec<String> = self.in_binders.iter()
            .map(|v| v.display_name())
            .collect();
        lines.push(format!("{{ lambda {} ; let", in_names.join(" ")));
        
        for eqn in &self.eqns {
            let ins: Vec<String> = eqn.inputs.iter().map(|a| match a {
                Atom::Var(v) => v.display_name(),
                Atom::Lit(f) => format!("{:.1}", f),
            }).collect();
            let outs: Vec<String> = eqn.outputs.iter().map(|v| v.display_name()).collect();
            lines.push(format!("    {} = {} {}", outs.join(" "), eqn.primitive.name(), ins.join(" ")));
        }
        
        let out_names: Vec<String> = self.out_atoms.iter()
            .map(|a| match a {
                Atom::Var(v) => v.display_name(),
                Atom::Lit(f) => format!("{:.1}", f),
            })
            .collect();
        lines.push(format!("  in ({}) }}", out_names.join(", ")));
        
        lines.join("\n")
    }
}

#[derive(Debug, Clone)]
pub struct JaxprTracer {
    pub var: Var,
}

impl JaxprTracer {
    pub fn new() -> Self {
        JaxprTracer { var: Var::new() }
    }

    pub fn from_var(var: Var) -> Self {
        JaxprTracer { var }
    }
}

impl Default for JaxprTracer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum JaxprValue {
    Concrete(f64),
    Tracer(JaxprTracer),
}

impl JaxprValue {
    pub fn to_atom(&self) -> Atom {
        match self {
            JaxprValue::Concrete(f) => Atom::Lit(*f),
            JaxprValue::Tracer(t) => Atom::Var(t.var.clone()),
        }
    }

    pub fn is_tracer(&self) -> bool {
        matches!(self, JaxprValue::Tracer(_))
    }

    pub fn get_var(&self) -> Option<&Var> {
        match self {
            JaxprValue::Tracer(t) => Some(&t.var),
            _ => None,
        }
    }
}

use std::cell::RefCell;

thread_local! {
    static JAXPR_TRACE_EQNS: RefCell<Vec<JaxprEqn>> = RefCell::new(Vec::new());
    static JAXPR_TRACE_ACTIVE: RefCell<bool> = RefCell::new(false);
}

pub fn start_jaxpr_trace() {
    reset_var_counter();
    JAXPR_TRACE_EQNS.with(|eqns| eqns.borrow_mut().clear());
    JAXPR_TRACE_ACTIVE.with(|active| *active.borrow_mut() = true);
}

pub fn end_jaxpr_trace() -> Vec<JaxprEqn> {
    JAXPR_TRACE_ACTIVE.with(|active| *active.borrow_mut() = false);
    JAXPR_TRACE_EQNS.with(|eqns| eqns.borrow().clone())
}

pub fn is_jaxpr_tracing() -> bool {
    JAXPR_TRACE_ACTIVE.with(|active| *active.borrow())
}

pub fn record_eqn(eqn: JaxprEqn) {
    JAXPR_TRACE_EQNS.with(|eqns| eqns.borrow_mut().push(eqn));
}

pub fn jaxpr_bind(prim: Primitive, args: Vec<JaxprValue>) -> JaxprValue {
    let inputs: Vec<Atom> = args.iter().map(|a| a.to_atom()).collect();
    let out_var = Var::new();
    let eqn = JaxprEqn::new(prim, inputs, vec![out_var.clone()]);
    record_eqn(eqn);
    JaxprValue::Tracer(JaxprTracer::from_var(out_var))
}

pub fn make_jaxpr_inputs(num_inputs: usize) -> Vec<JaxprValue> {
    (0..num_inputs)
        .map(|_| JaxprValue::Tracer(JaxprTracer::new()))
        .collect()
}

pub fn finalize_jaxpr(
    in_tracers: Vec<JaxprValue>,
    out_values: Vec<JaxprValue>,
) -> Jaxpr {
    let eqns = end_jaxpr_trace();
    
    let in_binders: Vec<Var> = in_tracers
        .iter()
        .map(|t| t.get_var().unwrap().clone())
        .collect();
    
    let out_atoms: Vec<Atom> = out_values.iter().map(|v| v.to_atom()).collect();
    
    Jaxpr::new(in_binders, eqns, out_atoms)
}
