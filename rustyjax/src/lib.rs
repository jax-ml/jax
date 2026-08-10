use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

type VarId = u32;

// === IR ===

#[derive(Clone, Copy)]
enum Prim {
    Add,
    Mul,
}

impl Prim {
    fn name(self) -> &'static str {
        match self {
            Prim::Add => "add",
            Prim::Mul => "mul",
        }
    }

    fn eval(self, x: i64, y: i64) -> i64 {
        match self {
            Prim::Add => x + y,
            Prim::Mul => x * y,
        }
    }

    fn abstract_eval(self, _ins: [Aval; 2]) -> Aval {
        Aval::Int
    }

    // The XLA lowering rule: StableHLO op mnemonic.
    fn stablehlo(self) -> &'static str {
        match self {
            Prim::Add => "stablehlo.add",
            Prim::Mul => "stablehlo.multiply",
        }
    }
}

#[derive(Clone, Copy)]
enum Aval {
    Int,
}

// An operand: variable reference or literal. Copy, 16 bytes (tag + i64).
#[derive(Clone, Copy)]
enum Atom {
    Var(VarId),
    Lit(i64),
}

struct Eqn {
    prim: Prim,
    inputs: [Atom; 2], // all prims binary for now; becomes SmallVec later
    out: VarId,
}

// Flat SSA equation list. VarIds number invars first (0..n_invars), then one
// per eqn output, in order. Avals live in a side table indexed by VarId.
#[pyclass(frozen)]
struct Jaxpr {
    n_invars: u32,
    eqns: Vec<Eqn>,
    outvar: Atom,
    avals: Vec<Aval>,
}

// === Tracing ===

// A trace in progress. No ambient state on the Rust side: the Python shim owns
// a contextvars stack of these ("omnistaging" dispatch — staging is decided by
// whether a trace is active, not by tracer-typed arguments; cf. jax#3370) and
// passes the current one explicitly to every op.
#[pyclass]
struct TraceCtx {
    n_invars: u32,
    eqns: Vec<Eqn>,
    avals: Vec<Aval>,
}

impl TraceCtx {
    fn aval_of(&self, x: Atom) -> Aval {
        match x {
            Atom::Var(v) => self.avals[v as usize],
            Atom::Lit(_) => Aval::Int,
        }
    }
}

#[pymethods]
impl TraceCtx {
    #[new]
    fn new(in_types: Vec<PyRef<'_, Ty>>) -> Self {
        let avals: Vec<Aval> = in_types.iter().map(|t| t.0).collect();
        TraceCtx { n_invars: avals.len() as u32, eqns: vec![], avals }
    }

    fn tracers(&self) -> Vec<Tracer> {
        (0..self.n_invars).map(|id| Tracer { id }).collect()
    }

    fn finish(&mut self, outvar: Atom) -> Jaxpr {
        Jaxpr {
            n_invars: self.n_invars,
            eqns: std::mem::take(&mut self.eqns),
            outvar,
            avals: std::mem::take(&mut self.avals),
        }
    }
}

#[pyclass(frozen)]
struct Tracer {
    id: VarId,
}

#[pymethods]
impl Tracer {
    fn __repr__(&self) -> String {
        format!("Tracer<int>({})", var_name(self.id))
    }
}

// How an op argument arrives from Python: a Tracer unwraps to its Var, a
// Python int becomes a literal. Doing this as a FromPyObject instance means
// op signatures can just say `x: Atom` and the boundary glue enforces it.
impl<'py> FromPyObject<'py> for Atom {
    fn extract_bound(x: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(t) = x.downcast::<Tracer>() {
            Ok(Atom::Var(t.get().id))
        } else if let Ok(i) = x.extract::<i64>() {
            Ok(Atom::Lit(i))
        } else {
            Err(PyTypeError::new_err(format!(
                "expected Tracer or int, got {}",
                x.get_type().name()?
            )))
        }
    }
}

// What an op hands back to Python: a Tracer when staging, an int when eager.
#[derive(IntoPyObject)]
enum Val {
    Tracer(Tracer),
    Int(i64),
}

fn bind(ctx: Option<&Bound<'_, TraceCtx>>, prim: Prim, x: Atom, y: Atom) -> PyResult<Val> {
    match ctx {
        Some(ctx) => {
            let mut ctx = ctx.borrow_mut();
            let aval = prim.abstract_eval([ctx.aval_of(x), ctx.aval_of(y)]);
            let out = ctx.avals.len() as VarId;
            ctx.avals.push(aval);
            ctx.eqns.push(Eqn { prim, inputs: [x, y], out });
            Ok(Val::Tracer(Tracer { id: out }))
        }
        // No active trace: eager evaluation on concrete ints.
        None => match (x, y) {
            (Atom::Lit(a), Atom::Lit(b)) => Ok(Val::Int(prim.eval(a, b))),
            _ => Err(PyValueError::new_err("Tracer used outside of a trace")),
        },
    }
}

// === Python API ===

#[pyfunction]
fn add(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, y: Atom) -> PyResult<Val> {
    bind(ctx, Prim::Add, x, y)
}

#[pyfunction]
fn mul(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, y: Atom) -> PyResult<Val> {
    bind(ctx, Prim::Mul, x, y)
}

#[pyclass(frozen, name = "Type")]
struct Ty(Aval);

#[pymethods]
impl Ty {
    fn __repr__(&self) -> &'static str {
        "int"
    }
}

#[pyfunction]
fn eval(jaxpr: PyRef<'_, Jaxpr>, args: Vec<i64>) -> PyResult<i64> {
    if args.len() != jaxpr.n_invars as usize {
        return Err(PyValueError::new_err(format!(
            "expected {} args, got {}",
            jaxpr.n_invars,
            args.len()
        )));
    }
    let mut env = args; // env[v] = value of VarId v; eqn outputs are sequential
    let get = |env: &[i64], x: Atom| match x {
        Atom::Var(v) => env[v as usize],
        Atom::Lit(l) => l,
    };
    for eqn in &jaxpr.eqns {
        debug_assert_eq!(eqn.out as usize, env.len());
        let val = eqn.prim.eval(get(&env, eqn.inputs[0]), get(&env, eqn.inputs[1]));
        env.push(val);
    }
    Ok(get(&env, jaxpr.outvar))
}

// === Printing ===

fn var_name(v: VarId) -> String {
    // a, b, ..., z, v26, v27, ...
    match v {
        0..=25 => ((b'a' + v as u8) as char).to_string(),
        _ => format!("v{v}"),
    }
}

fn atom_str(x: Atom) -> String {
    match x {
        Atom::Var(v) => var_name(v),
        Atom::Lit(l) => l.to_string(),
    }
}

// SSA name of an atom in the emitted MLIR, materializing literals as
// stablehlo.constant ops. Invars are %arg0..; everything else is %t0..
fn atom_ssa(x: Atom, names: &[String], body: &mut String, tmp: &mut u32) -> String {
    match x {
        Atom::Var(v) => names[v as usize].clone(),
        Atom::Lit(l) => {
            let n = format!("%t{}", *tmp);
            *tmp += 1;
            body.push_str(&format!("    {n} = stablehlo.constant dense<{l}> : tensor<i64>\n"));
            n
        }
    }
}

#[pymethods]
impl Jaxpr {
    // Lower to StableHLO MLIR text. (Aval::Int ↦ tensor<i64>; the only case.)
    fn to_stablehlo(&self) -> String {
        let mut names: Vec<String> = (0..self.n_invars).map(|i| format!("%arg{i}")).collect();
        let (mut body, mut tmp) = (String::new(), 0);
        for eqn in &self.eqns {
            let a = atom_ssa(eqn.inputs[0], &names, &mut body, &mut tmp);
            let b = atom_ssa(eqn.inputs[1], &names, &mut body, &mut tmp);
            let out = format!("%t{tmp}");
            tmp += 1;
            body.push_str(&format!("    {out} = {} {a}, {b} : tensor<i64>\n", eqn.prim.stablehlo()));
            debug_assert_eq!(eqn.out as usize, names.len());
            names.push(out);
        }
        let ret = atom_ssa(self.outvar, &names, &mut body, &mut tmp);
        let args: Vec<String> = (0..self.n_invars).map(|i| format!("%arg{i}: tensor<i64>")).collect();
        format!(
            "module @rustyjax {{\n  func.func @main({}) -> tensor<i64> {{\n{body}    return {ret} : tensor<i64>\n  }}\n}}\n",
            args.join(", ")
        )
    }

    fn __repr__(&self) -> String {
        let invars: Vec<String> = (0..self.n_invars).map(var_name).collect();
        let eqns: Vec<String> = self
            .eqns
            .iter()
            .map(|e| {
                format!(
                    "    {}:int = {} {} {}",
                    var_name(e.out),
                    e.prim.name(),
                    atom_str(e.inputs[0]),
                    atom_str(e.inputs[1])
                )
            })
            .collect();
        format!(
            "{{ lambda {} .\n  let\n{}\n  in {} }}",
            invars.join(" "),
            eqns.join("\n"),
            atom_str(self.outvar)
        )
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TraceCtx>()?;
    m.add_class::<Tracer>()?;
    m.add_class::<Jaxpr>()?;
    m.add_class::<Ty>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add("int_type", Py::new(m.py(), Ty(Aval::Int))?)?;
    Ok(())
}
