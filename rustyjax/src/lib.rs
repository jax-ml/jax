use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

type VarId = u32;

// === IR ===

#[derive(Clone, Copy, PartialEq, Eq)]
enum DType {
    F32,
    I32,
    I64,
}

impl DType {
    fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::I64 => "i64",
        }
    }

    fn is_float(self) -> bool {
        matches!(self, DType::F32)
    }

    fn is_int(self) -> bool {
        !self.is_float()
    }
}

// The abstract value: shape + dtype (jax's ShapedArray).
#[derive(Clone, PartialEq)]
struct Aval {
    shape: Vec<i64>,
    dtype: DType,
}

impl Aval {
    fn scalar(dtype: DType) -> Self {
        Aval { shape: vec![], dtype }
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }

    // "f32[128,512]" (repr) / "tensor<128x512xf32>" (MLIR)
    fn str(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!("{}[{}]", self.dtype.name(), dims.join(","))
    }

    fn mlir(&self) -> String {
        let dims: String = self.shape.iter().map(|d| format!("{d}x")).collect();
        format!("tensor<{}{}>", dims, self.dtype.name())
    }
}

// Numpy-style broadcasting: align trailing dims; 1s stretch.
fn broadcast_shapes(a: &[i64], b: &[i64]) -> Result<Vec<i64>, String> {
    let r = a.len().max(b.len());
    let (pa, pb) = (r - a.len(), r - b.len());
    (0..r)
        .map(|i| {
            let da = if i >= pa { a[i - pa] } else { 1 };
            let db = if i >= pb { b[i - pb] } else { 1 };
            match (da, db) {
                _ if da == db => Ok(da),
                (1, _) => Ok(db),
                (_, 1) => Ok(da),
                _ => Err(format!("cannot broadcast {a:?} with {b:?}")),
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Lit {
    I(i64),
    F(f64),
}

// An operand: variable reference or (scalar) literal. Copy, 16 bytes.
#[derive(Clone, Copy)]
enum Atom {
    Var(VarId),
    Lit(Lit),
}

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Max,
}

// Closed enum of primitives. Static params live in the variants; array
// operands live in the Eqn.
#[derive(Clone)]
enum Prim {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Tanh,
    Sqrt,
    Matmul, // 2D (m,k)@(k,n) or batched 3D (b,m,k)@(b,k,n)
    Take,   // table (n,d), int indices (s,) -> (s,d)
    Reshape { shape: Vec<i64> },
    Transpose { perm: Vec<i64> },
    Reduce { op: ReduceOp, axis: i64, keepdims: bool }, // axis normalized in resolve
}

impl Prim {
    fn is_binop(&self) -> bool {
        matches!(self, Prim::Add | Prim::Sub | Prim::Mul | Prim::Div)
    }

    fn name(&self) -> String {
        match self {
            Prim::Add => "add".into(),
            Prim::Sub => "sub".into(),
            Prim::Mul => "mul".into(),
            Prim::Div => "div".into(),
            Prim::Exp => "exp".into(),
            Prim::Tanh => "tanh".into(),
            Prim::Sqrt => "sqrt".into(),
            Prim::Matmul => "matmul".into(),
            Prim::Take => "take".into(),
            Prim::Reshape { shape } => format!("reshape[{shape:?}]"),
            Prim::Transpose { perm } => format!("transpose[{perm:?}]"),
            Prim::Reduce { op, axis, keepdims } => {
                let op = match op {
                    ReduceOp::Sum => "reduce_sum",
                    ReduceOp::Max => "reduce_max",
                };
                format!("{op}[axis={axis}{}]", if *keepdims { ",keepdims" } else { "" })
            }
        }
    }

    // The shape/type-inference interpreter rule.
    fn abstract_eval(&self, ins: &[Aval]) -> Result<Aval, String> {
        let same_dtype = |a: &Aval, b: &Aval| {
            if a.dtype != b.dtype {
                Err(format!("dtype mismatch: {} vs {}", a.str(), b.str()))
            } else {
                Ok(a.dtype)
            }
        };
        match self {
            Prim::Add | Prim::Sub | Prim::Mul | Prim::Div => {
                let (a, b) = (&ins[0], &ins[1]);
                Ok(Aval { shape: broadcast_shapes(&a.shape, &b.shape)?, dtype: same_dtype(a, b)? })
            }
            Prim::Exp | Prim::Tanh | Prim::Sqrt => {
                if !ins[0].dtype.is_float() {
                    return Err(format!("{} requires a float operand, got {}", self.name(), ins[0].str()));
                }
                Ok(ins[0].clone())
            }
            Prim::Matmul => {
                let (a, b) = (&ins[0], &ins[1]);
                let dtype = same_dtype(a, b)?;
                match (&a.shape[..], &b.shape[..]) {
                    ([m, k1], [k2, n]) if k1 == k2 => Ok(Aval { shape: vec![*m, *n], dtype }),
                    ([ba, m, k1], [bb, k2, n]) if ba == bb && k1 == k2 => {
                        Ok(Aval { shape: vec![*ba, *m, *n], dtype })
                    }
                    _ => Err(format!("matmul shape mismatch: {} @ {}", a.str(), b.str())),
                }
            }
            Prim::Take => {
                let (t, i) = (&ins[0], &ins[1]);
                match (&t.shape[..], &i.shape[..]) {
                    ([_, d], [s]) if i.dtype.is_int() => {
                        Ok(Aval { shape: vec![*s, *d], dtype: t.dtype })
                    }
                    _ => Err(format!("take expects 2D table and 1D int indices, got {} and {}", t.str(), i.str())),
                }
            }
            Prim::Reshape { shape } => {
                let (from, to): (i64, i64) = (ins[0].shape.iter().product(), shape.iter().product());
                if from != to {
                    return Err(format!("cannot reshape {} to {shape:?}", ins[0].str()));
                }
                Ok(Aval { shape: shape.clone(), dtype: ins[0].dtype })
            }
            Prim::Transpose { perm } => {
                let rank = ins[0].rank();
                let mut seen = vec![false; rank];
                for &p in perm {
                    if p < 0 || p as usize >= rank || seen[p as usize] {
                        return Err(format!("bad permutation {perm:?} for {}", ins[0].str()));
                    }
                    seen[p as usize] = true;
                }
                if perm.len() != rank {
                    return Err(format!("bad permutation {perm:?} for {}", ins[0].str()));
                }
                let shape = perm.iter().map(|&p| ins[0].shape[p as usize]).collect();
                Ok(Aval { shape, dtype: ins[0].dtype })
            }
            Prim::Reduce { axis, keepdims, .. } => {
                let a = &ins[0];
                if *axis < 0 || *axis as usize >= a.rank() {
                    return Err(format!("reduce axis {axis} out of range for {}", a.str()));
                }
                let mut shape = a.shape.clone();
                if *keepdims {
                    shape[*axis as usize] = 1;
                } else {
                    shape.remove(*axis as usize);
                }
                Ok(Aval { shape, dtype: a.dtype })
            }
        }
    }
}

struct Eqn {
    prim: Prim,
    inputs: Vec<Atom>,
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

#[pymethods]
impl TraceCtx {
    #[new]
    fn new(in_types: Vec<PyRef<'_, Ty>>) -> Self {
        let avals: Vec<Aval> = in_types.iter().map(|t| t.0.clone()).collect();
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

// Compute input avals (giving scalar literals a dtype: adopt a Var operand's
// dtype, else i64/f32 by literal kind) and normalize negative reduce axes.
fn resolve(ctx: &TraceCtx, mut prim: Prim, args: &[Atom]) -> Result<(Prim, Vec<Aval>), String> {
    let var_dtype = args.iter().find_map(|a| match a {
        Atom::Var(v) => Some(ctx.avals[*v as usize].dtype),
        _ => None,
    });
    let avals = args
        .iter()
        .map(|a| match a {
            Atom::Var(v) => Ok(ctx.avals[*v as usize].clone()),
            Atom::Lit(_) if !prim.is_binop() => {
                Err(format!("literal operands not supported for {}", prim.name()))
            }
            Atom::Lit(l) => {
                let dtype = match (var_dtype, l) {
                    (Some(d), Lit::I(_)) => d,
                    (Some(d), Lit::F(_)) if d.is_float() => d,
                    (Some(d), Lit::F(_)) => {
                        return Err(format!("float literal with {} operand", d.name()))
                    }
                    (None, Lit::I(_)) => DType::I64,
                    (None, Lit::F(_)) => DType::F32,
                };
                Ok(Aval::scalar(dtype))
            }
        })
        .collect::<Result<Vec<Aval>, String>>()?;
    if let Prim::Reduce { axis, .. } = &mut prim {
        if *axis < 0 {
            *axis += avals[0].rank() as i64;
        }
    }
    Ok((prim, avals))
}

#[pyclass(frozen)]
struct Tracer {
    id: VarId,
}

#[pymethods]
impl Tracer {
    fn __repr__(&self) -> String {
        format!("Tracer({})", var_name(self.id))
    }
}

// How an op argument arrives from Python: a Tracer unwraps to its Var, a
// Python number becomes a literal. Doing this as a FromPyObject instance means
// op signatures can just say `x: Atom` and the boundary glue enforces it.
impl<'py> FromPyObject<'py> for Atom {
    fn extract_bound(x: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(t) = x.downcast::<Tracer>() {
            Ok(Atom::Var(t.get().id))
        } else if let Ok(i) = x.extract::<i64>() {
            Ok(Atom::Lit(Lit::I(i)))
        } else if let Ok(f) = x.extract::<f64>() {
            Ok(Atom::Lit(Lit::F(f)))
        } else {
            Err(PyTypeError::new_err(format!(
                "expected Tracer or number, got {}",
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

fn bind(ctx: Option<&Bound<'_, TraceCtx>>, prim: Prim, args: Vec<Atom>) -> PyResult<Val> {
    match ctx {
        Some(ctx) => {
            let mut ctx = ctx.borrow_mut();
            let (prim, in_avals) = resolve(&ctx, prim, &args).map_err(PyValueError::new_err)?;
            let aval = prim.abstract_eval(&in_avals).map_err(PyValueError::new_err)?;
            let out = ctx.avals.len() as VarId;
            ctx.avals.push(aval);
            ctx.eqns.push(Eqn { prim, inputs: args, out });
            Ok(Val::Tracer(Tracer { id: out }))
        }
        // No active trace: eager evaluation on concrete int scalars.
        None => {
            let ints: Option<Vec<i64>> = args
                .iter()
                .map(|a| match a {
                    Atom::Lit(Lit::I(v)) => Some(*v),
                    _ => None,
                })
                .collect();
            match (ints.as_deref(), &prim) {
                (Some([x, y]), Prim::Add) => Ok(Val::Int(x + y)),
                (Some([x, y]), Prim::Sub) => Ok(Val::Int(x - y)),
                (Some([x, y]), Prim::Mul) => Ok(Val::Int(x * y)),
                (Some([x, y]), Prim::Div) if *y != 0 => Ok(Val::Int(x / y)),
                (Some(_), _) => Err(PyValueError::new_err(format!(
                    "{} not supported in eager mode",
                    prim.name()
                ))),
                (None, _) => Err(PyValueError::new_err("Tracer used outside of a trace")),
            }
        }
    }
}

// === Python API ===

macro_rules! binop {
    ($name:ident, $prim:expr) => {
        #[pyfunction]
        fn $name(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, y: Atom) -> PyResult<Val> {
            bind(ctx, $prim, vec![x, y])
        }
    };
}

macro_rules! unop {
    ($name:ident, $prim:expr) => {
        #[pyfunction]
        fn $name(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom) -> PyResult<Val> {
            bind(ctx, $prim, vec![x])
        }
    };
}

binop!(add, Prim::Add);
binop!(sub, Prim::Sub);
binop!(mul, Prim::Mul);
binop!(div, Prim::Div);
binop!(matmul, Prim::Matmul);
binop!(take, Prim::Take);
unop!(exp, Prim::Exp);
unop!(tanh, Prim::Tanh);
unop!(sqrt, Prim::Sqrt);

#[pyfunction]
fn reshape(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, shape: Vec<i64>) -> PyResult<Val> {
    bind(ctx, Prim::Reshape { shape }, vec![x])
}

#[pyfunction]
fn transpose(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, perm: Vec<i64>) -> PyResult<Val> {
    bind(ctx, Prim::Transpose { perm }, vec![x])
}

#[pyfunction]
fn reduce_sum(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, axis: i64, keepdims: bool) -> PyResult<Val> {
    bind(ctx, Prim::Reduce { op: ReduceOp::Sum, axis, keepdims }, vec![x])
}

#[pyfunction]
fn reduce_max(ctx: Option<&Bound<'_, TraceCtx>>, x: Atom, axis: i64, keepdims: bool) -> PyResult<Val> {
    bind(ctx, Prim::Reduce { op: ReduceOp::Max, axis, keepdims }, vec![x])
}

#[pyclass(frozen, name = "Type")]
struct Ty(Aval);

#[pymethods]
impl Ty {
    #[new]
    fn new(shape: Vec<i64>, dtype: &str) -> PyResult<Self> {
        let dtype = match dtype {
            "f32" => DType::F32,
            "i32" => DType::I32,
            "i64" => DType::I64,
            _ => return Err(PyValueError::new_err(format!("unknown dtype: {dtype}"))),
        };
        Ok(Ty(Aval { shape, dtype }))
    }

    fn __repr__(&self) -> String {
        self.0.str()
    }
}

// Reference interpreter; scalar-i64 jaxprs only (arrays go through XLA).
#[pyfunction]
fn eval(jaxpr: PyRef<'_, Jaxpr>, args: Vec<i64>) -> PyResult<i64> {
    let scalar_i64 = |a: &Aval| a.shape.is_empty() && a.dtype == DType::I64;
    if !jaxpr.avals.iter().all(scalar_i64) {
        return Err(PyValueError::new_err(
            "interpreter supports only scalar i64 jaxprs; use compile() for arrays",
        ));
    }
    if args.len() != jaxpr.n_invars as usize {
        return Err(PyValueError::new_err(format!(
            "expected {} args, got {}",
            jaxpr.n_invars,
            args.len()
        )));
    }
    let mut env = args; // env[v] = value of VarId v; eqn outputs are sequential
    let get = |env: &[i64], x: Atom| match x {
        Atom::Var(v) => Ok(env[v as usize]),
        Atom::Lit(Lit::I(l)) => Ok(l),
        Atom::Lit(Lit::F(_)) => Err(PyValueError::new_err("float literal in i64 jaxpr")),
    };
    for eqn in &jaxpr.eqns {
        debug_assert_eq!(eqn.out as usize, env.len());
        let val = match (&eqn.prim, &eqn.inputs[..]) {
            (Prim::Add, &[x, y]) => get(&env, x)? + get(&env, y)?,
            (Prim::Sub, &[x, y]) => get(&env, x)? - get(&env, y)?,
            (Prim::Mul, &[x, y]) => get(&env, x)? * get(&env, y)?,
            (Prim::Div, &[x, y]) if get(&env, y)? != 0 => get(&env, x)? / get(&env, y)?,
            (p, _) => {
                return Err(PyValueError::new_err(format!(
                    "interpreter does not support {}",
                    p.name()
                )))
            }
        };
        env.push(val);
    }
    get(&env, jaxpr.outvar)
}

// === Lowering to StableHLO ===

// "1.000000000e-6"-style literal (MLIR float syntax requires a decimal point).
fn fmt_float(v: f64) -> String {
    format!("{v:.9e}")
}

fn fmt_lit(l: Lit, dtype: DType) -> String {
    match (l, dtype.is_float()) {
        (Lit::I(v), false) => v.to_string(),
        (Lit::I(v), true) => fmt_float(v as f64),
        (Lit::F(v), true) => fmt_float(v),
        (Lit::F(_), false) => unreachable!("resolve() rejects float literals with int operands"),
    }
}

// "array<i64: 1, 0, 2>" (empty: "array<i64>")
fn arr_i64(vals: &[i64]) -> String {
    if vals.is_empty() {
        return "array<i64>".into();
    }
    let vs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
    format!("array<i64: {}>", vs.join(", "))
}

struct Lower<'a> {
    avals: &'a [Aval],
    names: Vec<String>,
    body: String,
    tmp: u32,
}

impl Lower<'_> {
    fn fresh(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp - 1)
    }

    fn push(&mut self, line: String) {
        self.body.push_str("    ");
        self.body.push_str(&line);
        self.body.push('\n');
    }

    // SSA name + aval of an operand, materializing literals as constants.
    fn atom(&mut self, x: Atom, lit_dtype: DType) -> (String, Aval) {
        match x {
            Atom::Var(v) => (self.names[v as usize].clone(), self.avals[v as usize].clone()),
            Atom::Lit(l) => {
                let aval = Aval::scalar(lit_dtype);
                let n = self.fresh();
                self.push(format!(
                    "{n} = stablehlo.constant dense<{}> : {}",
                    fmt_lit(l, lit_dtype),
                    aval.mlir()
                ));
                (n, aval)
            }
        }
    }

    fn broadcast_to(&mut self, name: String, from: &Aval, to: &Aval) -> String {
        if from.shape == to.shape {
            return name;
        }
        let pad = to.shape.len() - from.shape.len();
        let dims: Vec<i64> = (0..from.shape.len() as i64).map(|i| i + pad as i64).collect();
        let n = self.fresh();
        self.push(format!(
            "{n} = \"stablehlo.broadcast_in_dim\"({name}) <{{broadcast_dimensions = {}}}> : ({}) -> {}",
            arr_i64(&dims),
            from.mlir(),
            to.mlir()
        ));
        n
    }

    // Emit one equation; returns the output's SSA name.
    fn emit(&mut self, eqn: &Eqn, out: &Aval) -> String {
        let o = self.fresh();
        match &eqn.prim {
            Prim::Add | Prim::Sub | Prim::Mul | Prim::Div => {
                let mnemonic = match eqn.prim {
                    Prim::Add => "stablehlo.add",
                    Prim::Sub => "stablehlo.subtract",
                    Prim::Mul => "stablehlo.multiply",
                    _ => "stablehlo.divide",
                };
                let (an, aa) = self.atom(eqn.inputs[0], out.dtype);
                let (bn, ba) = self.atom(eqn.inputs[1], out.dtype);
                let an = self.broadcast_to(an, &aa, out);
                let bn = self.broadcast_to(bn, &ba, out);
                self.push(format!("{o} = {mnemonic} {an}, {bn} : {}", out.mlir()));
            }
            Prim::Exp | Prim::Tanh | Prim::Sqrt => {
                let mnemonic = match eqn.prim {
                    Prim::Exp => "stablehlo.exponential",
                    Prim::Tanh => "stablehlo.tanh",
                    _ => "stablehlo.sqrt",
                };
                let (x, _) = self.atom(eqn.inputs[0], out.dtype);
                self.push(format!("{o} = {mnemonic} {x} : {}", out.mlir()));
            }
            Prim::Matmul => {
                let (an, aa) = self.atom(eqn.inputs[0], out.dtype);
                let (bn, ba) = self.atom(eqn.inputs[1], out.dtype);
                let (batch, (lc, rc)) = if aa.rank() == 3 { (vec![0], (2, 1)) } else { (vec![], (1, 0)) };
                self.push(format!(
                    "{o} = \"stablehlo.dot_general\"({an}, {bn}) <{{dot_dimension_numbers = \
                     #stablehlo.dot<lhs_batching_dimensions = {batch:?}, rhs_batching_dimensions = {batch:?}, \
                     lhs_contracting_dimensions = [{lc}], rhs_contracting_dimensions = [{rc}]>}}> : \
                     ({}, {}) -> {}",
                    aa.mlir(),
                    ba.mlir(),
                    out.mlir()
                ));
            }
            Prim::Take => {
                let (tn, ta) = self.atom(eqn.inputs[0], out.dtype);
                let (in_, ia) = self.atom(eqn.inputs[1], DType::I32);
                self.push(format!(
                    "{o} = \"stablehlo.gather\"({tn}, {in_}) <{{dimension_numbers = \
                     #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], \
                     start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, \
                     slice_sizes = {}}}> : ({}, {}) -> {}",
                    arr_i64(&[1, ta.shape[1]]),
                    ta.mlir(),
                    ia.mlir(),
                    out.mlir()
                ));
            }
            Prim::Reshape { .. } => {
                let (x, xa) = self.atom(eqn.inputs[0], out.dtype);
                self.push(format!("{o} = stablehlo.reshape {x} : ({}) -> {}", xa.mlir(), out.mlir()));
            }
            Prim::Transpose { perm } => {
                let (x, xa) = self.atom(eqn.inputs[0], out.dtype);
                self.push(format!(
                    "{o} = \"stablehlo.transpose\"({x}) <{{permutation = {}}}> : ({}) -> {}",
                    arr_i64(perm),
                    xa.mlir(),
                    out.mlir()
                ));
            }
            Prim::Reduce { op, axis, keepdims } => {
                let (x, xa) = self.atom(eqn.inputs[0], out.dtype);
                let init = self.fresh();
                let init_val = match (op, xa.dtype.is_float()) {
                    (ReduceOp::Sum, true) => fmt_float(0.0),
                    (ReduceOp::Sum, false) => "0".into(),
                    (ReduceOp::Max, true) => "0xFF800000".into(), // f32 -inf
                    (ReduceOp::Max, false) => i64::MIN.to_string(),
                };
                let scalar = Aval::scalar(xa.dtype);
                self.push(format!("{init} = stablehlo.constant dense<{init_val}> : {}", scalar.mlir()));
                let mnemonic = match op {
                    ReduceOp::Sum => "stablehlo.add",
                    ReduceOp::Max => "stablehlo.maximum",
                };
                let mut reduced_shape = xa.shape.clone();
                reduced_shape.remove(*axis as usize);
                let reduced = Aval { shape: reduced_shape, dtype: xa.dtype };
                let r = self.fresh();
                self.push(format!(
                    "{r} = stablehlo.reduce({x} init: {init}) applies {mnemonic} across dimensions = [{axis}] : ({}, {}) -> {}",
                    xa.mlir(),
                    scalar.mlir(),
                    reduced.mlir()
                ));
                if *keepdims {
                    self.push(format!("{o} = stablehlo.reshape {r} : ({}) -> {}", reduced.mlir(), out.mlir()));
                } else {
                    return r;
                }
            }
        }
        o
    }
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
        Atom::Lit(Lit::I(l)) => l.to_string(),
        Atom::Lit(Lit::F(l)) => l.to_string(),
    }
}

#[pymethods]
impl Jaxpr {
    fn __len__(&self) -> usize {
        self.eqns.len()
    }

    // Lower to StableHLO MLIR text.
    fn to_stablehlo(&self) -> String {
        let mut lw = Lower {
            avals: &self.avals,
            names: (0..self.n_invars).map(|i| format!("%arg{i}")).collect(),
            body: String::new(),
            tmp: 0,
        };
        for eqn in &self.eqns {
            let out = self.avals[eqn.out as usize].clone();
            let name = lw.emit(eqn, &out);
            debug_assert_eq!(eqn.out as usize, lw.names.len());
            lw.names.push(name);
        }
        let (ret, ret_aval) = lw.atom(self.outvar, DType::I64);
        let args: Vec<String> = (0..self.n_invars as usize)
            .map(|i| format!("%arg{i}: {}", self.avals[i].mlir()))
            .collect();
        format!(
            "module @rustyjax {{\n  func.func @main({}) -> {} {{\n{}    return {ret} : {} \n  }}\n}}\n",
            args.join(", "),
            ret_aval.mlir(),
            lw.body,
            ret_aval.mlir()
        )
    }

    fn __repr__(&self) -> String {
        let invars: Vec<String> = (0..self.n_invars as usize)
            .map(|i| format!("{}:{}", var_name(i as VarId), self.avals[i].str()))
            .collect();
        let eqns: Vec<String> = self
            .eqns
            .iter()
            .map(|e| {
                let ins: Vec<String> = e.inputs.iter().map(|a| atom_str(*a)).collect();
                format!(
                    "    {}:{} = {} {}",
                    var_name(e.out),
                    self.avals[e.out as usize].str(),
                    e.prim.name(),
                    ins.join(" ")
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
    for f in [
        wrap_pyfunction!(add, m)?,
        wrap_pyfunction!(sub, m)?,
        wrap_pyfunction!(mul, m)?,
        wrap_pyfunction!(div, m)?,
        wrap_pyfunction!(matmul, m)?,
        wrap_pyfunction!(take, m)?,
        wrap_pyfunction!(exp, m)?,
        wrap_pyfunction!(tanh, m)?,
        wrap_pyfunction!(sqrt, m)?,
        wrap_pyfunction!(reshape, m)?,
        wrap_pyfunction!(transpose, m)?,
        wrap_pyfunction!(reduce_sum, m)?,
        wrap_pyfunction!(reduce_max, m)?,
        wrap_pyfunction!(eval, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("int_type", Py::new(m.py(), Ty(Aval::scalar(DType::I64)))?)?;
    Ok(())
}
