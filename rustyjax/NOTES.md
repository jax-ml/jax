# rustyjax notes

Goal: prototype JAX internals in Rust behind the existing Python API. Hoping for:
speed; and a test of whether Rust's ADTs/traits make tracing internals clearer
and safer. Status report to JAX team due ~2026-08-14.

## Log

### 2026-08-04 — design discussion + first POC

Design decisions (from discussion of rustc/Cranelift/LLVM/oxc/rust-analyzer precedent):

- **Primitives: closed enum + `match`**, not trait objects (`Box<dyn>`). No serious
  Rust compiler uses trait objects for IR nodes: exhaustiveness checking turns
  "add an op" into a compiler-guided checklist, and enums avoid a heap allocation
  + vtable per node. `enum_dispatch` crate noted as a co-location option if match
  arms sprawl. Open registry (JAX's actual design) only needed for out-of-tree
  primitives; retrofit later if missed.
- **Arity: flat operand lists, structured params.** "Flat vs precise" is the wrong
  axis; "operands vs params" is right (cf. Cranelift Opcode vs InstructionData).
  Enum variants carry static params (`Reduce { axes }`), the Eqn carries operands.
- **IR is a flat SSA equation list** (`Vec<Eqn>`), not a recursive tree — transpose
  walks backward, partial-eval splits the list; trees make both awkward.
- **Avals in a side table** `VarId -> Aval` (rustc MIR `LocalDecls` style), not on
  operands: SSA = one def/many uses, so inline avals duplicate + can disagree;
  bare-`VarId` operands keep `Atom` as `Copy` (8 bytes, no clones on hot paths).
  In an array language, typing = abstract eval = an interpreter; run once at
  construction, memoize on the binder.
- Convention: VarIds number invars first, then one per eqn in order — so eval's
  env is just a `Vec` pushed in order, no hashmap.

POC (`src/lib.rs`, ~230 lines): ints only, add/mul, trace-to-jaxpr + eval
interpreter, eager fallback outside traces, jax-style pretty-printer.
`jax-rust-test.py` passes.

Toolchain facts learned:

- rustup one-liner installs to `~/.cargo` (needs `PATH="$HOME/.cargo/bin:$PATH"`).
- **pyo3** = the Rust↔Python FFI crate; **maturin** = its build tool. No need to
  install maturin: `uv pip install ./rustyjax` drives it via the pyproject build
  backend. Rebuild after Rust edits: `VIRTUAL_ENV=$PWD/.venv uv pip install --reinstall -q .`
- pyo3 ergonomics surprisingly good: `#[pyclass]` structs, `#[pyfunction]`,
  extract `Tracer`-or-`int` args by downcast-then-extract. `#[pyclass(frozen)]`
  avoids interior-mutability ceremony for immutable objects (Tracer, Jaxpr).
- Trace state: Rust `thread_local!` + `RefCell<Option<TraceCtx>>` — the moral
  equivalent of JAX's trace stack, one deep for now.
- Whole thing compiled and passed on the first build. (LLM wrote it, but still.)

Known cut corners / next steps:

- No trace nesting (needed for grad-of-jit etc.) — thread_local holds one ctx.
- Leaked tracers from a finished trace aren't detected (stale VarIds).
- Single output jaxprs; single `int` type; no `SmallVec` yet (all prims binary,
  `[Atom; 2]` is enough).
- Next milestones: more types (float, shaped arrays) to make avals non-trivial;
  a second interpreter (jvp) to stress the enum-of-prims decision; nested traces.

### 2026-08-07 — Rust-side ambient state removed; trace dispatch moved to Python

Found a real bug while walking through the thread_local code: the nested-trace
guard was tautological (`if tr.is_none() { install }; tr.is_some()` — always
true), so nesting would have silently clobbered the outer trace. Compiled fine;
Rust's type system can't see logic errors. Point for "test error paths early,"
mild point against LLM-wrote-it confidence. Fixed, then made moot:

**Redesign: trace context is now an explicit argument** to every Rust function;
the "which trace is active" policy lives in the Python shim as a `contextvars`
stack (`python/rustyjax/__init__.py`). Rust side has zero ambient state — the
`thread_local!`/`RefCell` machinery is gone. Layout is now maturin "mixed":
Python package + `rustyjax._core` extension module.

Design discussion that led here (Dougal's JAX history — good report material):

- Considered ctx-carried-by-tracers (no global anywhere, `bind` finds the trace
  from its args). That's early JAX/autograd's design, and it was *deliberately
  abandoned* for omnistaging (https://github.com/jax-ml/jax/pull/3370): implicit
  constant folding is wrong-by-default, and closure-captured tracers from outer
  traces make args uninspectable when tracing higher-order functions. Ambient
  context is load-bearing, not an implementation wart.
- So: ambient state is required, but it can live on the *Python* side where the
  policy is (contextvars > threading.local: async-aware, token-based nesting).
  Rust keeps mechanism only, all signatures honest. Omnistaging falls out:
  dispatch is "is a trace active," so `add(2, 3)` under a trace stages an eqn.
- Nested traces now work structurally for free (set/reset tokens). Still
  undetected: mixing tracers *across* live traces (needs levels, the other half
  of omnistaging), leaked tracers, reuse of a `finish`ed TraceCtx.

pyo3 notes: non-frozen `#[pyclass]` gives per-object dynamic borrow checking
(`PyRefMut` ≈ RefCell semantics), so "mutable ctx object passed explicitly" is
native pyo3 shape. `Option<&Bound<TraceCtx>>` maps Python `None` cleanly.
`std::mem::take` to move Vecs out of `&mut self` in `finish`.

### 2026-08-10 — declarative FFI boundary

Applied "parameter types describe the contract, glue enforces it" everywhere:

- `impl FromPyObject for Atom` (Tracer-or-int dispatch as a typeclass instance)
  → op signatures are `fn add(ctx: Option<&Bound<TraceCtx>>, x: Atom, y: Atom)`.
- `#[derive(IntoPyObject)] enum Val { Tracer(..), Int(..) }` for the staged-or-
  eager return; no more type-erased `PyObject` returns.
- Constructor takes `Vec<PyRef<Ty>>` (accepts list/tuple, not generators — fine
  for an internal API); pyo3's generated TypeErrors are decent, and it *adds*
  "argument 'x':" context to our custom FromPyObject error for free.
- Consequence: `py: Python` GIL token gone from all op functions; `bind` and
  `finish` are now FFI-free pure IR code. ~15 lines net deleted. Good evidence
  for the is-Rust-clearer question: the boundary is 2 conversion decls, not
  inline dispatch in every function.

### 2026-08-10 (later) — XLA backend running

`traced.compile().eval(args)` now goes through real XLA (CPU), bypassing the
Rust interpreter entirely. Division of labor per design B:

- **Rust: lowering is interpreter #3.** `Prim::stablehlo()` rule + a
  `Jaxpr::to_stablehlo()` pass emitting StableHLO MLIR *text* (~30 lines).
  Literals materialize as `stablehlo.constant` at use sites; `Aval::Int` ↦
  `tensor<i64>`. Text-not-bytecode is fine at this scale and easy to eyeball.
- **Python shim: the XLA session** via jaxlib's bundled CPU PJRT client — no
  need for Rust XLA bindings (xla-rs / PJRT-C-API route deferred; would matter
  for the speed story, not for semantics).

The painful part was jaxlib's private API drift (~5 iterations, each error
shallow but undocumented; grepping jax's own source for usage was the way):

- `client.compile(mlir_text)` → now `compile_and_load(mlir_text,
  executable_devices=DeviceList(...), compile_options=CompileOptions())`.
- `buffer_from_pyval` is gone; array creation is `xc.batched_device_put(aval,
  sharding, [np_array], [device], enable_x64=True)` (without `enable_x64` it
  wants jax's config state: "enable_x64_state is not set").
- The C++ `SingleDeviceSharding` must be *subclassed in Python* to add
  `memory_kind` and `_to_xla_hlo_sharding` — normally jax's subclass does this.
- Takeaway for the report: xla_client is an internal API and behaves like one;
  a real rustyjax should target the PJRT C API directly instead.

Maturin gotcha: python-source files are *copied* into the venv at install; must
reinstall after editing the shim too (or use `uv pip install -e .`).

Verified: original test (12), two-arg fn, constant-output jaxpr, and Rust
interpreter agreement with XLA on the same jaxpr.

### 2026-08-10 (later still) — Python-JAX trace-time baseline

`trace-benchmark.py` (repo root): decoder-only transformer fwd pass in plain
jnp (no flax; hand-written layernorm/softmax so rustyjax can mirror it
op-for-op). 8 layers, d_model=512, seq=128 → 648-eqn jaxpr. On this machine
(M-series mac, CPU), jax@main:

- true trace (caches cleared): **~12.5 ms = ~52k eqns/sec = ~19 µs/eqn** ← the
  number rustyjax has to beat
- warm "trace": 0.39 ms — but that's jax's jaxpr cache hitting, not tracing
  (30x gap; found by accident, worth knowing that modern jax memoizes whole
  jaxprs keyed on fn identity + avals)
- repo's own benchmarks/tracing_benchmark.py exists but is pallas/TPU-focused
  + needs google_benchmark; borrowed its clear_caches methodology instead.

To run the same model through rustyjax we need ~15 more primitives (dot_general
or matmul, reshape, transpose, gather/take, reduce_{sum,max}, exp, rsqrt, sub,
div, where/select, broadcast, tanh for gelu) + shaped f32 avals — that's the
real next milestone, and it forces the Aval::Int → ShapedArray generalization
anyway.

### 2026-08-10 (evening) — transformer traces through rustyjax; 110x

Extended the core: `Aval = {shape: Vec<i64>, dtype}` (f32/i32/i64), 13 prims
(elementwise +-*/ with numpy broadcasting, exp/tanh/sqrt, matmul 2D+batched-3D,
take, reshape/transpose/reduce with params in the enum variants — the
"operands flat, params structured" design finally exercised), scalar literals
made dtype-polymorphic (adopt the var operand's dtype). Lowering handles
broadcast insertion, dot_general/gather/reduce in StableHLO text. Old
interpreter demoted to scalar-i64-only reference. `trace-benchmark-rusty.py`
mirrors the transformer op-for-op in prefix style.

**Results** (same machine, identical model after the mask/softmax/gelu tweaks,
8 layers / d_model 512 / seq 128):

- python jax true trace: 616 eqns, 10.8 ms (17.5 us/eqn)
- python jax jaxpr-cache hit: 0.38 ms (not really tracing)
- **rustyjax: 487 eqns, 0.11 ms (0.23 us/eqn) — ~98x vs true trace, ~3x vs
  jax's cache hit**

2026-08-13: merged the two benchmark scripts into one `trace_benchmark.py`
(Dougal: divergence risk). Model written once against a 13-op backend
namespace; per-system code is just the op table (jnp lambdas vs rj functions)
+ trace/execute entry points. Numbers unchanged (jax 595 eqns / 11.1 ms /
18.7 us/eqn — slightly fewer eqns now that jax also traces the explicit mean
formula; rustyjax 487 eqns / 0.12 ms / 0.24 us/eqn; numerics still 2.6e-8).
Benchmark-methodology gotcha caught in review: first version rebuilt the 102
rj.Type objects inside the timed region (+0.13 ms, 2x on rustyjax's number!) —
at these speeds the harness is as easy to get wrong as the code under test.
- numerics: both executed via XLA CPU on same inputs, max|diff| 2.6e-8 on
  scale-0.05 outputs → same computation. (Eqn counts differ because jnp emits
  convert/broadcast/integer_pow bookkeeping; numeric equivalence through XLA is
  the meaningful check, not HLO text diff.)
- hand-rolled StableHLO emission (dot_general/gather/reduce generic+pretty
  syntax) parsed and compiled by jaxlib's bundled MLIR on the first try.

Fairness caveats for the report: rustyjax's 0.11 ms still includes Python-side
per-op dispatch (shim fn + contextvar.get). But rustyjax also does less than
jax's tracing (no pytrees, no weak types, no jit-dispatch layers, no source
info tracking) — so 110x is "prototype vs product", not like-for-like. Still:
the per-eqn budget (0.23 us incl. a Python call) says the Rust core itself is
nearly free; Python call overhead is now the floor.

Decisions made unilaterally (flagged to Dougal):

- causal mask + tokens are *arguments*, not closed-over constants → no array
  constants in the IR yet (needs a const pool eventually).
- no pytrees: trace takes a flat list of types; the twin flattens by hand.
  Pytree flattening is Python-shim policy when it comes.
- no operator overloading on Tracer yet (needs ctx access from dunder methods —
  design-B wrinkle); model written in prefix style.
- dtype rules: strict equality, no promotion; scalar lits adopt var dtype.
- matmul: 2D and batch-1 3D only. take: 2D table, 1D int indices only.
