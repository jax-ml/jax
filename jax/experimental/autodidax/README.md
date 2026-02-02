# Autodidax

A minimal JAX-like system with a Rust core, based on the [autodidax tutorial](https://jax.readthedocs.io/en/latest/autodidax.html).

## Prerequisites

- **Rust** (1.70+): Install via [rustup](https://rustup.rs/)
- **Python** (3.10+)
- **maturin**: For building Python extensions from Rust

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Install maturin
uv pip install maturin --system
# or: pip install maturin
```

## Building

From this directory:

```bash
cd jax/experimental/autodidax

# Build and install (development mode)
maturin develop

# Or build a release wheel
maturin build --release
```

## Quick Test

```bash
python -c "
import autodidax_core as core
print('add(1, 2) =', core.bind1('add', [1.0, 2.0]))

t = core.RustJvpTracer(3.0, 1.0)
print('JVP tracer: primal=%s, tangent=%s' % (t.primal, t.tangent))
"
```

## Usage

```python
import autodidax_core as _rust

# Basic operations via bind1
result = _rust.bind1('add', [1.0, 2.0])  # 3.0
result = _rust.bind1('mul', [3.0, 4.0])  # 12.0
result = _rust.bind1('sin', [0.0])       # 0.0

# Forward-mode autodiff (JVP) via tracers
def jvp(f, primals, tangents):
    tracers_in = [_rust.RustJvpTracer(p, t) for p, t in zip(primals, tangents)]
    out = f(*tracers_in)
    return out.primal, out.tangent

def square(x):
    return _rust.jvp_bind1('mul', [x, x])

primal, tangent = jvp(square, (3.0,), (1.0,))
# primal=9.0, tangent=6.0 (derivative of x^2 at x=3)

# Build Jaxpr IR
_rust.start_jaxpr_trace()
inputs = _rust.make_jaxpr_tracers(1)
output = _rust.jaxpr_bind1('mul', [inputs[0], inputs[0]])
jaxpr = _rust.finalize_jaxpr(inputs, [output])
print(jaxpr)
# { lambda _0 ; let
#     _1 = mul _0 _0
#   in (_1) }
```

## Primitives

| Name | Signature | Description |
|------|-----------|-------------|
| `add` | `(x, y) -> x + y` | Addition |
| `mul` | `(x, y) -> x * y` | Multiplication |
| `neg` | `(x) -> -x` | Negation |
| `sin` | `(x) -> sin(x)` | Sine |
| `cos` | `(x) -> cos(x)` | Cosine |
| `greater` | `(x, y) -> x > y` | Greater than |
| `less` | `(x, y) -> x < y` | Less than |

## Project Structure

```
autodidax/
├── Cargo.toml         # Rust crate config
├── README.md          # This file
├── __init__.py        # Python package
├── core.py            # JVP & make_jaxpr API
├── primitives.py      # Primitive operations
├── primitives_test.py # Tests
└── src/               # Rust implementation
    ├── lib.rs         # Crate root
    ├── bindings.rs    # PyO3 bindings
    ├── jvp_trace.rs   # Forward-mode autodiff
    ├── jaxpr.rs       # Jaxpr IR
    ├── eval_trace.rs  # Evaluation trace
    ├── primitive.rs   # Primitive definitions
    ├── aval.rs        # Abstract values
    └── tracer.rs      # Tracer types
```
