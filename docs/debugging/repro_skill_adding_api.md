# Adding repro support for a new API

There are **four components** that must be added.

| Component | File | Purpose |
|---|---|---|
| 1. `api_boundary` annotation | Source file of the API | Marks the API so the repro framework can intercept it |
| 2. Trampoline | `jax/_src/repro/trampolines.py` | Intercepts calls and redirects them to the repro boundary function |
| 3. Repro boundary function | `jax/_src/repro/repro_api.py` | The recorded/replayed function that calls the real API |
| 4. Tests | `tests/repro_test.py` | Regression tests verifying collection and replay |

You can look for examples in some past commits:
  - the one that adds pl.core_map support
  - the one that adds pl.emit_pipeline support

## 1. Add `api_boundary` annotation

In the source file where the API is defined, add the `api_boundary` decorator
with a `repro_api_name`. Import `traceback_util` at the top of the file:

```python
from jax._src import traceback_util

@functools.partial(traceback_util.api_boundary,
                   repro_api_name="jax.experimental.pallas.kernel")
def kernel(...):
```

The `repro_api_name` is a dotted string identifying the API. It must match the
name used in the trampoline registration.

## 2. Add a trampoline

In `jax/_src/repro/trampolines.py`, add a function decorated with
`@api_trampoline("repro_api_name")`.

**Pattern A: Direct Call API** (e.g., `jax.jit`, `jax.grad`).
For APIs where the trampoline replaces the call directly:

```python
@api_trampoline("jax.some_api")
def some_api_trampoline(real_api_fun: Callable):
  from jax._src.repro.repro_api import some_api_call

  def some_api_trampoline(*args, **kwargs):
    return some_api_call(*args, **kwargs)
  some_api_trampoline.real_api_fun = real_api_fun
  return some_api_trampoline
```

**Pattern B: Curried/Decorator API** (e.g., `pl.kernel`, `pl.core_map`).
For APIs that return a callable (decorator pattern), the trampoline must
handle the currying. For example, `pl.kernel` can be called as
`pl.kernel(body, out_type=x, mesh=mesh)(operands)` or as
`pl.kernel(out_type=x, mesh=mesh)(body)(operands)`:

```python
@api_trampoline("jax.experimental.pallas.kernel")
def pallas_kernel_trampoline(real_api_fun: Callable):
  from jax._src.repro.repro_api import pallas_kernel_call

  def pallas_kernel_trampoline(
      body=_api.NotSpecified(), out_type=(), **kernel_kwargs):
    kernel_kwargs["out_type"] = out_type
    if isinstance(body, _api.NotSpecified):
      # Decorator factory mode
      def decorator(body):
        def call_with_operands(*operands):
          return pallas_kernel_call(body, kernel_kwargs, *operands)
        return call_with_operands
      return decorator
    else:
      # Direct call mode
      def call_with_operands(*operands):
        return pallas_kernel_call(body, kernel_kwargs, *operands)
      return call_with_operands

  pallas_kernel_trampoline.real_api_fun = real_api_fun
  return pallas_kernel_trampoline
```

**Pattern C: Uncurry Trampoline** (e.g., `pallas_call`). For simpler curried
APIs, use the `uncurry_trampoline` helper by adding an entry to the
`_uncurry_trampolines` list in `trampolines.py`:
```python
("jax.experimental.pallas.pallas_call", "pallas_call"),
```

**Key rule**: The trampoline must set `.real_api_fun = real_api_fun` on the
returned function, so the framework can access the original API.

## 3. Add a repro boundary function

In `jax/_src/repro/repro_api.py`, add a function decorated with
`@partial(repro_boundary, repro_api_name="...")`:

```python
@partial(repro_boundary, repro_api_name="pallas_kernel_call")
def pallas_kernel_call(body: Callable, kernel_kwargs: dict[str, Any],
                       *operands):
  from jax._src.pallas import helpers as pallas_helpers  # type: ignore
  return repro_bypass_wrapper(pallas_helpers.kernel)(
    body, **kernel_kwargs)(*operands)
```

Key rules:
- Use `repro_bypass_wrapper(module.api_function)` to call the real API,
  bypassing repro tracking on the inner call.
- Import the API module inside the function body (not at module level) to
  avoid circular imports.
- The function signature should separate: user functions (body/kernel), API
  configuration (kwargs/params), and data arguments (operands/arrays).

## 4. Add tests

In `tests/repro_test.py`, add tests near the related API tests.

For Pallas APIs that need both interpret mode and native mode, follow this
pattern:

```python
@jtu.parameterized_filterable(
  kwargs=[dict(interpret=interpret)
               for interpret in [False, True]
  ])
def test_pallas_kernel_basic(self, *, interpret: bool):
  mesh = pltpu.create_tensorcore_mesh('x', num_cores=1)

  if jtu.device_under_test() not in ["tpu", "cpu"]:
    self.skipTest("Test runs only on CPU and TPU")

  def body(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1

  @jax.jit
  def g(x):
    return pl.kernel(body, out_type=x, mesh=mesh, interpret=interpret)(x)

  def f(x):
    if interpret or jtu.device_under_test() == "tpu":
      return g(x)
    else:  # On CPU without interpret, we trace or export
      _ = g.trace(x)
      # OR: _ = export.export(g, platforms=("tpu",))(x)
      return 0.

  x = jnp.arange(8 * 128, dtype=jnp.int32).reshape((8, 128))
  with tracker.flags_override(fake_array_threshold=x.size + 1):
    self.collect_and_check(f, x)
```

CPU vs TPU behavior:
- **TPU**: Run both `interpret=True` and `interpret=False` natively.
- **CPU with `interpret=True`**: Run directly if the API supports CPU
  interpret mode, otherwise trace.
- **CPU with `interpret=False`**: Use `jax.export` (if the API supports it
  for the target platform) or `g.trace(x)` (traces the jaxpr without
  lowering).

When to use `g.trace(x)` vs `export.export()`:
- Use `export.export(g, platforms=("cuda",))(x)` for GPU APIs
  (like `plgpu.kernel`).
- Use `export.export(g, platforms=("tpu",))(x)` for TPU APIs **if** the
  lowering works for the kernel.
- Use `g.trace(x)` if the export fails (e.g., due to memory space issues in
  simple test kernels).
