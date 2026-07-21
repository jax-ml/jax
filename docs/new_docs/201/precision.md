---
nosearch: true
---

(jax-201-precision)=
# Matmul precision

<!--* freshness: { reviewed: '2026-07-09' } *-->

On accelerators, matrix multiplication is the workhorse operation — and the
hardware typically offers several ways to run it, trading accuracy for speed:
true `float32` arithmetic, TensorFloat32 on NVIDIA tensor cores, one or
several `bfloat16` passes on TPU, various `float8` modes, and so on. By
default, JAX leans toward speed: `float32` dot products may be computed with
reduced-precision arithmetic internally (`bfloat16` on TPU, TF32 on recent
GPUs). That default is right for most neural-network workloads and surprising
for everything else, so JAX gives you explicit control at every level: per
operation, and globally.

The `precision` argument — accepted by {func}`jax.lax.dot_general`,
{func}`jax.lax.dot`, and the `jax.numpy` functions built on them
(`jnp.dot`, `jnp.matmul`, `@`, `jnp.einsum`, convolutions) — is how you say
what you want. It accepts two kinds of value: a *dot algorithm* (the modern,
precise interface) or a coarse-grained {class}`~jax.lax.Precision` level (the
classic one).

## Dot algorithms: saying exactly what you want

The most direct way to control a dot product is to name the algorithm that
computes it, by passing a {class}`jax.lax.DotAlgorithmPreset` (or its name as
a string) as `precision`:

```python
import jax.numpy as jnp
from jax import lax

y = jnp.dot(x, x, precision="F32_F32_F32")                        # true float32
y = jnp.dot(x, x, precision="BF16_BF16_F32")                      # bf16 inputs, f32 accumulation
y = jnp.dot(x, x, precision=lax.DotAlgorithmPreset.TF32_TF32_F32) # TF32 tensor cores
```

Preset names follow the pattern `LHS_RHS_ACCUM`: the element types the
left- and right-hand sides are rounded to, and the type used for
accumulation. The available presets:

- `DEFAULT` — an algorithm is selected based on input and output types.
- `F32_F32_F32`, `F64_F64_F64` — plain full-precision arithmetic.
- `F16_F16_F16`, `F16_F16_F32` — half-precision inputs, accumulating in
  half or single precision.
- `BF16_BF16_BF16`, `BF16_BF16_F32` — likewise for `bfloat16`.
- `BF16_BF16_F32_X3`, `_X6`, `_X9` — the `_X` suffix means the algorithm
  uses that many `bfloat16` operations to *emulate* higher precision: `_X3`
  approaches `float32` accuracy, `_X6` and `_X9` exceed it, at
  proportionally higher cost.
- `TF32_TF32_F32`, `TF32_TF32_F32_X3` — TensorFloat32, and its 3-operation
  higher-precision emulation.
- `ANY_F8_ANY_F8_F32`, `ANY_F8_ANY_F8_F32_FAST_ACCUM` — any `float8` input
  types, accumulating into `float32`; the `FAST_ACCUM` variant uses faster,
  less accurate accumulation (e.g. cuBLASLt's fast-accumulation mode).
- `ANY_F8_ANY_F8_ANY`, `ANY_F8_ANY_F8_ANY_FAST_ACCUM` — as above, with the
  accumulation type controlled by `preferred_element_type`.

A few properties make this interface pleasant to use in practice:

- **Any input dtypes are accepted.** JAX inserts casts so that the operands
  reach the hardware in the algorithm's storage types — you can pass
  `float32` arrays with `precision="BF16_BF16_F32"` and the rounding is
  handled for you.
- **The output type matches the inputs** (under the usual promotion rules),
  regardless of the algorithm's internal accumulation type — so switching
  algorithms doesn't ripple type changes through your program. To instead
  keep the accumulator's type, use `preferred_element_type`:

  ```python
  x16 = jnp.ones((4, 4), jnp.float16)
  jnp.dot(x16, x16, precision="F16_F16_F32")                # f16 result
  jnp.dot(x16, x16, precision="F16_F16_F32",
          preferred_element_type=jnp.float32)               # keep the f32 accumulator
  ```

- **Autodiff carries the same algorithm through to the backward pass.** The
  transposed dots in the gradient computation carry the *exact same*
  `precision` argument as the primal — not permuted or altered. You can see
  this directly in the jaxpr of a gradient:

  ```python
  def loss(x, w):
    return jnp.sum(jnp.dot(x, w, precision='BF16_BF16_F32'))

  x, w = jnp.ones((4, 8)), jnp.ones((8, 2))
  print(jax.jit(jax.grad(loss, argnums=1)).trace(x, w).jaxpr)
  ```

  ```
  { lambda ; a:f32[4,8] b:f32[8,2]. let
      c:f32[4,2] = dot_general[
        dimension_numbers=(([1], [0]), ([], []))
        precision=BF16_BF16_F32
        preferred_element_type=float32
      ] a b
      _:f32[] = reduce_sum[axes=(0, 1) out_sharding=None] c
      d:f32[4,2] = broadcast_in_dim 1.0:f32[]
      e:f32[2,8] = dot_general[
        dimension_numbers=(([0], [0]), ([], []))
        precision=BF16_BF16_F32
        preferred_element_type=float32
      ] d a
      f:f32[8,2] = transpose[permutation=(1, 0)] e
    in (f,) }
  ```

  Both `dot_general`s — the forward one and the transposed one that computes
  the gradient — carry `precision=BF16_BF16_F32`. (If you need a *different*
  backward-pass algorithm, express that with {func}`jax.custom_vjp`.)
- **Support is platform-dependent, and checked at compile time.** Requesting
  an algorithm the backend can't provide is a compile-time error, not a
  silent fallback — e.g. `precision="F16_F16_F32"` on CPU fails with
  `The precision 'F16_F16_F32' is not supported by dot_general on CPU`.

If no preset fits, you can specify a fully custom algorithm with
{class}`jax.lax.DotAlgorithm`, choosing the operand precision types,
accumulation type, and the number of decomposed operations directly.

## `Precision`: the classic three levels

The `precision` argument also accepts the older, coarser
{class}`jax.lax.Precision` levels, which say *how precise* rather than
*which algorithm*, with device-dependent meanings. They affect only
`float32` computations, and have no effect on CPU:

- `Precision.DEFAULT` (aliases `'default'`, `'fastest'`, and `None`):
  fastest, least accurate. On TPU, computes in `bfloat16`; on GPU, uses
  TF32 where available.
- `Precision.HIGH` (aliases `'high'`, `'bfloat16_3x'`, `'tensorfloat32'`):
  slower, more accurate. On TPU, three `bfloat16` passes; on GPU, TF32.
- `Precision.HIGHEST` (aliases `'highest'`, `'float32'`): slowest, most
  accurate. On TPU, six `bfloat16` passes; on GPU, true `float32`.

```python
jnp.dot(x, x, precision='highest')   # give me real float32, whatever it costs
```

These remain widely used and perfectly serviceable, but when you care about
the exact numerics — for reproducibility, for cross-platform agreement, or
for squeezing out f8/f16 throughput — prefer naming a dot algorithm, which
pins down the computation rather than a device-dependent accuracy level.

## Setting a default globally

To change the default for every dot-like operation that doesn't specify its
own `precision`, use the `jax_default_matmul_precision` config — as a context
manager, a config update, or an environment variable. It accepts the same
values as the `precision` argument, including dot algorithm preset names:

```python
# scoped:
with jax.default_matmul_precision('highest'):
  result = f(x)

# process-wide:
jax.config.update('jax_default_matmul_precision', 'BF16_BF16_F32_X3')
```

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest python train.py
```

A common recipe when debugging suspected numerics problems: run once under
`jax.default_matmul_precision('highest')` — if the mystery disappears, you're
looking at reduced-precision matmul accumulation, not a bug.

## Precision is not dtype

Finally, a distinction worth keeping sharp: everything on this page controls
how dot products are *computed* for given inputs. It's separate from the
choice of dtype your data is *stored* in ({ref}`jax-101-arrays` covers
defaults and {doc}`/type_promotion` the promotion rules). Storing model
parameters or activations in `bfloat16` changes memory footprint and
bandwidth everywhere; `precision` changes arithmetic inside individual
operations. Performance work on accelerators usually involves deciding both.
