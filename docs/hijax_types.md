---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst,py:light
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
---

```{raw-cell}

---
Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---
```

(hijax-types)=
# Defining new JAX types with hijax

JAX's built-in currency is the array: functions you transform take arrays in
and produce arrays out, and every intermediate the tracing machinery sees has
an array type like `f32[3,4]`. When you want to work with aggregate data, the
usual tool is a
[pytree](https://docs.jax.dev/en/latest/working-with-pytrees.html): you
bundle arrays into containers, and JAX transparently flattens the bundle
into its array leaves at every boundary.

But sometimes transparency is exactly what you don't want. Some data is best
modeled as a new *type*, with its own identity:

* it should appear in jaxprs as a single value of a single type, not as a
  spray of array leaves;
* it has internal invariants, so users should only produce and consume it
  through a fixed set of operations, rather than by freely constructing or
  pattern-matching its components;
* its *tangent type* may differ from its primal structure, so that
  derivatives with respect to it aren't just "the same pytree, but for
  tangents";
* it may have its own notion of batching under `vmap`.

Hijax types (or "hi types") provide this. You subclass `HiType` to define
the type, register a Python class as carrying values of that type, and write
hijax primitives whose input and output types mention the new type. This
document walks through the whole story with one running example: a
quantized array type.

We'll assume some familiarity with hijax primitives; see
{ref}`hijax-custom-derivatives` for an introduction to them. Like everything
hijax, this is experimental: expect imports from `jax.experimental.hijax`,
and expect the APIs to evolve.

### TL;DR

* Subclass `HiType` and implement `lo_ty`, `lower_val`, and `raise_val` to
  say how the type and its values lower to ordinary ("lojax") arrays, then
  call `register_hitype` to associate your value class with your type.
* Write `VJPHiPrimitive` subclasses whose `in_avals`/`out_aval` mention the
  new type; these are the only way values of the type get produced and
  consumed.
* For autodiff, implement `to_tangent_aval` on the type, and VJP/JVP rules
  on the primitives.
* For `vmap`, implement `dec_rank` and `inc_rank` on the type along with a
  `MappingSpec` subclass of your own design, and `batch` rules on the
  primitives. Mapped-over hi type arguments require an explicit `axis_size`
  and spec-valued `in_axes`/`out_axes` entries.

## Example: quantized arrays

Say we want to work with arrays quantized to `int8`. A quantized array is
really a pair of arrays: the `int8` values, and a floating point scale
shared by each row (that is, we quantize along the last axis, one scale per
row, as in common per-row/per-channel quantization schemes):

```{code-cell}
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
# (8 CPU devices, for the shard_map section at the end)

from dataclasses import dataclass

import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class QArray:
  qvalue: jax.Array   # int8[*leading, n]
  scale: jax.Array    # f32[*leading]
```

We could register `QArray` as a pytree and be done. But consider what we'd
give up:

* **Invariants.** The two components are coupled: `scale` must have the
  shape of `qvalue` minus its last axis, and `qvalue` is only meaningful
  together with its `scale`. As a pytree, nothing stops code from crossing
  the streams; under transformations, JAX itself sees only independent
  leaves.
* **Types in jaxprs.** As a pytree, a quantized array appears in traced
  code as two unrelated array values. We'd rather see one value, of one
  type, so jaxprs say what they mean.
* **Tangents.** A quantized array's values live on a discrete grid, so it
  makes no sense to perturb them along the grid. But a pytree's tangent
  type is forced to be the pytree of its leaves' tangent types — and the
  tangent type of an integer array like `qvalue` is a `float0` array,
  which can only carry a trivial payload. So as a pytree, a quantized
  array would admit no useful perturbations at all. What we want is to
  choose a tangent type for the quantized array as a whole, such as the
  *continuous* `f32` arrays that the quantized values approximate.

So instead we'll make `QArray` a hijax type.

## The type

A hijax type is a subclass of `HiType`. The required core is small:

* `lo_ty` says which lojax (array) types make up the type;
* `lower_val` and `raise_val` convert values to and from that list of
  arrays;
* the type must be hashable and comparable for equality (a frozen dataclass
  gives us both).

This is like the pytree flatten/unflatten interface, but it lives at the
level of *types*: given only the type, JAX can compute the lowered types,
without needing a value in hand.

```{code-cell}
from jax.experimental.hijax import HiType, ShapedArray, register_hitype

@dataclass(frozen=True)
class QArrayTy(HiType):
  shape: tuple[int, ...]

  # lowering: which array types make up this type, and how values convert
  def lo_ty(self):
    return [ShapedArray(self.shape, jnp.dtype('int8')),
            ShapedArray(self.shape[:-1], jnp.dtype('float32'))]
  def lower_val(self, q):
    return [q.qvalue, q.scale]
  def raise_val(self, qvalue, scale):
    return QArray(qvalue, scale)

  # autodiff: tangents of quantized arrays are plain float arrays (see below)
  def to_tangent_aval(self):
    return ShapedArray(self.shape, jnp.dtype('float32'))

  # printing, e.g. in jaxprs
  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    return f'q8[{",".join(map(str, self.shape))}]'
  __repr__ = str_short

register_hitype(QArray, lambda q: QArrayTy(q.qvalue.shape))
```

The `register_hitype` call associates the value class with the type: its
second argument computes the type of any given value, analogous to how
`jax.typeof` maps an array to its `ShapedArray` type. Indeed after
registration, `jax.typeof` works on `QArray`s, and JAX transformations
accept them anywhere a value is expected.

## The primitives

With a pytree, users construct and take apart values freely. With a hijax
type, values are produced and consumed only by hijax primitives whose
declared types mention the new type. That's where invariants get enforced:
if every primitive preserves them, they always hold.

Our two primitives are `quantize` and `dequantize`, written with the
`VJPHiPrimitive` API from {ref}`hijax-custom-derivatives`. Each declares
its input and output types, gives its implementation in `expand`, and
(looking ahead to autodiff) carries a straight-through-estimator VJP rule:

```{code-cell}
from jax.experimental.hijax import VJPHiPrimitive

class Quantize(VJPHiPrimitive):
  def __init__(self, x_aval):
    if x_aval.dtype != jnp.dtype('float32'): raise TypeError(x_aval.dtype)
    self.in_avals = (x_aval,)
    self.out_aval = QArrayTy(x_aval.shape)
    self.params = {}
    super().__init__()

  def expand(self, x):
    scale = jnp.max(jnp.abs(x), axis=-1) / 127.
    qvalue = jnp.round(x / scale[..., None]).astype(jnp.int8)
    return QArray(qvalue, scale)

  # straight-through estimator: differentiate as if it's the identity
  def vjp_fwd(self, nzs_in, x):
    return self(x), None

  def vjp_bwd_retval(self, _res, g):
    return (g,)

class Dequantize(VJPHiPrimitive):
  def __init__(self, q_aval):
    self.in_avals = (q_aval,)
    self.out_aval = ShapedArray(q_aval.shape, jnp.dtype('float32'))
    self.params = {}
    super().__init__()

  def expand(self, qx):
    return qx.qvalue.astype('float32') * qx.scale[..., None]

  def vjp_fwd(self, nzs_in, qx):
    return self(qx), None

  def vjp_bwd_retval(self, _res, g):
    return (g,)

def quantize(x):
  return Quantize(jax.typeof(x))(x)

def dequantize(qx):
  return Dequantize(jax.typeof(qx))(qx)
```

Notice that `Quantize`'s `out_aval` and `Dequantize`'s `in_avals` are
`QArrayTy`s: the new type appears in primitive type signatures just like
array types do. Also notice `expand` freely constructs and inspects the
`QArray` value class; primitive implementations are inside the abstraction
boundary.

Everything works eagerly:

```{code-cell}
x = jnp.array([[1., 2., 3.],
               [4., -5., 6.]])

qx = quantize(x)
print(qx)
print(jax.typeof(qx))
print(dequantize(qx))
```

## Hi types in jaxprs

When we trace, the quantized array appears as a single value of type
`q8[2,3]`, produced by one equation and consumed by another:

```{code-cell}
jax.make_jaxpr(lambda x: dequantize(quantize(x)))(x)
```

Compare to the pytree approach, where the same computation would show four
array-typed intermediates with no indication that they pair up. The hi type
only disappears at lowering time, when `expand` is traced and each
`q8[...]`-typed value is expanded into the array components given by
`lo_ty`.

`jit` works, with quantized arrays as arguments, results, and
intermediates:

```{code-cell}
print(jax.jit(lambda x: dequantize(quantize(x)))(x))   # QArray internal

qx2 = jax.jit(quantize)(x)                             # QArray result
print(jax.typeof(qx2))

print(jax.jit(dequantize)(qx2))                        # QArray argument
```

## Autodiff and tangent types

Here's where hi types earn their keep. On the type, we implemented

```python
  def to_tangent_aval(self):
    return ShapedArray(self.shape, jnp.dtype('float32'))
```

which says: the tangent type of a quantized array is a plain `f32` array.
No pytree can express this: a pytree's tangent type is always the pytree
of its leaves' tangent types, and for the `int8` leaf `qvalue` that means
a trivial `float0` tangent.

Together with the straight-through VJP rules on the primitives, gradients
flow through quantization as if it were the identity:

```{code-cell}
def f(x):
  return jnp.sum(dequantize(quantize(x)))
```

```{code-cell}
print(jax.grad(f)(x))
```

And differentiating with respect to a quantized array input produces a
plain float array, as the tangent type dictates:

```{code-cell}
def g(qx):
  return jnp.sum(dequantize(qx) ** 2)

print(jax.grad(g)(qx))
print(jax.typeof(jax.grad(g)(qx)))
```

Notice that making the tangent type an `f32` array was a *choice*, and
there's a real design space here. We could instead have made the tangent
type of `QArrayTy` be `QArrayTy` itself, so that tangents and cotangents
are quantized too — a different tradeoff, sensible for different
applications. (For that choice, since the tangent type is then a hi type,
we'd also implement `vspace_zero` and `vspace_add` on it so autodiff can
instantiate and accumulate cotangents.) This flexibility is why hi types
are a user extension point: for each piece of JAX — tracing, lowering,
autodiff, and batching — you set up how your type participates, however
your situation needs.

## `vmap` and mapping specs

What does it mean to map over a quantized array? For arrays, `vmap`'s
`in_axes` and `out_axes` are axis indices, and JAX can infer the mapped
axis size from the argument's shape. For a general hi type, JAX doesn't
guess: *you* define a "mapping spec" type that says how values of your type
are mapped, users pass instances of it as `in_axes`/`out_axes` entries, and
they pass `axis_size` explicitly when it can't be inferred from an array
argument.

For our quantized arrays, thanks to the per-row scales, a batch of
`QArray`s is just a bigger `QArray`: stacking `n` quantized arrays of type
`q8[2,3]` along a new leading axis gives a `q8[n,2,3]`, with `qvalue` of
shape `(n, 2, 3)` and `scale` of shape `(n, 2)`. So the only mapping notion
we need is "the leading axis," and our spec type doesn't need to carry any
data at all:

```{code-cell}
from jax.experimental.hijax import MappingSpec

@dataclass(frozen=True)
class QArraySpec(MappingSpec):
  pass  # QArrays are only mapped along their leading axis
```

(Specs can be as rich as your type demands. A tuple-like hi type might use
a spec carrying one axis per component; see the `TupSpec` example in
`tests/hijax_test.py`.)

On the type, we implement `dec_rank` and `inc_rank`, the hi type analogues
of "remove the mapped axis" and "add the mapped axis." They take the axis
size and a spec, and return the element type and the batched type,
respectively:

```{code-cell}
def qarray_dec_rank(self, size, spec):
  assert isinstance(spec, QArraySpec) and self.shape[0] == size
  return QArrayTy(self.shape[1:])

def qarray_inc_rank(self, size, spec):
  assert isinstance(spec, QArraySpec)
  return QArrayTy((size, *self.shape))

QArrayTy.dec_rank = qarray_dec_rank
QArrayTy.inc_rank = qarray_inc_rank
```

(We're attaching methods to the class as we go, notebook-style; in real
code these would just be more methods in the `class QArrayTy` definition.)

On the primitives, we implement `batch` rules. A `batch` rule receives the
batched arguments along with their mapping specs (`None` for unbatched
arguments, an integer axis for batched array arguments, and a spec instance
for batched hi type arguments), and returns the batched result along with
its mapping spec. Note that a rule should be prepared for any combination
of batched and unbatched arguments:

```{code-cell}
def quantize_batch(self, axis_data, args, in_dims):
  x, = args
  d, = in_dims
  if d is None:
    return quantize(x), None
  x = jnp.moveaxis(x, d, 0)
  return quantize(x), QArraySpec()
Quantize.batch = quantize_batch

def dequantize_batch(self, axis_data, args, in_dims):
  qx, = args
  d, = in_dims
  if d is None:
    return dequantize(qx), None
  assert isinstance(d, QArraySpec)
  return dequantize(qx), 0
Dequantize.batch = dequantize_batch
```

Because per-row quantization applies at any rank, both rules can just apply
the unbatched operation to the stacked value — the hallmark of a type whose
batches are values of the same type family.

Now we can `vmap`. Mapping *to* a quantized array output, the axis size is
inferred from the array argument as usual, and we pass a spec for
`out_axes`:

```{code-cell}
xs = jnp.arange(24., dtype='float32').reshape(4, 2, 3)

qxs = jax.vmap(quantize, out_axes=QArraySpec())(xs)
print(jax.typeof(qxs))
print(qxs.qvalue.shape, qxs.scale.shape)
```

Mapping *over* a quantized array input, we pass a spec for `in_axes` — and
since there's no array argument to infer the axis size from, we must pass
`axis_size` explicitly:

```{code-cell}
xs_roundtrip = jax.vmap(dequantize, in_axes=QArraySpec(), axis_size=4)(qxs)
print(jax.typeof(xs_roundtrip))
```

All the usual compositions work — `vmap` of `jit`,

```{code-cell}
print(jax.typeof(jax.vmap(jax.jit(dequantize), in_axes=QArraySpec(),
                          axis_size=4)(qxs)))
```

`vmap` of `grad`, and so on:

```{code-cell}
def norm_quantized(x):
  return jnp.sum(dequantize(quantize(x)) ** 2)

print(jax.vmap(jax.grad(norm_quantized))(xs).shape)
```

## `shard_map` and partition specs

Finally, sharding. What does it mean to partition a quantized array across
devices? Once again the components move together: if we shard the rows,
each device should hold a `QArray` of its rows' quantized values *and*
their scales. And once again there's a design choice to make. We'll say a
quantized array can be sharded along its leading axes only, so the
quantized axis stays whole and every row travels with its scale.

For `shard_map`, we express this with an `HiPspec` subclass — the
partition spec analogue of the `MappingSpec` above. Users pass instances
of it as `in_specs`/`out_specs` entries, and its `to_lo` method says how
it translates to one `jax.P` partition spec per lowered component (in
`lo_ty` order):

```{code-cell}
from jax.experimental.hijax import HiPspec

@dataclass(frozen=True)
class QArrayP(HiPspec):
  spec: jax.P  # partitioning of the leading axes; the last axis stays whole

  def to_lo(self):
    return (self.spec, self.spec)  # qvalue and scale shard together
```

(Since `scale` has one axis fewer than `qvalue`, handing both components
the same partition spec says exactly that the leading axes shard together
while the trailing axis of `qvalue` is untouched.)

On the type, `shard` and `unshard` compute the per-device shard type from
the global type and vice versa, delegating to the component types:

```{code-cell}
def qarray_shard(self, mesh, manual_axes, check_vma, spec):
  qvalue_ty, _ = self.lo_ty()
  qspec, _ = spec.to_lo()
  return QArrayTy(qvalue_ty.shard(mesh, manual_axes, check_vma, qspec).shape)

def qarray_unshard(self, mesh, check_vma, spec):
  qvalue_ty, _ = self.lo_ty()
  qspec, _ = spec.to_lo()
  return QArrayTy(qvalue_ty.unshard(mesh, check_vma, qspec).shape)

QArrayTy.shard = qarray_shard
QArrayTy.unshard = qarray_unshard
```

Now quantized arrays can cross `shard_map` boundaries in either direction.
(This is where we use the 8 CPU devices requested in this document's first
cell.) Producing a sharded quantized array:

```{code-cell}
mesh = jax.make_mesh((4,), ('i',))
jax.set_mesh(mesh)

rows = jax.device_put(jnp.arange(24., dtype='float32').reshape(8, 3),
                      jax.P('i'))

@jax.jit
@jax.shard_map(in_specs=jax.P('i'), out_specs=QArrayP(jax.P('i')))
def quantize_shards(x):
  assert jax.typeof(x).shape == (2, 3)   # each device sees two rows
  return quantize(x)

qrows = quantize_shards(rows)
print(jax.typeof(qrows))
print(qrows.qvalue.sharding.spec, qrows.scale.sharding.spec)
```

And consuming one, where each device sees a per-shard `QArray` of its own
rows:

```{code-cell}
@jax.jit
@jax.shard_map(in_specs=QArrayP(jax.P('i')), out_specs=jax.P('i'))
def dequantize_shards(qx):
  assert jax.typeof(qx) == QArrayTy((2, 3))  # a per-device QArray shard
  return dequantize(qx)

print(jnp.max(jnp.abs(dequantize_shards(qrows) - rows)))
```

Because scales are per-row, quantizing shard-by-shard agrees exactly with
quantizing globally — the same property that made batching pleasant:

```{code-cell}
qrows_global = quantize(rows)
assert (qrows.qvalue == qrows_global.qvalue).all()
assert (qrows.scale == qrows_global.scale).all()
```

(For autodiff *through* a `shard_map`, there's a bit more to implement:
`to_tangent_spec` and `to_ct_spec` on the spec type, and `nospec` on the
hi type, which is used to shard autodiff residuals.)

## What we haven't covered

A few more corners of the interface: types can implement
`leading_axis_spec` so that hi type values can be carried through
`jax.lax.scan`, and on the primitive side there are hooks for customizing
rematerialization and dead code elimination.

As ever with hijax, `tests/hijax_test.py` is a good source of worked
examples, and {ref}`hijax-custom-derivatives` covers the primitive-side
API — including JVP rules, symbolic zeros, and custom linearization — in
more depth.
