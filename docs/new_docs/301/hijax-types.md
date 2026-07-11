---
jupytext:
  formats: md:myst
  notebook_metadata_filter: nosearch
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
nosearch: true
---

(jax-301-hijax-types)=
# Defining new JAX types with hijax

<!--* freshness: { reviewed: '2026-07-10' } *-->

JAX's built-in currency is the array: functions you transform take arrays in
and produce arrays out, and every intermediate the tracing machinery sees has
an array type like `f32[3,4]`. When you want to work with aggregate data, the
usual tool is a
pytree ({ref}`jax-101-pytrees`): you
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
* it may have its own notion of batching under `vmap`;
* it can carry sharding information in the type, participating in JAX's
  explicit sharding mode.

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
* For sharding in types (explicit mode), record sharding data on your type
  (e.g. a `NamedSharding` field), consume it in `lo_ty`, and propagate it
  in your primitives' typing rules.

## Example: quantized arrays

Say we want to work with arrays quantized to `int8`. A quantized array is
really a pair of arrays: the `int8` values, and a floating point scale
shared by each row (that is, we quantize along the last axis, one scale per
row, as in common per-row/per-channel quantization schemes):

```{code-cell}
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
# (8 CPU devices, for the sharding sections at the end)

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

We also give the type a `sharding` field, recording how values are
partitioned across devices; it can be ignored until the sharding sections
near the end of this document. We reuse JAX's `NamedSharding` to describe
the partitioning of the `qvalue` component, and derive `scale`'s
partitioning from it by dropping the last axis. Nothing about the field is
special to JAX, which never interprets it: only our own methods consume
it, chiefly `lo_ty`, which stamps the component types with their
shardings. You're free to track sharding information on your type with an
object of your own design instead, so long as you consume it the same
way.

```{code-cell}
from jax.experimental.hijax import HiType, ShapedArray, register_hitype
from jax.sharding import NamedSharding

@dataclass(frozen=True)
class QArrayTy(HiType):
  shape: tuple[int, ...]
  sharding: NamedSharding  # qvalue's sharding; scale's is derived from it

  # lowering: which array types make up this type, and how values convert
  def lo_ty(self):
    scale_sharding = self.sharding.update(spec=jax.P(*self.sharding.spec[:-1]))
    return [ShapedArray(self.shape, jnp.dtype('int8'),
                        sharding=self.sharding),
            ShapedArray(self.shape[:-1], jnp.dtype('float32'),
                        sharding=scale_sharding)]
  def lower_val(self, q):
    return [q.qvalue, q.scale]
  def raise_val(self, qvalue, scale):
    return QArray(qvalue, scale)

  # autodiff: tangents of quantized arrays are plain float arrays (see below)
  def to_tangent_aval(self):
    return ShapedArray(self.shape, jnp.dtype('float32'),
                       sharding=self.sharding)

  # printing, e.g. in jaxprs
  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    dims = [str(d) if p is None else f'{d}@{p}'
            for d, p in zip(self.shape, self.sharding.spec)]
    return f'q8[{",".join(dims)}]'
  __repr__ = str_short

register_hitype(QArray, lambda q: QArrayTy(q.qvalue.shape,
                                           jax.typeof(q.qvalue).sharding))
```

The `register_hitype` call associates the value class with the type: its
second argument computes the type of any given value, analogous to how
`jax.typeof` maps an array to its `ShapedArray` type. (Ours reads both the
shape and the sharding off the `qvalue` component — every array carries a
sharding, trivial when no mesh is in play.) Indeed after registration,
`jax.typeof` works on `QArray`s, and JAX transformations accept them
anywhere a value is expected.

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
    self.out_aval = QArrayTy(x_aval.shape, x_aval.sharding)
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
    self.out_aval = ShapedArray(q_aval.shape, jnp.dtype('float32'),
                                sharding=q_aval.sharding)
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

### Primitives outside, container inside

The `expand` methods above construct `QArray` values and read their
attributes directly. It's important that this kind of direct container
manipulation happen *only* in `expand` (and in the type's own methods,
like `lower_val` and `raise_val`). Everywhere else — in particular, in
any function you might `jit`, differentiate, or `vmap` — hi values should
be produced and consumed only by applying primitives.

The reason is what traced code actually sees. Under a trace, a quantized
array is not a `QArray` instance: it's a `Tracer` of type `q8[...]`. So
reading an attribute in traced code fails —

```{code-cell}
try:
  jax.jit(lambda qx: qx.qvalue)(qx)
except AttributeError as e:
  print('AttributeError:', e)
```

and, worse, calling the constructor on traced arrays doesn't fail right
away. It smuggles `Tracer`s inside a container that JAX treats as one
opaque concrete value, and the mistake surfaces later, as a confusing
error far from its cause (here a missing constant handler; under `grad`
it's a leaked-tracer error):

```{code-cell}
def bad_quantize(x):
  scale = jnp.max(jnp.abs(x), axis=-1) / 127.
  return QArray(jnp.round(x / scale[..., None]).astype('int8'), scale)

try:
  jax.jit(bad_quantize)(x)
except TypeError as e:
  print('TypeError:', e)
```

In `expand` it's a different story: by the time `expand` runs, JAX has
committed to implementing the primitive in terms of the type's lojax
components, and its `QArray` arguments are genuine `QArray` instances
(holding lojax values — possibly traced ones). There, manipulating the
value as a plain container is exactly right.

(The top-level peeks at attributes like `qx.qvalue` elsewhere in this
document are fine for the same reason eager `expand` is fine: they run
eagerly, on concrete values. But inside any function that might get
traced, stick to primitives.)

### A more realistic op: dense × quantized matmul

Conversion ops alone make for a thin API: in practice, a quantized array
type earns its keep in ops that consume the type directly. The classic
example is an inference-style matmul, where the activations `x` are an
ordinary dense `f32` array and the weights are quantized:

```{code-cell}
class MatmulQ(VJPHiPrimitive):
  def __init__(self, x_aval, q_aval):
    if not (isinstance(q_aval, QArrayTy) and len(x_aval.shape) == 2 and
            len(q_aval.shape) == 2 and x_aval.shape[1] == q_aval.shape[0]):
      raise TypeError(f'bad matmul_q operand types: {x_aval} @ {q_aval}')
    self.in_avals = (x_aval, q_aval)
    x_spec, q_spec = x_aval.sharding.spec, q_aval.sharding.spec
    if x_spec[1] is not None or q_spec[0] is not None:
      raise TypeError('matmul_q requires unsharded contraction axes, got '
                      f'{x_aval} @ {q_aval}')
    out_sharding = x_aval.sharding.update(spec=jax.P(x_spec[0], q_spec[1]))
    self.out_aval = ShapedArray((x_aval.shape[0], q_aval.shape[1]),
                                jnp.dtype('float32'), sharding=out_sharding)
    self.params = {}
    super().__init__()

  def expand(self, x, qw):
    # fold the per-row scales into the dense operand, then apply one matmul
    # directly against the int8 payload
    return (x * qw.scale) @ qw.qvalue.astype(jnp.float32)

  def vjp_fwd(self, nzs_in, x, qw):
    return self(x, qw), (x, qw)

  def vjp_bwd_retval(self, res, g):
    x, qw = res
    w = dequantize(qw)   # rules are traced code: use primitives here
    # cotangents live where the primals live, so use the primal operands'
    # shardings to disambiguate the (possibly sharded) contractions
    return (jnp.matmul(g, w.T, out_sharding=jax.typeof(x).sharding),
            jnp.matmul(x.T, g, out_sharding=jax.typeof(w).sharding))

def matmul_q(x, qw):
  return MatmulQ(jax.typeof(x), jax.typeof(qw))(x, qw)
```

A few things to notice. The type signature mixes array types and hi types
freely: `in_avals` is a `(ShapedArray, QArrayTy)` pair, checked at
construction time so that a shape mismatch fails immediately, with both
operand types pretty-printed. And `expand` exploits the representation:
because the scales apply per-row along the contraction axis, they can be
folded into the dense operand, so the heavy matmul runs directly against
the `int8` payload rather than a dequantized copy. Owning the op as a
single primitive lets us state that rewriting once, in one place.

Also notice the discipline from the previous section in action: `expand`
reads `qw.scale` and `qw.qvalue` as container attributes, while the VJP
rules — which are ordinary traced code — go through the `dequantize`
primitive instead.

(The typing rule also computes an output *sharding* — output rows
partitioned like `x`'s, output columns like `qw`'s, with the contracted
axes required to be unsharded — and the backward rule passes
`out_sharding` hints to its matmuls. Both are explained in the
explicit-sharding section below; outside of explicit mode all these
shardings are trivial and the extra code is inert.)

```{code-cell}
w = jnp.arange(12., dtype='float32').reshape(3, 4) / 12.
qw = quantize(w)

print(matmul_q(x, qw))
print(x @ dequantize(qw))  # reference
```

## Hi types in jaxprs

When we trace, the quantized array appears as a single value of type
`q8[2,3]`, produced by one equation and consumed by another:

```{code-cell}
jax.jit(lambda x: dequantize(quantize(x))).trace(x).jaxpr
```

Compare to the pytree approach, where the same computation would show four
array-typed intermediates with no indication that they pair up. The hi type
only disappears at lowering time, when `expand` is traced and each
`q8[...]`-typed value is expanded into the array components given by
`lo_ty`.

Ops with mixed operand kinds read just as directly — one equation with a
`f32[2,3] @ q8[3,4] -> f32[2,4]` signature:

```{code-cell}
jax.jit(matmul_q).trace(x, qw).jaxpr
#
# `jit` works, with quantized arrays as arguments, results, and
# intermediates:
```

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

The same goes for ops with mixed operand kinds, like the quantized
matmul: differentiating a loss with respect to both operands gives an
`f32` gradient for the dense activations *and* an `f32` gradient for the
quantized weights:

```{code-cell}
def loss(x, qw):
  return jnp.sum(matmul_q(x, qw) ** 2)

grad_x, grad_qw = jax.grad(loss, argnums=(0, 1))(x, qw)
print(jax.typeof(grad_x), jax.typeof(grad_qw))
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
a spec carrying one axis per component; see the tuple example at the end
of this document.)

On the type, we implement `dec_rank` and `inc_rank`, the hi type analogues
of "remove the mapped axis" and "add the mapped axis." They take the axis
size and a spec, and return the element type and the batched type,
respectively:

```{code-cell}
def qarray_dec_rank(self, size, spec):
  assert isinstance(spec, QArraySpec) and self.shape[0] == size
  return QArrayTy(self.shape[1:],
                  self.sharding.update(spec=jax.P(*self.sharding.spec[1:])))

def qarray_inc_rank(self, size, spec):
  assert isinstance(spec, QArraySpec)
  return QArrayTy((size, *self.shape),
                  self.sharding.update(spec=jax.P(None, *self.sharding.spec)))

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

## `scan` and the leading axis

`jax.lax.scan` can loop over a stacked hi value, consuming one slice per
step — and it can carry and produce hi values too. Where `vmap` asks the
*user* for a mapping spec, `scan` always walks the leading axis, so it
instead asks the *type*: the one extra method to implement is
`leading_axis_spec`, which returns the mapping spec describing your type's
leading axis. The `dec_rank` and `inc_rank` methods from the `vmap`
section do the rest of the work.

```{code-cell}
def qarray_leading_axis_spec(self):
  return QArraySpec()

QArrayTy.leading_axis_spec = qarray_leading_axis_spec
```

Scanning over the stack of quantized arrays from the previous section, the
body sees one `q8[2,3]` per step. As with `vmap`'s `axis_size`, when all
the scanned-over values are hi types there's no leading-axis size to
infer, so we pass `length` explicitly (if any scanned-over value is an
array, `scan` infers the length from it and `length` can be omitted):

```{code-cell}
def sum_dequantized(total, qx):
  return total + jnp.sum(dequantize(qx)), ()

total, () = jax.lax.scan(sum_dequantized, 0., qxs, length=4)
print(total)
```

Hi values also work as stacked outputs and as the loop carry. Here the
carry is a quantized array, re-quantized after each accumulation step, and
the second output stacks one fresh `QArray` per step into a `q8[4,2,3]`:

```{code-cell}
def accum_quantized(qtotal, x):
  return quantize(dequantize(qtotal) + x), quantize(2 * x)

qzero = quantize(jnp.zeros((2, 3), 'float32'))
qtotal, qys = jax.lax.scan(accum_quantized, qzero, xs)
print(jax.typeof(qtotal))
print(jax.typeof(qys))
```

## Sharding in types: explicit mode

Finally, sharding. In JAX's explicit sharding mode (see [the parallelism
guide](https://docs.jax.dev/en/latest/parallel.html)), shardings are part
of array *types*: `jax.typeof` reports how a value is partitioned across
the mesh, sharding propagation happens while tracing, and mismatches
surface as type errors. The `sharding` field on `QArrayTy`, and the typing
rules on our primitives that propagate it, are exactly what let hi types
participate. What does it mean to partition a quantized array across
devices? The components move together: if we shard the rows, each device
holds its rows' quantized values *and* their scales. As usual there's a
design choice to make, and we made it back in `lo_ty`: `qvalue` carries
the type's sharding, and `scale` shards like it with the last axis
dropped, so every row travels with its scale.

Let's see it work. We make a mesh (this is where we use the 8 CPU devices
requested in this document's first cell), shard some rows across it, and
quantize — the shardings propagate through our typing rules into the
result type, which `jax.typeof` displays with `@` markers:

```{code-cell}
mesh = jax.make_mesh((4,), ('i',))
jax.set_mesh(mesh)

rows = jax.device_put(jnp.arange(24., dtype='float32').reshape(8, 3),
                      jax.P('i'))

qrows = quantize(rows)
print(jax.typeof(qrows))
print(qrows.qvalue.sharding.spec, qrows.scale.sharding.spec)
print(jax.typeof(dequantize(qrows)))
```

The same holds while tracing — hi types in jaxprs now display their
shardings, computed by our `out_aval` rules:

```{code-cell}
jax.jit(lambda x: dequantize(quantize(x))).trace(rows).jaxpr
```

The quantized matmul propagates shardings too: its typing rule partitions
output rows like `x`'s rows and output columns like `qw`'s columns
(requiring the contracted axes to be unsharded, since neither operand can
say how a sharded contraction should land):

```{code-cell}
w2 = jnp.arange(12., dtype='float32').reshape(3, 4) / 12.

print(jax.typeof(matmul_q(rows, quantize(w2))))                # rows sharded

qw2 = quantize(jax.device_put(w2, jax.P(None, 'i')))
print(jax.typeof(qw2))                                         # cols sharded
print(jax.typeof(matmul_q(jnp.ones((2, 3), 'float32'), qw2)))
```

And the contraction check fires as a type error, at trace time:

```{code-cell}
xk = jax.device_put(jnp.ones((2, 8), 'float32'), jax.P(None, 'i'))
qk = quantize(jax.device_put(jnp.ones((8, 3), 'float32'), jax.P('i')))

try:
  matmul_q(xk, qk)
except TypeError as e:
  print('TypeError:', e)
```

Autodiff composes with all of this. Recall that `MatmulQ`'s backward rule
passed `out_sharding` hints to its matmuls: that's because the cotangent
for `qw` contracts over the row axis, which may be sharded (as it is
here), and explicit mode refuses to guess how an all-sharded contraction
should land. The right answer is the primal operand's sharding —
cotangents live where their primals live:

```{code-cell}
def qloss(x, qw):
  return jnp.sum(matmul_q(x, qw) ** 2)

grad_rows, grad_qw2 = jax.grad(qloss, argnums=(0, 1))(rows, quantize(w2))
print(jax.typeof(grad_rows), jax.typeof(grad_qw2))
```

One caution: JAX does not cross-check a hi primitive's declared output
sharding against what its `expand` actually produces. The declared
`out_aval` is what downstream code sees while tracing; `expand` determines
what happens at runtime. Keeping them consistent is part of the typing
rule's job.

## `shard_map` and partition specs

Explicit mode partitions values while keeping one global view of the
program. Its complement is `shard_map`, which gives a per-device view. To
cross that boundary, a quantized array needs one more piece: a partition
spec type of our own, saying how `shard_map`'s `in_specs`/`out_specs`
apply to it. We keep the same design as above: a quantized array is
sharded along its leading axes only.

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
  shard_ty = qvalue_ty.shard(mesh, manual_axes, check_vma, qspec)
  return QArrayTy(shard_ty.shape, shard_ty.sharding)

def qarray_unshard(self, mesh, check_vma, spec):
  qvalue_ty, _ = self.lo_ty()
  qspec, _ = spec.to_lo()
  full_ty = qvalue_ty.unshard(mesh, check_vma, qspec)
  return QArrayTy(full_ty.shape, full_ty.sharding)

QArrayTy.shard = qarray_shard
QArrayTy.unshard = qarray_unshard
```

Now quantized arrays can cross `shard_map` boundaries in either
direction, continuing with the mesh and `rows` from the previous section.
Producing a sharded quantized array:

```{code-cell}
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
  assert jax.typeof(qx).shape == (2, 3)  # a per-device QArray shard
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

## Additional examples

The recipe is always the same — a value class, a `HiType` with the
`lo_ty`/`lower_val`/`raise_val` lowering triple, and primitives whose type
signatures mention the new type — so further examples can be read quickly.

### Rank-1 arrays

A rank-1 array represents an `m × n` matrix as an outer product of two
vectors, `col : f32[m]` and `row : f32[n]`. The point of the
representation is to never materialize the `m × n` product: ops consume
the factors directly. (A general low-rank type, with `f32[m, r]` and
`f32[r, n]` factors, is more of the same.)

```{code-cell}
@dataclass(frozen=True)
class Rank1:
  col: jax.Array   # f32[m]
  row: jax.Array   # f32[n]

@dataclass(frozen=True)
class Rank1Ty(HiType):
  shape: tuple[int, int]   # (m, n), the dense shape represented
  sharding: NamedSharding  # sharding of the dense shape; the factors' derive

  def lo_ty(self):
    (m, n), spec = self.shape, self.sharding.spec
    return [ShapedArray((m,), jnp.dtype('float32'),
                        sharding=self.sharding.update(spec=jax.P(spec[0]))),
            ShapedArray((n,), jnp.dtype('float32'),
                        sharding=self.sharding.update(spec=jax.P(spec[1])))]
  def lower_val(self, r1):
    return [r1.col, r1.row]
  def raise_val(self, col, row):
    return Rank1(col, row)

  # rank-1 matrices aren't closed under addition, so tangents are dense
  def to_tangent_aval(self):
    return ShapedArray(self.shape, jnp.dtype('float32'),
                       sharding=self.sharding)

  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    dims = [str(d) if p is None else f'{d}@{p}'
            for d, p in zip(self.shape, self.sharding.spec)]
    return f'r1[{",".join(dims)}]'
  __repr__ = str_short

def typeof_rank1(r1):
  col_s, row_s = jax.typeof(r1.col).sharding, jax.typeof(r1.row).sharding
  sharding = col_s.update(spec=jax.P(col_s.spec[0], row_s.spec[0]))
  return Rank1Ty((r1.col.shape[0], r1.row.shape[0]), sharding)

register_hitype(Rank1, typeof_rank1)
```

The type records the dense shape it represents and prints as e.g.
`r1[6,5]`. Note the tangent-type choice, which differs from `QArrayTy`'s
for a different reason: the sum of two rank-1 matrices is in general
rank 2, so rank-1 matrices aren't closed under addition and can't serve
as their own tangent space. Perturbations leave the manifold, and
tangents and cotangents are dense. The sharding story is the same as
before, with the field consumed in `lo_ty`: the dense row axis is carried
by `col` and the dense column axis by `row`.

Unlike `QArray`, whose values were only ever created inside `quantize`,
here construction from factors is itself a primitive — and there's an
accessor primitive too, so that *rules* (which are ordinary traced code)
can get at the factors without touching container attributes:

```{code-cell}
class Outer(VJPHiPrimitive):
  def __init__(self, col_aval, row_aval):
    if not (len(col_aval.shape) == 1 and len(row_aval.shape) == 1):
      raise TypeError(f'bad outer factor types: {col_aval}, {row_aval}')
    sharding = col_aval.sharding.update(
        spec=jax.P(col_aval.sharding.spec[0], row_aval.sharding.spec[0]))
    self.in_avals = (col_aval, row_aval)
    self.out_aval = Rank1Ty((col_aval.shape[0], row_aval.shape[0]), sharding)
    self.params = {}
    super().__init__()

  def expand(self, col, row):
    return Rank1(col, row)

  def vjp_fwd(self, nzs_in, col, row):
    return self(col, row), (col, row)

  def vjp_bwd_retval(self, res, g):   # g is dense, f32[m, n]
    col, row = res
    return g @ row, col @ g

class Factors(VJPHiPrimitive):
  def __init__(self, r1_aval):
    self.in_avals = (r1_aval,)
    self.out_aval = tuple(r1_aval.lo_ty())  # the factor types are the lo types
    self.params = {}
    super().__init__()

  def expand(self, r1):
    return (r1.col, r1.row)

class MatmulR1(VJPHiPrimitive):
  def __init__(self, x_aval, r1_aval):
    if not (isinstance(r1_aval, Rank1Ty) and len(x_aval.shape) == 2 and
            x_aval.shape[1] == r1_aval.shape[0]):
      raise TypeError(f'bad matmul_r1 operand types: {x_aval} @ {r1_aval}')
    self.in_avals = (x_aval, r1_aval)
    out_sharding = x_aval.sharding.update(
        spec=jax.P(x_aval.sharding.spec[0], r1_aval.sharding.spec[1]))
    self.out_aval = ShapedArray((x_aval.shape[0], r1_aval.shape[1]),
                                jnp.dtype('float32'), sharding=out_sharding)
    self.params = {}
    super().__init__()

  def expand(self, x, r1):
    return jnp.outer(x @ r1.col, r1.row)   # never materialize col ⊗ row

  def vjp_fwd(self, nzs_in, x, r1):
    col, row = factors(r1)  # rules are traced code: primitives, not attributes
    return self(x, r1), (x, col, row)

  def vjp_bwd_retval(self, res, g):
    x, col, row = res
    return jnp.outer(g @ row, col), x.T @ g

def outer(col, row):
  return Outer(jax.typeof(col), jax.typeof(row))(col, row)

def factors(r1):
  return Factors(jax.typeof(r1))(r1)

def matmul_r1(x, r1):
  return MatmulR1(jax.typeof(x), jax.typeof(r1))(x, r1)
```

(Note `Factors` declares its output type as a *tuple* of types —
`in_avals` entries and `out_aval` can be pytrees of types — and that it
carries no autodiff rules at all, since we only apply it in forward
passes; rules are only needed for the transformations you actually use.
Note also `MatmulR1`'s backward rule computes the dense operand's
cotangent as an outer product, exploiting the representation the same way
`expand` does.)

```{code-cell}
col = jnp.arange(6., dtype='float32') / 6.
row = jnp.arange(5., dtype='float32') / 5.
r1 = outer(col, row)
print(jax.typeof(r1))

acts = jnp.ones((3, 6), 'float32')
print(jnp.max(jnp.abs(matmul_r1(acts, r1) - acts @ jnp.outer(col, row))))
```

Jaxprs, `jit`, and autodiff all work as before. The gradient with respect
to a rank-1 operand is dense, as the tangent type dictates, while
gradients chained through the constructor land back on the factors:

```{code-cell}
jax.jit(matmul_r1).trace(acts, r1).jaxpr
```

```{code-cell}
def r1_loss(x, r1):
  return jnp.sum(matmul_r1(x, r1) ** 2)

print(jax.typeof(jax.grad(r1_loss, argnums=1)(acts, r1)))

def factor_loss(col, row):
  return jnp.sum(matmul_r1(acts, outer(col, row)) ** 2)

g_col, g_row = jax.grad(factor_loss, argnums=(0, 1))(col, row)
print(jax.typeof(g_col), jax.typeof(g_row))
```

And the typing rules propagate shardings, just like `QArrayTy`'s:

```{code-cell}
print(jax.typeof(outer(jax.device_put(jnp.zeros(8, 'float32'), jax.P('i')),
                       row)))
```

We stopped here, but `vmap`, `scan`, and `shard_map` support would follow
the same recipes as in the sections above: `dec_rank`/`inc_rank` and a
mapping spec, `leading_axis_spec`, and `shard`/`unshard` with an `HiPspec`
partition spec type.

### Tuples

Our last example is a generic container: a tuple whose elements are any
JAX values. Where `QArrayTy` and `Rank1Ty` had a fixed component
structure, `TupTy` is parameterized by its component *types* — and every
method delegates to them:

```{code-cell}
import itertools

@dataclass(frozen=True)
class HiTup:
  elts: tuple

@dataclass(frozen=True)
class TupTy(HiType):
  tys: tuple  # component types: array types, or other hi types

  # lowering delegates to the component types
  def lo_ty(self):
    return [lo for ty in self.tys for lo in ty.lo_ty()]
  def lower_val(self, tup):
    return [lo for ty, elt in zip(self.tys, tup.elts)
            for lo in ty.lower_val(elt)]
  def raise_val(self, *los):
    los = iter(los)
    return HiTup(tuple(ty.raise_val(*itertools.islice(los, len(ty.lo_ty())))
                       for ty in self.tys))

  # so does the tangent type
  def to_tangent_aval(self):
    return TupTy(tuple(ty.to_tangent_aval() for ty in self.tys))

  def str_short(self, short_dtypes=False, mesh_axis_types=False):
    return ('Tup{' +
            ','.join(t.str_short(short_dtypes, mesh_axis_types)
                     for t in self.tys) + '}')
  __repr__ = str_short

register_hitype(HiTup, lambda t: TupTy(tuple(map(jax.typeof, t.elts))))
```

Two things fall out of the delegation for free. First, elements can
themselves be hi types — tuples of tuples, or a tuple holding a `QArray` —
since `lo_ty` and friends just recurse. Second, there's no `sharding`
field this time: the stored component types carry their own shardings, a
third way of handling sharding-in-types alongside `QArrayTy`'s single
field and `Rank1Ty`'s dense-shape spec.

The constructor and accessor primitives are familiar from the rank-1
example. One new ingredient: the element index is a *static parameter*,
passed via `params` and available as `self.idx`. And since each element
can be batched along its own axis (or not at all), the mapping spec
carries one entry per component:

```{code-cell}
@dataclass(frozen=True)
class TupSpec(MappingSpec):
  val: tuple  # one axis entry per component

def tup_dec_rank(self, size, spec):
  return TupTy(tuple(ty.dec_rank(size, s)
                     for ty, s in zip(self.tys, spec.val)))

def tup_inc_rank(self, size, spec):
  return TupTy(tuple(ty.inc_rank(size, s)
                     for ty, s in zip(self.tys, spec.val)))

TupTy.dec_rank = tup_dec_rank
TupTy.inc_rank = tup_inc_rank

class MakeTup(VJPHiPrimitive):
  def __init__(self, elt_avals):
    self.in_avals = tuple(elt_avals)
    self.out_aval = TupTy(tuple(elt_avals))
    self.params = {}
    super().__init__()

  def expand(self, *elts):
    return HiTup(elts)

  def batch(self, axis_data, args, in_dims):
    return make_tup(*args), TupSpec(tuple(in_dims))

class GetTupElt(VJPHiPrimitive):
  def __init__(self, tup_aval, idx):
    self.in_avals = (tup_aval,)
    self.out_aval = tup_aval.tys[idx]
    self.params = dict(idx=idx)
    super().__init__()

  def expand(self, tup):
    return tup.elts[self.idx]

  def batch(self, axis_data, args, in_dims):
    tup, = args
    spec, = in_dims
    if spec is None:
      return get_tuple_element(tup, self.idx), None
    return get_tuple_element(tup, self.idx), spec.val[self.idx]

def make_tup(*elts):
  return MakeTup(map(jax.typeof, elts))(*elts)

def get_tuple_element(tup, idx):
  return GetTupElt(jax.typeof(tup), idx)(tup)
```

(Note `GetTupElt`'s batch rule handles unbatched inputs — hi primitive
batch rules are invoked even when no argument is batched, so `in_dims`
entries can be `None`.)

Tuples work, they nest, and they hold other hi types:

```{code-cell}
tup = make_tup(jnp.arange(3.), 5.)
print(jax.typeof(tup))
print(get_tuple_element(tup, 1))

nested = make_tup(make_tup(1., 2.), jnp.arange(2.))
print(jax.typeof(nested))
print(jax.jit(lambda t: get_tuple_element(get_tuple_element(t, 0), 1))(nested))

print(jax.typeof(make_tup(qx, 3.)))  # a quantized array element
```

The per-component mapping spec is the payoff under `vmap`: each element
gets its own `in_axes`/`out_axes` entry, visible in the types. Here the
first element is mapped on the way in, and only the second on the way
out:

```{code-cell}
def swap(t):
  a, b = get_tuple_element(t, 0), get_tuple_element(t, 1)
  return make_tup(b, a)

out = jax.vmap(swap, in_axes=TupSpec((0, None)), out_axes=TupSpec((None, 0)),
               axis_size=3)(tup)
print(jax.typeof(tup), '->', jax.typeof(out))
```

And since the component types carry their own shardings, sharding-in-types
needs nothing extra at all:

```{code-cell}
print(jax.typeof(make_tup(rows, jnp.float32(1.))))
```

For autodiff rules on these primitives, and `scan` and `shard_map`
support (via a per-component `HiPspec`), see the `TupTy` example in
`tests/hijax_test.py`.

## What we haven't covered

On the primitive side, there are also hooks for customizing
rematerialization and dead code elimination.

As ever with hijax, `tests/hijax_test.py` is a good source of worked
examples, and {ref}`hijax-custom-derivatives` covers the primitive-side
API — including JVP rules, symbolic zeros, and custom linearization — in
more depth.
