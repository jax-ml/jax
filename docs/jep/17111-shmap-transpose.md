# Efficient transposition of replication-inducing collectives
*mattjj@*, *dougalm@*

*August 2023*

## Motivation

We have an efficiency problem in automatically transposing `shmap`s containing
certain collectives. The issue arises with `psum` and `all_gather`, specifically
when the output of the collective is returned to the caller as an unmapped
output. And it's not an edge case: for example, it arises when applying `grad`
to a `shmap`-based batch data parallel neural network loss function which uses
`psum` to compute the total loss.

We've known about this problem for some time. An analogous issue exists with
`pmap`, though it's been worked around by keeping `grad` inside `pmap` rather than
outside. A primary goal of the incomplete avals-with-names work was to address a
version of this transpose efficiency problem. This doc draws on those ideas,
while extending and revising them to handle more cases and to be much easier to
land. Indeed the solution proposed here only affects the `shmap` implementation.
The rest of the system need not be changed (yet).

The main purpose of this doc is to define this transpose efficiency problem and
propose an easy-to-land solution.

This doc is not about:
* logical axis names on arrays (the only axis names here are just like in
  `shmap` and OG `pmap`);
* changing autodiff semantics (all the numbers and (non)errors are staying the
  same, we're just making things more efficient);
* allowing user code to reflect on any new information, or really affecting user
  code at all.

## Problem: efficient transpose of `psum` or `all_gather` depends on whether cotangents are invariant across devices

Consider this semi-realistic example, meant to resemble a replicated-parameter
batch data parallel loss function:

```python
devices = jax.devices()  # 8 devices

@partial(shmap, mesh=Mesh(devices, ('batch',)),
         in_specs=(P(None, None), P('batch', None)),
         out_specs=P())
def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  local_loss = jnp.mean(jnp.sum(predictions - targets, -1))
  global_loss = lax.pmean(local_loss, 'batch'))
  return global_loss
```

Notice the `out_specs=P()`, which indicates an unmapped output. If you're not
familiar with the notion of unmapped outputs, see the appendix at the bottom of
this document.

Most of the details in the `loss` example aren't important. All that matters for
our purposes is that we're applying `psum` (or rather `pmean = lambda x, name:
psum(x, name) / psum(1, name)`) at the end. So a distilled version looks like
this:

```python
# Example 1: shmap involving psum and unmapped output with inefficient transpose
f1 = shmap(lambda x: psum(g(x), 'i'),
           in_specs=P('i'), out_specs=P())
```

We even simplified notation by suppressing the `mesh` argument. In the examples to
follow it can be inferred from context.

What does the transpose look like? Writing `t` to mean function transpose, we
could evaluate `t(f1)(ybar)` for any `ybar` efficiently by applying the function
`¿f1_transpose?` below:

```python
# An efficient "transpose" of Example 1 (but don't transpose this again!)
¿f1_transpose? = shmap(t(g), in_specs=P(), out_specs=P('i'))
```

But that's not the transpose we currently get as t(f1).

Instead, the current recipe for transposition is roughly that we switch
`in_specs` and `out_specs`, do some division rescaling for unmapped outputs, and
transpose the body. Because `psum` is its own transpose (as an all-reduce sum),
we end up producing this transpose:

```python
# The transpose we currently get for Example 1 (which is fine to transpose again)
t(f1) = shmap(lambda ybar: t(g)(psum(ybar / 8, 'i')),
              in_specs=P(), out_specs=P('i'))
```

This transpose gets the numbers right, but it's wasteful. We know statically
from the transpose's `in_specs=P()` that `ybar` has the same value for each function
instance, i.e. that its value is device-invariant for devices along the mesh
axis named `i`, and yet we apply a `psum` to it! That uses expensive communication
just to multiply the value on each device by 8. (Here 8 refers to the size of
axis i. The division by 8 comes from the original function's `out_specs=P()`; it
and the trivial `psum` basically cancel each other out.)

What are we doing wrong? We're not exploiting the fact that cotangents `ybar`
corresponding to `f1`'s unmapped outputs are guaranteed to be device-invariant;
instead, we're defensively `psum`ming them as if they weren't because `psum`'s
transpose can't be sure given the local information it has. Sometimes the `psum`
is necessary, as in transposing `f2` with respect to its first argument:

```python
# Example 2: shmap involving psum and *mapped* output with efficient transpose
f2 = shmap(lambda x, y: psum(g(x), 'i') * y,
          in_specs=(P('i'), P('i')), out_specs=P('i'))

# The transpose we currently get for Example 2 is efficient
t(f2, 0) = shmap(lambda y, zbar: t(g)(psum(zbar * y, 'i')),
                in_specs=(P('i'), P('i')), out_specs=P('i'))
```

Intuitively, if our transpose machinery could tell the difference between
Example 1 and Example 2, we could do better by avoiding the psum and division
where possible.

The inefficient examples can be even smaller. Consider transposing this cursed
identity function:

```python
# Example 3: cursed identity
cursed_identity = shmap(lambda x: x, P(), P())

# Currently we get these inefficient transposes
t(cursed_identity) = shmap(lambda x: psum(x / 8, 'i'), P(), P())
t(t(cursed_identity)) = shmap(lambda x: psum(psum(x / 8 / 8, 'i'), 'i')), P(), P())
...
```

It keeps getting bigger the more we transpose. How embarrassing!

And `psum` isn't the only culprit. Something analogous holds true for
`all_gather`:

```python
# Example 4: all_gather to an unmapped output
f4 = shmap(lambda x: all_gather(x, 'i'), P('i'), P())

# Currently we get this inefficient transpose
t(f4) = shmap(lambda ybar: psum_scatter(ybar / 8, 'i'), P(), P('i'))
```

This program is a bit artificial. Why do an `all_gather` and feed the result into
an unmapped output, rather than skipping the `all_gather` in the body and just
using `out_specs=P('i')` to collect the results? But even though it's cooked-up,
this example nevertheless exhibits a transpose which unnecessarily performs
communication (we could have just performed a non-communicating slice),
analogous to Example 1 for `psum`.

Also analogously to the `psum` examples, the defensive `psum_scatter` is
necessary in some cases:

```python
# Example 5: all_gather to a mapped output
f5 = shmap(lambda x, y: all_gather(x, 'i') * y,
           in_specs=(P('i'), P('i')), out_specs=P('i'))

# Currently we get this efficient transpose
t(f5, 0) = shmap(lambda y, zbar: psum_scatter(zbar * y, 'i'),
                 in_specs=(P('i'), P('i')), out_specs=P('i'))
```

So how do we avoid these inefficient transposes?

## Solutions

Here are two solution ideas. They aren't mutually exclusive. But (spoilers) the
second one is better, and it's all we need.

### Partial solution "P-sum": build the ability to express a `psum` into `out_specs`

This solution is a bit of a strawperson because it would offer only an awkward
way to write programs. And it wouldn't even fix everything! But it's worth
considering, if only to motivate a more complete solution.

Example 4 above is artificial because we could have just used `out_specs` instead
of an `all_gather` in the body:

```python
# Example 4 again
f4 = shmap(lambda x: all_gather(x, 'i'), P('i'), P())

# Why didn't we just write it like this?
f4_better = shmap(lambda x: x, P('i'), P('i'))
```

The `f4_better` version doesn't have any transposition problems, since the
transpose problems arise from collectives in the body.

Analogously, we could fix Example 1 by extending `out_specs` so that they can
express summing:

```python
# Example 1 again
f1 = shmap(lambda x: psum(g(x), 'i'),
           in_specs=P('i'), out_specs=P())

# What if we could write an output sum like this?
f1_better = shmap(g, in_specs=P('i'), out_specs=P(sum='i'))  # sum='i' means sum over that axis

# Then it could transpose like this:
t(f1_better) = shmap(t(g), in_specs=P(), out_specs=P('i'))
t(t(f1_better)) = shmap(t(t(g)), in_specs=P('i'), P(sum='i'))
```

So offering `psum`s built into `out_specs` fixes the transpose problem of
Example 1. But it doesn't fully fix the cursed identity transpose in Example 3:

```python
# Example 3 again
cursed_identity = shmap(lambda x: x, P(), P())

# How it would transpose with the P-sum partial solution:
t(cursed_identity) = shmap(lambda x: x / 8, P(), P(sum='i'))
t(t(cursed_identity)) = shmap(lambda x: x / 8, P(), P(sum='i'))
```

It's an improvement since the program doesn't continue to get bigger as we keep
transposing, but we're still doing wasteful communication.

### Full solution: statically track device-varying vs device-invariant intermediates, plus new primitives

This solution has two components:
1. track when values are guaranteed to be device-invariant vs device-varying
   over particular mesh axes, and
2. decompose `psum` into a two-step process, introducing a new `pbroadcast`
   primitive, and introduce new primitives for `all_gather` and its transposes.

Morally, the tracking of device-invariant vs device-varying information is a
type-level consideration. But for the expedience of our first implementation, we
don't need to literally add the information to abstract values or jaxpr types.
Before we get to implementation, we'll first introduce the idea using types.

Also to follow is a discussion of making the user API convenient and backward
compatible. But to first introduce the idea, we'll ignore convenience and
instead write code that is as explicit as possible.

#### Tracking device invariance in avals (a.k.a. avals-with-names, revived)

We can sometimes tell from static information alone that the values of some
intermediate variables in the body of a `shmap` are guaranteed to be invariant
along a mesh axis, in the sense that the function instances (and their
corresponding devices) along the mesh axis must all be computing with the same
value. We'll call such values device-invariant. For values that are not
device-invariant, we'll say they're device-varying, though really we mean
potentially device-varying from the point of view of the type system.

To encode device variance in types, we'll extend the syntax of types for arrays.
We'll write things like `x:f32[3,4]{i}` to indicate that `x` is (potentially)
device-varying along mesh axis `i` (and device-invariant over any other mesh
axes of the `shmap`). More generally, we'll say the grammar for array type
syntax is something like

```
shaped_array ::= <dtype>[<int_literal>, ...]<device_variance_type>
device_variance_type ::= {<axis_name>, ...}
```

We'll also update the typing rules to handle device variance types:
* for first-order primitives other than collectives
  - for multi-arity primitives, the operand device variance types must be equal
    where shapes must be equal, e.g. `mul x:f32[s1]{r1} y:f32[s2][r2]` requires
    `r1 == r2` in addition to `s1 == s2`
  - the output device variance type must be the same as the operand(s)
* for higher-order primitives
  - we just instantiate any type variables including the device variance type
    (and checking types for equality checks their device variance types are
    equal)
  - (when performing type inference, e.g. for branches of a `cond`, we take the
    union of the sets of axis names in device variance types)
* for first-order collectives
  - a collective can either accept a device-varying or device-invariant input
    (along a mesh axis corresponding to its axis name parameter); it's an error
    to pass a device-invariant operand to a collective which accepts
    device-varying operands and vice-versa
  - a collective can either produce a device-varying or device-invariant output
  - see the table below
As a side benefit, whatever logic implements this type checking can subsume
`shmap`'s "static analysis" check for whether a `shmap` body function is
compatible with any unmapped `out_specs`.

Here's a table summarizing the device variance typing for collective primitives:

| Name | Device variance type | Example | Lowers to HLO | Transpose |
| ---  |         ---          |   ---   |     ---       |    ---    |
| `psum2` | `Varying -> Invariant` | `y:f32[3]{j} = psum(x:f32[3]{i,j}, axis='i')` | `AllReduceSum` (communication) | `pbroadcast` |
| `pbroadcast` | `Invariant -> Varying` | `y:f32[3]{i} = pbroadcast(x:f32[3], 'i')` | no-op (no communication) | `psum` |
| `all_to_all` | `Varying -> Varying` | `y:f32[16]{i} = all_to_all(x:f32[16]{i}, 'i', 0, 0)` `AllToAll` (communication) | `all_to_all` |
| `axis_index` | `() -> Varying` | `idx:i32[]{i} = axis_index('i')` | `ReplicaId` and some arithmetic (no communication) | n/a |
| `psum_scatter` | `Varying -> Varying` | `y:f32[2]{i} = psum_scatter(x:f32[16]{i}, 'i')` | `ReduceScatterSum` (communication) | `all_gather` |
| `all_gather` | `Varying -> Varying` | `y:f32[16]{i} = all_gather(x:f32[2]{i}, 'i')` | `AllGather` (communication) | `psum_scatter` |
| `pscatter` | `Invariant -> Varying` | `y:f32[2]{i} = pscatter(x:f32[16], 'i')` | `lambda x: x[axis_index('i'), None]` (no communication) | `all_gather_invariant` |
| `all_gather_invariant` | `Varying -> Invariant` | `y:f32[16] = all_gather_invariant(x:f32[2]{i}, 'i')` | `AllGather` (communication) | `pscatter` |


There are some surprising things here!
* We introduced several new primitives, including
  - `pbroadcast`, which interestingly lowers to a no-op
  - `all_gather_invariant`, which lowers to the same thing as `all_gather` but
    has a different device variance type (essentially `all_gather` has a
    `pbroadcast` fused into it, whereas `all_gather_invariant` does not)
  - `pscatter` which is the dual (transpose) of `all_gather_invariant`
* all_gather has a device-varying result

Intuitively, the reason to introduce `pbroadcast` (other than to make the typing
rules work) is so that `psum` can transpose to a physical no-op. The reason we
need `all_gather` to have a device-varying result is so that we can transpose it
to `psum_scatter`; if we instead left it with a device-invariant result, we
might need a downstream `pbroadcast`, and that composition would transpose to an
inefficient `psum` followed by slicing / `pscatter`. So instead we have a
`pbroadcast` "fused into" the `all_gather`, thus allowing for an efficient
transpose into `psum_scatter`. We provide `all_gather_invariant` and its
transpose `pscatter` mainly for completeness; it's unlikely users will need it
(it corresponds to the situation in Example 4, which is easy to write
differently using `out_specs`).

Interestingly, the `psum` and `pbroadcast` transpose pair correspond to the
`psum_idrev` and `id_psumrev` that users introduced while training LLMs with
`pmap`.

#### How this system solves the inefficient transpose examples

Consider again the simplified motivating example:

```python
# Example 1 again
f1 = shmap(lambda x: psum(g(x), 'i'),
           in_specs=P('i'), out_specs=P())

# Example 1 with intermediate device variance types annotated
@partial(shmap, in_specs=P('i'), out_specs=P())
def f1(x: f32[3,4]{i}):
  w:f32[]{i} = g(x)
  y:f32[]{} = psum(w, 'i')
  return y
```

With these new rules, the transpose is:

```python
# Example 1 transpose using device variance types (go ahead and transpose this again!)
t(f1) = shmap(lambda ybar: t(g)(pbroadcast(ybar, 'i')),
              in_specs=P(), out_specs=P('i'))

# Example 1 transpose with intermediate device variance types annotated
@partial(shmap, in_specs=P('i'), out_specs=P())
def f1_transpose(ybar: f32[]):
  wbar:f32[]{i} = pbroadcast(ybar, 'i')
  xbar:f32[3,4]{i} = transpose(g)(wbar)
  return xbar
```

where evaluating the `pbroadcast` application involves no communication or FLOPs
at all; it's a no-op. Notice that if we keep transposing the body does not grow
in size; indeed `t(t(f1)) == f1`. Efficiency achieved!

And we wouldn't mess up the other examples either, so long as we `pbroadcast` to
make the types check where needed:

```python
# Example 2 rewritten with explicit pbroadcast
f2 = shmap(lambda x, y: pbroadcast(psum(g(x), 'i'), 'i') * y,
           in_specs=(P('i'), P('i')), out_specs=P('i'))

# Example 2 transpose using device variance types
t(f2, 0) = shmap(lambda y, zbar: t(g)(pbroadcast(psum(zbar * y, 'i'), 'i')),
                 in_specs=(P('i'), P('i')), out_specs=P('i'))


# Example 3 again
cursed_identity = shmap(lambda x: x, P(), P())
# Notice here the body is `f32[...] -> f32[...]`, i.e. no device varying type.

# Example 3 transpose using device variance types
t(cursed_identity) = shmap(lambda x: x, P(), P())
t(t(cursed_identity)) = shmap(lambda x: x, P(), P())
```

Intuitively, in Example 1 we now only have "half the original psum", whereas in
Example 2 we get both "halves". For Example 3 we never need any operations in
the body at all.

For the `all_gather` examples, Example 4 would need to use
`all_reduce_invariant` to have an efficient transpose (though it'd be better to
instead use `out_specs` instead of the collective in the body):

```python
# Example 4 rewritten with explicit all_reduce_invariant
f4 = shmap(lambda x: all_gather_invariant(x, 'i'), P('i'), P())

# Example 4 with intermediate device variance types annotated
@partial(shmap, P('i'), P())
def f4(x:f32[1]{i}):
  y:f32[8]{} = all_gather_invariant(x, 'i')
  return y

# Example 4 transpose with intermediate device variance types annotated
@partial(shmap, in_specs=P(), out_specs=P('i'))
def f4_transpose(ybar:f32[8]):
  xbar:f32[1]{i} = pscatter(ybar, 'i')
  return xbar
```

For Example 5, using the device-varying `all_gather` works as we'd want:

```python
# Example 5 with intermediate device variance types annotated
@partial(shmap, in_specs=(P('i'), P('i')), out_specs=P('i'))
def f5(x:f32[1]{i}, y:f32[8]{i}):
  z:f32[8]{i} = all_gather(x, 'i')
  w:f32[8]{i} = z * y
  return w

# Transpose with respect to first argument
@partial(shmap, in_specs=(P('i'), P('i')), out_specs=P('i'))
def f5_transpose(y:f32[8]{i}, wbar:f32[8]{i}):
  zbar:f32[8]{i} = wbar * y
  xbar:f32[1]{i} = psum_scatter(zbar, 'i')
  return xbar
```

### How to make the API convenient for users (and backward compatible)

But what user wants to write `pbroadcast`s? And what developer wants to break
lots of existing user code involving `psum`s which are not fed into unmapped
outputs? Not me!

Instead we can automatically insert the `pbroadcast`s. It's a bit analogous to how
we do automatic rank promotion at the `jax.numpy` layer, inserting broadcasts to
avoid rank mismatch errors in binary operators. But it's much simpler since we
don't need to contend with shape tuples. The typical rule is: whenever we see a
multi-arity operation where the operands disagree in their device variance
types, take the union of operands' device variance types' axis name sets and
insert `pbroadcast`s to lift each operand to the resulting device variance type.

Automatically inserting `pbroadcast`s just before they're needed may mean we apply
the same `pbroadcast` to the same operand multiple times, creating common
subexpressions. When we transpose, those could turn into a sum-of-`psum`s rather
than a `psum`-of-sum. We'll rely on the compiler to clean that up as appropriate.
If it's a problem then we could add some simple memoization to the
`pbroadcast`-insertion pass.

The user API for `all_gather` will mean `all_gather_p` by default (not
`all_gather_invariant_p`), covering the common case and meaning no `pbroadcast`s
must be inserted.

We can provide an option on `shmap` to disable this automatic insertion of
`pbroadcast`s, in which case it'll be up to the user to ensure type-correctness.
This explicit option may be appealing to some who want to be explicit about
where the `psum`s occur in the backward pass.

### How to implement the solution

The key to making the implementation lightweight is that **we aren't going to
add these types to avals or jaxprs**. At least, not at first. That can be
expensive because it requires updating the rest of JAX, e.g. all consumers of
avals and jaxprs may need to handle the new types. We're not falling for that
again!

Instead we're going to keep these extended types as metadata internal to
`shmap`, just like the current "replication checking for `out_specs`" machinery
is internal to `shmap`. Indeed this solution amounts to a relatively small
extension to that existing machinery: it was already tracking the same
information; now we're just adding the `pbroadcast`s.

We have at least two options for where to perform the `pbroadcast` insertion:
1. just before transposition, in the transpose rule, where we have a jaxpr of
   the computation to be transposed;
2. in every `shmap` body, whether eagerly executed or staged out, like the
   current "replication checking for `out_specs`" machinery.
The former may end up being easier since we only have to handle the jaxpr case,
and only linear primitives. But we'll start by trying the latter so the
implementation here is a strict revision/extension to the existing
replication-checking logic.

## Appendix: defining and motivating maps with unmapped inputs and outputs

For concreteness, we'll mostly focus on `shmap`, though these same ideas apply
to e.g. `pmap` and probably `xmap`.

An argument/input is _unmapped_ along a mesh axis when the corresponding entry
of `in_specs` doesn't mention that mesh axis's name. Logically it means that
each function instance along that mesh axis gets the same value for the
argument. To the caller, each operand is sliced according to the mesh axes over
which the operand is mapped, whereas there is no slicing for mesh axes over
which the operand is unmapped.

An output is _unmapped_ along a mesh axis when the corresponding entry of
`out_specs` doesn't mention that mesh axis's name. Logically it means each
function instance along that mesh axis must return the same value. To the
caller, each result of the `shmap` is formed by concatenating the return values
of every function instance along which the outputs are mapped, whereas for mesh
axes over which the output is unmapped only one copy of the value is used.

See [the `shmap`
JEP](https://docs.jax.dev/en/latest/jep/14273-shard-map.html) for examples
of unmapped inputs and outputs. For comparison, in `vmap` unmapped
inputs/outputs are indicated by using `in_axes` / `out_axes` of `None` (rather
than an `int`).

Here are reasons we like unmapped inputs and outputs for `shmap`:
* **Same expressiveness as `pjit`.** Anything `pjit` can do, the `shmap` escape
  hatch should be able to do too. Or else we'd have a lacking escape hatch! If
  we didn't have unmapped outputs in `shmap` then we couldn't express the same
  batch-parallel loss function computations as `pjit`.
* **Closed-over inputs.** Closed-over inputs essentially correspond to unmapped
  inputs, and...
* **Closure under transposition.** Once we have unmapped inputs, it's natural to
  be able to transpose to unmapped outputs.

So unmapped outputs are both canonical and useful!
