---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(jax-101-random)=
# Pseudorandom numbers

<!--* freshness: { reviewed: '2026-07-09' } *-->

In most numerical libraries, drawing a random number means consulting a hidden
generator that updates itself between calls. JAX has no such generator, and no
hidden update. Randomness in JAX is driven by **keys**: ordinary immutable
array values that you pass around explicitly. Every function in
{mod}`jax.random` is a pure, deterministic function of its key argument — call
it twice with the same key and you get the same number twice, by design.

It's tempting to describe keys as "the generator state made explicit", but
that undersells the shift: there is no state *anywhere* in this design.
Nothing advances, nothing mutates. A key is a value, sampling is a function of
that value, and getting fresh randomness means deriving fresh key values. This
page covers why JAX works this way and the small set of habits for working
with keys well.

## Why not a global generator?

NumPy's `numpy.random` is the familiar stateful design: a global generator,
seeded once, silently advanced by every sampling call:

```{code-cell}
import numpy as np

np.random.seed(0)
print(np.random.uniform())
print(np.random.uniform())  # different: the hidden state advanced
```

The problem is that "the hidden state advanced" makes your results depend on
the exact order and number of sampling calls everywhere in your program:

```{code-cell}
np.random.seed(0)

def bar(): return np.random.uniform()
def baz(): return np.random.uniform()

def foo(): return bar() + 2 * baz()

print(foo())
```

This value is only reproducible because NumPy promises to run `bar()` before
`baz()`. Such sequencing promises are exactly what JAX needs to avoid: an
optimizing compiler should be free to reorder and parallelize work across
devices, and as we saw in {ref}`jax-101-tracing`, how many times your Python
code runs is an implementation detail of each transformation. JAX needs random
number generation that is **reproducible**, **parallelizable**, and
**vectorizable** — which rules out sampling functions that secretly read and
write shared state.

The solution isn't to make the generator state an explicit argument that you
shuttle in and out of functions. It's more radical: eliminate the state, and
make sampling a pure function of a key.

## Keys are values

Create a key from an integer seed with {func}`jax.random.key`:

```{code-cell}
import jax
from jax import random

key = random.key(42)
key
```

A key is a rank-0 array (shape `()`) with a special element type: its JAX
type is `key<fry>[]`, where `key<fry>` names the default Threefry PRNG
implementation.

```{code-cell}
print(jax.typeof(key))
```

Passing a key to a sampling function doesn't modify it or "use it up" in any
physical sense; the sample is simply a deterministic function of the key:

```{code-cell}
print(random.normal(key))
print(random.normal(key))  # same key, same value — necessarily
```

This means reproducibility is automatic: results depend only on the key
values your program constructs, never on execution order, call counts, or
which device ran what.

The flip side is that *distinct* random numbers require *distinct* keys, which
leads to the one rule of JAX randomness:

**Never reuse a key (unless you want identical outputs).** Feeding the same
key to two different samplers produces correlated results — depriving your
program of lifegiving chaos.

## Deriving new keys

To get fresh keys, derive them from a key you already have.
{func}`jax.random.split` deterministically produces any number of
statistically independent new keys:

```{code-cell}
key = random.key(42)
key, subkey = random.split(key)
print(random.normal(subkey))
```

The naming convention: the `subkey` is consumed immediately by a sampling
function, while the `key` is kept for deriving more keys later. There's
nothing special about which is which — all outputs of `split` are equally
valid, independent keys — but the convention helps track what's been consumed.
Treat a key that's been passed to `split` or to a sampler as spent.

`split` produces as many keys as you ask for in one shot:

```{code-cell}
key = random.key(42)
subkeys = random.split(key, num=4)
[float(random.normal(k)) for k in subkeys]
```

And {func}`jax.random.fold_in` derives a new key from a key and an integer,
which is ideal for generating a per-step or per-example key without carrying
any key-threading through your loop:

```{code-cell}
key = random.key(42)
for step in range(3):
  step_key = random.fold_in(key, step)
  print(f"step {step}: {random.normal(step_key)}")
```

Note the shape of this pattern: every `step_key` is derived directly from one
parent key, not from its predecessor. That's deliberate, and worth dwelling
on.

### Keep the key tree wide, not deep

Your program's keys form a tree, rooted at the seed, growing by derivation.
A common habit — one you'll see in older JAX code, including older versions of
these docs — is to grow that tree as a long chain, splitting off each step's
key from the previous step's:

```python
for step in range(num_steps):
  key, subkey = random.split(key)   # each key derived from the last: avoid!
  ...
```

Prefer wide derivation instead — all step keys hanging off one parent, via a
single `split(key, num_steps)` or via `fold_in(key, step)` as above. The
chained version has two problems, one computational and one statistical:

- **It serializes.** Each derivation depends on the previous one, so a chain
  of a million steps means a million sequential hash applications. Wide
  derivation is a single batched operation, free to vectorize and
  parallelize.

- **It courts collisions.** For a *fixed* key, the PRNG's underlying hash is
  a pseudorandom *permutation* of its input, so the keys produced by one
  `split` (or by `fold_in` over distinct integers) are guaranteed distinct.
  But viewed as a function *of the key*, the hash is not a permutation — it
  behaves like a random function. Every derivation hop is therefore an
  independent chance for two keys in your tree to coincide, and over a long
  chain in the default 64-bit key space, collision probability accumulates
  toward the birthday bound. A collision means identical random streams from
  the point of collision onward.

A handful of chained splits is harmless — the collision math only bites at
scale, and plenty of correct code splits a key a few times in sequence. But
for anything proportional to the length of training or the size of a dataset,
derive keys widely from a common parent.

## No sequential equivalence

NumPy guarantees that sampling N numbers one at a time yields the same
sequence as sampling N at once. JAX deliberately makes no such promise:

```{code-cell}
key = random.key(42)
subkeys = random.split(key, 3)
print("individually:", np.stack([random.normal(k) for k in subkeys]))

key = random.key(42)
print("all at once: ", random.normal(key, shape=(3,)))
```

Sequential equivalence would impose exactly the kind of ordering constraint
JAX's design exists to avoid. Giving it up means samples drawn from
independent keys don't depend on each other in any order — so generation can
be freely vectorized and sharded.

Since keys are just arrays, they compose with everything else in JAX. You can
`vmap` a sampler over a batch of keys:

```{code-cell}
import jax
jax.vmap(random.normal)(subkeys)
```

With the default PRNG implementation, this is *exactly* equivalent to calling
`random.normal` on each key separately — vectorizing over keys doesn't change
the values.

```{note}
These docs use the typed keys created by {func}`jax.random.key`. You may also
encounter older code using {func}`jax.random.PRNGKey`, which produces a raw
`uint32` array — it still works, but it's easy to misuse (nothing stops you
from doing arithmetic on it) and it doesn't record which PRNG implementation
it belongs to. Prefer `jax.random.key` in new code, and convert at boundaries
with {func}`jax.random.key_data` and {func}`jax.random.wrap_key_data` when
interfacing with systems that need raw arrays. See {ref}`jep-9263` for the
full story.
```

## Design and implementations

In one line: JAX's PRNG is a counter-based Threefry hash combined with a
functional splitting model, chosen so that generation has no sequencing
constraints at all — see {ref}`prng-design-jep` for the design rationale.

Threefry is the default of several available implementations. Alternatives
(selected per-key via the `impl` argument to {func}`jax.random.key`, or
globally via the `jax_default_prng_impl` config) trade off generation speed on
TPUs, shardability, bit-for-bit identical results across platforms, and exact
`vmap`-over-keys semantics. The {mod}`jax.random` module documentation has the
full comparison table; the default is the right choice unless PRNG generation
shows up in your profiles.

{mod}`jax.random` itself offers samplers for a wide range of distributions —
uniform, normal, categorical, permutations, and many more — all taking a key
as their first argument.

## Next steps

Keys handle randomness while keeping every function pure. The remaining
expressiveness topic is state — values that evolve as a program runs, and
genuine in-place mutation — covered in {ref}`jax-101-state`.
