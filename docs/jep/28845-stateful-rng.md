(stateful-randomness-jep)=
# JEP 28845: Stateful Randomness in JAX

[@jakevdp](http://github.com/jakevdp), *November 2025*

This document explores the addition of an **optional** stateful pseudo-random number generator (PRNG) for use in JAX; this is meant to be used alongside the classic functional PRNGs described in {ref}`pseudorandom-numbers` in cases where statefulness is convenient.

## Background

JAX has always required users to explicitly manage random state as part of its functional programming paradigm (see {ref}`prng-design-jep` for background on this). Although well-motivated, this is a frequently encountered [sharp bit](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers) for new users who are accustomed to stateful pseudorandom number APIs.

With the recent introduction of limited-scope [mutable refs](https://docs.jax.dev/en/latest/array_refs.html) in JAX, it is now possible to implement a stateful PRNG in JAX that retains most of the performance benefits of the existing functional PRNG, while providing a much more natural API for users familiar with NumPy, Pytorch, and other numerical computing libraries.

This JAX Enhancement Proposal (or [JEP](https://docs.jax.dev/en/latest/jep/index.html)) proposes the introduction of a Stateful PRNG API into {mod}`jax.experimental.random`, with a goal of eventual inclusino into {mod}`jax.random` itself.

## API Design

To align with best practices developed within the larger numerical Python community, we propose for the stateful PRNG API to align with To align with NumPy’s most recent PRNG API iteration, found in {class}`numpy.random.Generator`, and typically created using the {func}`numpy.random.default_rng` function. A full draft of the proposed implementation can be found at {jax-issue}`#28845`, but here we summarize the main features of the implementation.

A simplified version of the stateful PRNG Generator code looks like this (function and argument names follow the {mod}`numpy.random` APIs):

```python
def stateful_rng(seed: ArrayLike) -> StatefulPRNG:
  """Create a stateful PRNG Generator given an integer seed."""
  return StatefulPRNG(jax.random.key(seed), jax.new_ref(0))


@tree_util.register_dataclass
@dataclass(frozen=True)
class StatefulPRNG:
  """Stateful PRNG Generator class."""
  base_key: jax.Array
  counter: jax.core.Ref

  def key(self) -> jax.Array:
    """Generate a new jax PRNG key"""
    key = jax.random.fold_in(self.base_key, self.counter[...])
    jax.ref.addupdate(self.counter, ..., 1)  # increment counter
    return key

  def random(self, size: Sequence[int], dtype: DType = float):
    """Return random floats in the half-open interval [0, 1)"""
    return random.uniform(self.key(), shape=size, dtype=dtype)

  # uniform(), normal(), integers(), and others implemented similarly.
```

With this implementation exposed in the {mod}`jax.experimental.random` namespace, usage is virtually identical to that of {func}`numpy.random.default_rng`:

```python
>>> from jax.experimental.random import stateful_rng
>>> rng = stateful_rng(1701)
>>> rng.random((5,))
Array([0.09609699, 0.26730824, 0.5619041 , 0.24421775, 0.7715055 ], dtype=float32)
>>> rng.random((5,))  # state is updated -> new random draws!
Array([0.8131045 , 0.33873856, 0.88808906, 0.96005905, 0.7616446 ], dtype=float32)

>>> import numpy as np
>>> rng = np.random.default_rng(1701)
>>> rng.random((5,))
array([0.4020733 , 0.30563311, 0.67668051, 0.15821208, 0.79247763])
>>> rng.random((5,))
array([0.09419469, 0.36753944, 0.06388928, 0.96431608, 0.35200998])

```

Because the statefulness in {class}`jax.experimental.random.StatefulPRNG` is tracked via mutable refs, the random state will correctly update even if the generator is used in transformations like {func}`jax.jit`, which typically require pure functional semantics.

### Interaction with `vmap` and `shard_map`

The proposed stateful RNG design is based on refs, and so under `vmap` and `shard_map` it inherits the limitations of refs. So, for example, you cannot directly use an un-mapped `rng` within a vmapped function:

```python
rng = stateful_rng(0)

def f(x):
  return x + rng.uniform()

jax.vmap(f)(jnp.arange(10))
```
```pytb
Exception: performing an addupdate operation with vmapped value on an unbatched
           array reference of type Ref{int32[]}. Move the array reference to be
           an argument to the vmapped function?
```

For this reason we need the ability to split the generator in order to pass it to mapped or sharded code. For this we add a `split` method to the `StatefulPRNG` class that looks like this:

```python
class StatefulPRNG:
  ...

  def split(self, num: int | Sequence[int]) -> StatefulPRNG:
    return StatefulPRNG(
      base_key=jax.random.split(self.key(), num),
      counter=jnp.zeros(num, dtype=int),
    )
```

With this method present, the stateful rng can be explicitly split and passed to a vmapped function:

```python
rng = jax.experimental.random.stateful_rng(0)

def f(x, rng):
  return x + rng.uniform()

result = jax.vmap(f)(jnp.arange(5), rng.split(5))
print(result)  # [0.07174575 1.0163325  2.0435536  3.4391735  4.534091  ]
```

A similar approach would work for sharded computations, though `split` would likely have to grow a `sharding` argument.

This splitting brings up the question of what to do if a user attempts to generate random numbers directly from a split generator, like `rng.split(10).uniform()`. For this we follow the precedent of classic stateless `jax.random` APIs when receiving batched keys, and raise an informative error.

## Statistical Considerations

In the proposed design, the random state is tracked via a base key along with an integer counter that increments each time a key is generated. We chose this approach rather than mutating the key itself in order to avoid the pitfalls of iterative splits (see INSERT_REF_HERE); in particular it means that the stateful generator will always fully explore the 32-bit or 64-bit space of keys before looping back to zero and repeating the initial key.

## Advantages

The main advantage of this approach is familiarity: many users are familiar with NumPy, and familiar with its stateful RNG utilities. This would let them start using JAX more directly, without the learning curve of the unfamiliar functional PRNG API.

This does not just affect JAX users: for convenience, even JAX developers tend to context switch and use stateful NumPy APIs outside of transformations, where the functional PRNG is not necessary. This leads to confusion on the part of JAX users (see for example [this github discussion](https://github.com/jax-ml/jax/issues/30881)). Having a JAX-native stateful API would make it more convenient to always use JAX PRNGs in live demos and written tutorials.

Another pitfall of functional PRNGs is the possibility of accidental key reuse. Users unfamiliar with the need for explicit state may use keys multiple times, inadvertently generating statistically dependent random values (see for example [this StackOverflow question](https://stackoverflow.com/q/76135488)). By encouraging new JAX users to use a stateful PRNG, we avoid this silent trap.

Finally, the API affords the ability to call `rng.key()` in order to create a standard functional PRNG key, which can then be used in the typical functional mode: this is an easy onramp to explicitly-managed state in cases where it is warranted.

## Limitations

Implementing a stateful PRNG key via mutable refs comes with a few inherent limitations; in particular:

**Sequential dependence restricts the compiler:** Programs using such keys impose an inherent sequential dependence within the program, meaning that the compiler would not have the freedom to reorder operations that depend on pseudorandom values. The pitfall in this case is silent: it would be up to the user to recognize where this may become an issue, and instead switch to batched execution modes over pre-generated sequences of keys or values. Note, however, that this sequential dependence pitfall also exists when users follow the current usage recommendations in the JAX docs: [https://docs.jax.dev/en/latest/jax.random.html\#basic-usage](https://docs.jax.dev/en/latest/jax.random.html#basic-usage).

**Sequential dependence restricts the user:** Similarly, just as the compiler cannot reorder operations without changing the randomness, this sequential dependence also means the user cannot easily refactor code without changing the specific random draws. One potential example of this: suppose a stateful RNG is used within a neural network, and the user decides to swap an internal layer with one that has different random draws: this would consume a key and affect the random draws of all subsequent layers.

**Incompatiblity with remat:** Because mutable refs rely on JAX’s effect system, these APIs would not be usable in places where effects are not supported. In particular, this means that in JAX’s current implementation, stateful keys would not be compatible with `remat`, which might limit their usefulness within neural network implementations. The pitfall in this case is loud: attempting to use a mutable ref within remat will lead to an explicit error. There is a possibility that a future redesign of `remat` could remove this incompatibility (see {jax-issue}`#33018` for some progress on this).

**Refs cannot be return values:** Mutable refs cannot be present in the return values of transformed JAX functions, and the proposed stateful RNG object would inherit this limitation. This is also an explicit limitation: attempting to return a `StatefulPRNG` from a transformed function would lead to an explicit error.

## Evaluation

Our judgment is that the advantages of the stateful PRNG API potentially outweigh the limitations, and that we should introduce a new experimental {func}`~jax.experimental.random.stateful_rng` API in the {mod}`jax.experimental.random` module for now.
Once we get a feel for the usefulness of this, we may evenutally graduate this API to the {mod}`jax.random` module, perhaps with a `default_rng` alias in {mod}`jax.numpy.random`.
