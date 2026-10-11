(type-checking-jax)=
# Type checking JAX programs

<!--* freshness: { reviewed: '2026-10-11' } *-->

This guide covers practical advice for adding Python type annotations to
*your* JAX code and running static type checkers such as
[pytype](https://github.com/google/pytype), [mypy](https://github.com/python/mypy),
and [pyright](https://github.com/microsoft/pyright).

JAX's own annotation conventions and longer-term roadmap are documented in
{mod}`jax.typing` and {doc}`jep/12049-type-annotations`. This page focuses on
what to do in application code.

## Recommended annotations for arrays

For public helper functions that accept array-like values and return JAX
arrays, prefer:

- **Inputs:** {obj}`jax.typing.ArrayLike` — anything JAX can safely cast to an
  array (`jax.Array`, NumPy arrays, Python/NumPy scalars).
- **Outputs:** {class}`jax.Array` — a concrete JAX array (or tracer inside a
  transform).

```python
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp

def normalize(x: ArrayLike) -> Array:
  x_arr = jnp.asarray(x)
  return x_arr / jnp.linalg.norm(x_arr)
```

This matches the pattern used across much of JAX's public API. See
{mod}`jax.typing` for more detail, including why sequences such as `list` /
`tuple` are usually *not* treated as array inputs.

For local variables that hold JAX arrays, annotate with `jax.Array` as well
(see also {doc}`101/arrays`).

## What static type checkers can and cannot do

Static checkers are useful for catching simple mistakes (wrong argument
names, missing returns, obvious type mismatches). They are a weaker fit for
full correctness checking of JAX programs, for a few structural reasons:

1. **Duck typing.** Values flowing through JAX code are often interchangeable
   at runtime (`jax.Array`, NumPy arrays, tracers). Annotations that are too
   narrow produce false positives on valid code.
2. **Transformations.** Decorators such as {func}`jax.jit`, {func}`jax.grad`,
   and {func}`jax.vmap` rewrite call signatures in ways that static checkers
   historically handle poorly (see the long-standing
   [mypy decorator discussion](https://github.com/python/mypy/issues/1927)).
3. **Pytrees.** Nested containers of arrays are recursive and generic; that
   combination is awkward for many checkers, especially older pytype
   versions.

In practice, most JAX user code aims for *documentation-quality* annotations
plus IDE autocomplete, not exhaustive Level-3 proof of type safety. When a
checker cannot follow a valid pattern, a narrowly scoped
`# type: ignore[...]` (mypy/pyright) or `# pytype: disable=...` (pytype) on
that line is appropriate.

### Checker notes

| Checker | Typical role with JAX |
| --- | --- |
| **pytype** | Common inside Google; used heavily on code that depends on JAX. |
| **mypy** | Common in open-source projects; JAX's own CI runs mypy on parts of the tree. |
| **pyright** | Powers VS Code / Pylance completions; often the most visible day-to-day checker. |

You do not need to satisfy all three equally in application code. Pick the
checker your project already uses and keep annotations simple enough that it
stays helpful.

## Transformations and lost types

After a transformation, checkers often treat the result as effectively
untyped. A reliable workaround is to **restate the type at the use site**:

```python
from jax import Array, jit
import jax.numpy as jnp

def f(x: Array) -> Array:
  return jnp.sin(x) + 1

f_jit = jit(f)
y: Array = f_jit(jnp.arange(3.0))
```

The same pattern applies to {func}`jax.grad`, {func}`jax.vmap`, and friends:
annotate the value you assign, rather than expecting the decorator's return
type to carry precise information through the checker.

## Pytrees

Pytrees (dicts, lists, tuples, custom registered nodes, …) are central to JAX
APIs, but they are hard to type precisely. Practical options, from loosest to
stricter:

1. **`Any` for non-trivial pytrees.** This is still a reasonable default when
   the structure is dynamic or deeply nested. It documents intent poorly, but
   it avoids lying to the type checker.
2. **Concrete container annotations** when the shape is stable, for example
   `dict[str, Array]`, `tuple[Array, Array]`, or a {class}`typing.NamedTuple`
   / {class}`typing.TypedDict` of arrays.
3. **Custom pytree classes** (for example via {func}`jax.tree_util.register_pytree_node`
   or library helpers) with explicit attribute annotations on the class.

What usually does *not* work well today is a single recursive generic alias
that precisely describes every possible pytree; pytype in particular has
limited support for that pattern. Prefer the simplest annotation that still
helps readers and your IDE.

## Shape-aware typing (optional)

{mod}`jax.typing` intentionally does **not** encode array shapes or dtypes in
annotations. If you want shape-checked APIs (for example `Float32[Array, "b h w c"]`),
look at the third-party [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
project. It is widely used in the ecosystem and interoperates with runtime
checkers; it is not part of core JAX.

## Ecosystem libraries

Libraries built on JAX have their own typing conventions:

- [Flax](https://flax.readthedocs.io/) — see Flax docs for modules/structs and
  any library-specific guidance around pytree annotations.
- [Equinox](https://docs.kidger.site/equinox/) — modules are dataclasses/pytrees
  with first-class typing ergonomics; often paired with jaxtyping.

Prefer each library's documentation over inventing a parallel scheme in JAX
application code.

## Runtime checking during tests

Some projects enable a **runtime** type checker (for example
[beartype](https://github.com/beartype/beartype) or
[typeguard](https://github.com/agronholm/typeguard)) in tests only. That can
catch mistakes static checkers miss, at the cost of only covering executed
paths. This is optional and orthogonal to the `ArrayLike` / `Array`
conventions above.

## Further reading

- {mod}`jax.typing` — current annotation helpers and best-practice snippet
- {doc}`jep/12049-type-annotations` — design roadmap and non-goals for JAX itself
- {doc}`101/arrays` — `jax.Array` as the user-facing array type
- {doc}`101/pytrees` — pytree concepts
