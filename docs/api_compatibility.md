(api-compatibility)=

# API compatibility

JAX is constantly evolving, and we want to be able to make improvements to its
APIs. That said, we want to minimize churn for the JAX user community, and we
try to make breaking changes rarely.

JAX follows a 3 month deprecation policy. When an incompatible change is made
to an API, we will make our best effort to obey the following procedure:
* the change will be announced in `CHANGELOG.md` and in the doc string for the
  deprecated API, and the old API will issue a `DeprecationWarning`.
* three months after the `jax` release that deprecated an API, we may remove the
  deprecated API at any time. Note that three months is a *lower* bound, and is
  intentionally chosen to be faster than that of many more mature projects. In
  practice, deprecations may take considerably longer, particularly if there are
  many users of a feature. If a three month deprecation period becomes
  problematic, please raise this with us.

We reserve the right to change this policy at any time.

## What is covered?

Only public JAX APIs are covered, which includes the following modules:

* `jax`
* `jax.dlpack`
* `jax.image`
* `jax.lax`
* `jax.nn`
* `jax.numpy`
* `jax.ops`
* `jax.profiler`
* `jax.random`
* `jax.scipy`
* `jax.tree_util`

Not everything in these modules is public. Over time, we are working to separate
public and private APIs. Public APIs are documented in the JAX documentation.
Additionally, our goal is that all non-public APIs should have names
prefixed with underscores, although we do not entirely comply with this yet.

## What is not covered?

*  anything prefixed with an underscore.
* `jax._src`
* `jax.core`
* `jax.linear_util`
* `jax.lib`
* `jax.interpreters`
* `jax.experimental`
* `jax.example_libraries`
* `jax.test_util`


These lists are not exhaustive.