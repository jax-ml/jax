---
orphan: true
---
(deprecate-pmap)=
#`jax.pmap` deprecation

## What's going on?

This document is a work in progress and will be fleshed out in time for the JAX
0.8.0 release in October 2025. With that release, we will change the default
implementation of `jax.pmap` to an implementation based on `jax.jit` and
`jax.shard_map` via setting the JAX configuration flag `jax_pmap_shmap_merge` to
`True`.

## How do I know if this change broke my code?

## How can I disable this change for now?

Until [], it will be possible to temporarily use the old version of `jax.pmap`
by

- Setting the shell environment variable `JAX_PMAP_SHMAP_MERGE` to something
  false-like (e.g., 0);
- Setting the boolean flag `--jax_pmap_shmap_merge` to something false-like if
  your code parses flags with absl.
- Using this statement in your main file or anywhere before you call `jax.pmap`:
  ```python
  import jax
  jax.config.update("jax_pmap_shmap_merge", False)
  ```

NOTE: Please file a [bug](https://github.com/jax-ml/jax/issues) with a
reproducer so we can resolve it as quickly as possible and remove the old
version of `jax.pmap`.

<!--* freshness: { reviewed: '2025-09-25' } *-->
