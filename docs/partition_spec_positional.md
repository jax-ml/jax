PartitionSpec positional indices
=================================

JAX now supports numeric / positional entries in `PartitionSpec` that are
expanded against the active mesh. This is intended to make `shard_map`-based
library code less coupled to particular mesh axis names.

Semantics
---------

- Non-negative integers `n` refer to the `n`-th axis name of the active mesh
  (i.e. `mesh.axis_names[n]`).
- Negative integers behave like Python indexing (e.g. `-1` refers to the last
  axis).
- A single `-1` may appear in a `PartitionSpec` entry and expands to the tuple
  of all mesh axes not otherwise mentioned in the same `PartitionSpec`. If the
  expansion is empty it is effectively removed (treated like `None`).
- At most one `-1` may appear in a single partition spec; using `-1` multiple
  times is an error.

Examples
--------

Assume a mesh created as::

  mesh = jax.make_mesh((2, 2), ("i", "j"))

- `P(0)` -> `P("i")`
- `P(0, 1)` -> `P("i", "j")`
- `P(-1)` -> `P(("i", "j"))` (shard across all mesh axes)
- `P(0, -1)` -> expands with `0` mapping to `"i"` and `-1` to the remaining
  axis names (here `"j"`) producing `P("i", "j")`.

Error cases
-----------

- Integer indices out of range for the active mesh raise `ValueError`.
- Using `-1` more than once in the same `PartitionSpec` raises `ValueError`.

Best practices
--------------

- Use numeric indices when writing library-level `shard_map` functions that
  should be agnostic to users' mesh axis naming conventions.
- Prefer `P()` / `P(None)` for replication; `P(-1)` is specifically for
  sharding across all mesh axes.

See also
--------

- Issue: {jax-issue}`#34752`
- `jax.shard_map` documentation and examples
