---
orphan: true
---
(migrate-pmap)=
# Migrating to the new `jax.pmap`

## What's going on?

As of JAX 0.8.0, the default implementation of `jax.pmap` will be based on
`jax.jit` and
[`jax.shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html). The
new implementation is **_not_** a perfect replacement for the original and this
doc gives guidance for users who run into trouble.

This change makes `jax.pmap` integrate well with JAX shardings and simplifies
the implementation.

## Help! Fix me now!

**IMPORTANT**: This option is not a permanent fix. Until January 15, 2026, it
will be possible to temporarily use the old version of `jax.pmap` by doing one
of the following:

- Setting the shell environment variable `JAX_PMAP_SHMAP_MERGE` to something
  false-like (e.g., 0);
- Setting the boolean flag `--jax_pmap_shmap_merge` to something false-like if
  your code parses flags with `absl-py`.
- Using this statement in your main file or anywhere before you call `jax.pmap`:
  ```python
  import jax
  jax.config.update("jax_pmap_shmap_merge", False)
  ```

**NOTE**: Please file a [bug](https://github.com/jax-ml/jax/issues) with a
reproducer and tag [@danielsuo](https://github.com/danielsuo/) so we can resolve
it as quickly as possible under the new `jax.pmap`.

## How can I fix my code for the new `jax.pmap`?

Below are common errors we're collecting and suggestions for fixing them. This
is more work than setting `jax_pmap_shmap_merge=False`, but a more long-term
solution. However, we still recommend that new or important code be migrated to
`jax.shard_map`.

### `ValueError: Received incompatible devices ...`

#### Example
```
ValueError: Received incompatible devices for jitted computation. Got argument a
of allclose with shape float32[100] and device ids [0] on platform TPU and
argument b of allclose with shape float32[100] and device ids [0, 1] on platform
TPU
```
#### How this can happen

- `jax.pmap` no longer silently reshards inputs, as per the behavior of
  `jax.jit` and `jax.shard_map`. As a result, if inputs are sharded differently
  from how your `jax.pmap` expects, it will raise.

#### How to fix

- Pass an appropriate `jax.NamedSharding` to `jax.device_put` to explicitly
  reshard any offending inputs.
- Alternatively, redefine your `jax.pmap` with the appropriate `in_axes`,
  `backend`, and / or `devices` keywords to ensure `jax.pmap`'s mesh and
  expected input shardings match your operands.

### `ValueError: The context mesh ... should match the mesh passed to shard_map`

#### Example
```
ValueError: The context mesh AbstractMesh('x': 1, axis_types=(Manual,),
device_kind=TPU v3, num_cores=1) should match the mesh passed to shard_map
Mesh('y': 4, axis_types=(Auto,))
```

#### How this can happen

- This error can appear when nesting multiple `jax.pmap`s. This behavior is no
  longer supported since the `jax.pmap` API would not know anything about inner
  calls to `jax.pmap` and therefore not know about inner mesh axes.

#### How to fix

- Migrate to `jax.shard_map`. A single `jax.shard_map` can parallelize along
  multiple axes of inputs, with each of those axes assigned to the relevant axes
  of the device mesh.
- Alternatively, you can nest `jax.shard_map` calls or use `jax.smap`, which
  makes it easier to drop into [manual
  parallelism](https://docs.jax.dev/en/latest/notebooks/shard_map.html) mode one
  mesh axis at a time. This approach greatly simplifies nested parallelism.

## Performance implications

### Host local array to global array round-trip conversion

In multi-process JAX programs (i.e., `jax.process_count() > 1`), arrays might be
not be [fully
addressable](https://docs.jax.dev/en/latest/_autosummary/jax.Array.is_fully_addressable.html)
(i.e., "host local"), so the new `jax.pmap` will reshard the host-local array
into a global one before passing to `jax.jit` of `jax.shard_map` and back into a
host-local array when returning to user code.

This round-trip conversion cannot be avoided, so if the performance penalty is
too great, we recommend migrating your code to `jax.shard_map`.

### `int` array indexing

Indexing into a sharded array with an int (e.g., `arr[0]`) may now execute a
rank reduction computation. Depending on your use case, there may be
workarounds:

1. In a typical training loop, we might use a `jax.pmap`ed update function to
   operate on / carry training state and grab resulting metrics from the first
   `jax.pmap`'ed device for logging. In this case, it may be possible to
   use `None` for the relevant `in_axes` and `out_axes` passed to `jax.pmap`.
   This lets `jax.pmap` handle replication and will return an
   appropriately-shaped result that looks like it's from a single device for,
   say, logging metrics.
2. More generally, you can get the first shard of data without a reshape via
   `arr[0:1]` or `arr.addressable_shards[0].data`. Note that this will have a
   leading `(1,)` dimension that your code will need to handle.

## Migrating to `jax.shard_map`

In many cases, users can migrate from `jax.pmap` to `jax.jit(jax.shard_map)` by
calling
[`jax.make_array_from_process_local_data`](https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_process_local_data.html)
on their inputs and passing to `jax.jit(jax.shard_map)`. While the performance
hit from converting to and from global arrays remains, it is no longer in the
dispatch path as in the `jax.shard_map` implementation of `jax.pmap` and can
often be overlapped with compute or be called infrequently (i.e., before a train
loop and for occasionally grabbing metrics).

<!--* freshness: { reviewed: '2025-09-29' } *-->
