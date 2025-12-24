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

### `JaxRuntimeError: INVALID_ARGUMENT: CopyArrays ... same size`

#### Example
```
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: CopyArrays only supports destination
device list of the same size as the array device lists.
```

#### How this can happen

- This error can appear in a multi-host setting (i.e., `jax.process_count() > 1`)
  where users try to index into a sharded array (e.g., `x[0]`) with the intention
  of grabbing what is semantically a replica. Please see [Appendix A](#appendix-a)
  for more details.

#### How to fix

Instead of `x[0]`, use one of these approaches:

- **Access local data directly**: Use `.addressable_shards[0].data` to get the
  local shard without triggering global resharding.
- **Explicit resharding**: Use `jax.device_put(x, sharding)` with an appropriate
  `NamedSharding` to explicitly control how data is distributed.


### Using `jax.stages.Lowered` returned by `jax.pmap(f).lower(*args)`

Because of the default call path of a `jax.stages.Lowered` object, we miss the
conversion from host-local arrays to global arrays to pass into the underlying
`jax.shard_map(f)` as well as the conversion back from global arrays to host-local
arrays for the output. This can lead to unexpected behavior in the multi-host
setting. In this case, we recommend users call `jax.experimental.multihost_utils`'s
`host_local_array_to_global_array` on inputs and `global_array_to_host_local_array`
on outputs of `.compile()(*args)`to perform the necessary conversions.

### `JaxRuntimeError: INTERNAL: Core halted unexpectedly`

#### Example
```
jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly: Assertion args:
0x00000000 0x00000000 0x00000000 INTERNAL: Accelerator device halted prematurely,
perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc
TensorCoreSequencer:1:0x160 (from TensorCoreSequencer:1:0x208): scheckne:
```

#### How this can happen

- This error typically occurs in multi-host settings when process synchronization
  barriers are not properly aligned. The new `jax.pmap` implementation may have
  different synchronization semantics compared to the old implementation.

#### How to fix

- Replace any custom process barrier implementations with
  `jax.experimental.multihost_utils.sync_global_devices()`. This ensures all
  processes reach the same synchronization point before proceeding.
  ```python
  from jax.experimental import multihost_utils as mhu

  # Instead of custom barriers
  mhu.sync_global_devices("barrier_name")
  ```

## Performance implications

### `int` indexing into sharded arrays

The new implementation of `jax.pmap` uses `NamedSharding` instead of the legacy
`PmapSharding`. We've observe a common pattern with the old `jax.pmap` where
users shard stacked copies of an array to replicate (e.g., via
`jax.device_put_replicated`). These "sharded-but-really-replicated" arrays
suffer unnecessary communication overhead when `int` indexing (e.g., `x[0]`)
because JAX does not know the arrays are actually replicated. For a more
thorough discussion, please see the note on the multi-host setting in
[Appendix A](#appendix-a).

#### Option 1: Prevent unintended sharding (recommended)
Avoid creating the leading sharded dimension entirely.

- Use `jax.pmap`'s `out_axes=None` for arguments that should remain replicated.
The output will be fully replicated (e.g., `P(None, None)`), making access
cheap.
- For inputs: When using `jax.device_put`, specify `jax.P()` (fully replicated)
in the partition spec rather than relying on utilities that stack and shard.
(Note: `jax.device_put_replicated` and `jax.device_put_sharded` are deprecated
because they confusingly produce sharded arrays rather than replicated ones).

#### Option 2: Access local data directly
If you must work with a sharded array (or want potentially fewer changes to
code), you can access the local data shard directly without triggering JAX's
distributed consistency checks. Note that this is only recommended when bringing
data back to host (e.g., for logging, checkpointing). Instead of `x[0]`, use
`addressable_shards`:

```python
# Old slow way:
# result = x[0]

# New fast way:
# x.addressable_shards is a list of shards on the current process.
# We grab the first one, extract the data, and remove the leading dimension.
result = x.addressable_shards[0].data.squeeze(0)
```

In the example of `x` with shape `(8, 3, 4)`, `x.addressable_shards[0].data`
returns the local chunk of shape `(1, 3, 4)`. Calling `.squeeze(0)` results in
the desired `(3, 4)` shape without any cross-device communication. Both
solutions will eliminate the `_gather` operations seen in profiling.

### Host local array to global array round-trip conversion

In multi-process JAX programs (i.e., `jax.process_count() > 1`), arrays might be
not be [fully
addressable](https://docs.jax.dev/en/latest/_autosummary/jax.Array.is_fully_addressable.html)
(i.e., "host local"), so the new `jax.pmap` will reshard the host-local array
into a global one before passing to `jax.jit` of `jax.shard_map` and back into a
host-local array when returning to user code.

This round-trip conversion cannot be avoided, so if the performance penalty is
too great, we recommend migrating your code to `jax.shard_map`.

### Transforming `jax.pmap` e.g., `jax.jit`

We recommend keeping `jax.pmap` as the top-level transform since it is more
performant than under another transform. However, if your code must put
`jax.pmap` under another transform and the performance penalty is
unacceptable, please file a bug as described above.

### Buffer donation with `donate_argnums`

Buffer donation with `donate_argnums` is fully supported in the new `jax.pmap`
implementation, but performance depends on whether inputs are correctly sharded:

- **Correctly sharded inputs (fast path)**: Arrays with the expected local
  sharding use a zero-copy rewrap. Donation invalidates the original array as
  expected, with no additional memory overhead.

- **Incorrectly sharded inputs (slow path)**: Arrays that require resharding
  must be copied first, then the original is deleted. This causes a **brief 2x
  memory spike** before the original is freed. A warning is logged when this
  occurs.

To maximize donation efficiency, ensure your inputs are correctly sharded
before calling `pmap`. If you see the resharding warning and memory is tight,
consider migrating to `jax.shard_map` where you have full control over
input/output sharding.

## Migrating to `jax.shard_map`

In many cases, users can migrate from `jax.pmap` to `jax.jit(jax.shard_map)` by
calling
[`jax.make_array_from_process_local_data`](https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_process_local_data.html)
on their inputs and passing to `jax.jit(jax.shard_map)`. While the performance
hit from converting to and from global arrays remains, it is no longer in the
dispatch path as in the `jax.shard_map` implementation of `jax.pmap` and can
often be overlapped with compute or be called infrequently (i.e., before a train
loop and for occasionally grabbing metrics).

(appendix-a)=
## Appendix A: More details about `int` indexing into sharded arrays.

### What should `x[0]` return?

In **NumPy**, `x[0]` returns a rank-reduced array representing the first slice
along the first dimension. For example, if `x = np.ones((8, 3, 4))`, then `x[0]`
returns an array of shape `(3, 4)`.

In **JAX** (`jax.numpy`), `x[0]` semantically works the same way: it returns the
rank-reduced slice of the logical array `x`. However, performance depends on how
`x` is sharded or replicated across devices. Consider an array `x` with shape
`(8, 3, 4)` distributed across 8 devices (using `jax.P` as the short name for
`jax.sharding.PartitionSpec`P):

1.  **Fully Replicated:** `jax.P(None, None, None)`
    If `x` is fully replicated, every device holds a complete copy of the `(8,
    3, 4)` array. `x[0]` will have the shape `(3, 4)` and a partition spec
    `jax.P(None, None)`. Since every device already has `x`, this operation will
    slice on each device independently and requires **no communication**.

2.  **Sharded on Non-Leading Dimension:** `jax.P(None, 'x', None)`
    If `x` is sharded along the second dimension, `x[0]` results in shape `(3,
    4)` with partition spec `jax.P('x', None)`. Since the first dimension (the
    one being sliced) is unsharded, this operation also requires **no
    communication**.

3.  **Sharded on Leading Dimension:** `jax.P('x', None, None)`
    If `x` is sharded along the first dimension, `x[0]` results in shape `(3,
    4)` with partition spec `jax.P(None, None)`.
    *   **The Issue:** Because the first dimension is sharded, the data for
        `x[0]` physically resides *only* on the first device. To satisfy the
        output sharding `jax.P(None, None)` (which implies replication), JAX
        must broadcast the data from the first device to all other devices. This
        requires **communication**; JAX will gather the *entire* array of shape
        `(8, 3, 4)` to each device and then take a slice.

### The common performance pitfall

A common pattern among `jax.pmap` users involves arrays that are **semantically
replicated** (the user intends for them to be identical everywhere) but are
**physically sharded** (stacked along the leading dimension).

This happens implicitly (e.g., via `jax.pmap(..., out_axes=0)`) or explicitly
(e.g., via `jax.device_put_replicated`). Users often try to retrieve metrics or
checkpoints by calling `unreplicate` or `x[0]`, assuming it is a cheap
operation.

#### Example: The "unreplicate" anti-pattern

```python
from flax import jax_utils
import jax.numpy as jnp
import jax

# jax_utils.replicate calls jax.device_put_replicated.
# This stacks num_devices copies and SHARDS them over the stacked dimension.
# Logical Shape: (8, 3, 4) | Sharding: P('x', None, None)
train_state = jax_utils.replicate({'params': jnp.zeros((3, 4))})

# out_axes=0 by default, so the output remains sharded along dim 0.
train_step_pmapped = jax.pmap(lambda x: x)

# jax_utils.unreplicate performs a jax.tree_map(lambda x: x[0], tree).
# Users do this to grab metrics, log param statistics, checkpoint, etc.
train_state = jax_utils.unreplicate(train_step_pmapped(train_state))
```

#### The consequence
Even though the user knows `train_state` contains identical data on every
device, JAX sees an array with `shape (8, 3, 4)` and spec `jax.P('x', None,
None)` i.e., an array that is sharded along its leading dimension. JAX cannot
safely assume the data is identical on each device. Therefore, `x[0]` triggers a
gather of the entire array to all devices before slicing to ensure correctness.
This unnecessary communication causes performance degradation (visible as
_gather operations in a stack trace).

```
train
  └─ jax_utils.py:48  unreplicate
       └─ tree_util.py:354  tree_map
            └─ jax_utils.py:50  <lambda> (performing x[0])
                 └─ array.py:335  __getitem__
                      └─ indexing.py:734  rewriting_take
                           │
                           ▼
                           └─ indexing.py:784  _gather
                                └─ slicing.py:324  gather
                                     └─ PjitFunction(gather)
```

### Why was "old `jax.pmap`" fast?
Historically, `pmap` used `PmapSharding`, which had a fast-path optimization in
`jax.Array`'s `__getitem__` allowing it to return an array with a
`SingleDeviceSharding` (data residing on only one device).

However, current JAX uses `NamedSharding`. We do not strictly replicate the
legacy behavior because it breaks the semantics of array indexing. If we allowed
`x[0]` to return a `SingleDeviceSharding` array in a general context (e.g., in
the middle of a train step instead of when trying to bring data back to host for
reporting), only one device would have data while others would have nothing.
This is computationally problematic for subsequent operations.

The slowdown users experience now is JAX enforcing correct semantics: if you ask
for `x[0]` from an array sharded along its leading dimension, you get a fully
replicated result available on all devices, which requires communication.

### A note on the multi-host setting
`x[0]` will still give you the first slice along the first dimension of the
*logical* global array. In the multi-host setting, we will see a more drastic
version of the performance issues described above as all the hosts gather the
entire array to each device before slicing. In certain cases, users can even
face hard errors (e.g., `INVALID_ARGUMENT: CopyArrays only support...`).

In multi-host settings (e.g., 4 hosts × 2 devices = 8 devices total):

1. A global array with shape `(8, ...)` and `PartitionSpec('x')` has each slice
   distributed across all 8 devices spanning all hosts.

2. When you call `x[0]`, JAX needs to slice the first element and reshard the
   result so it's available to all hosts.

3. The `CopyArrays` operation in XLA requires source and destination to have the
   same device count. But each host only sees its *local* subset of devices (2
   in this example), not all 8. When JAX tries to create a resharded array, the
   device list mismatch triggers the error.

<!--* freshness: { reviewed: '2025-12-19' } *-->
