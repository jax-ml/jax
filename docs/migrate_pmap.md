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
doc gives guidance for users who run into trouble

This change makes `jax.pmap` integrate well with JAX shardings and simplifies
the implementation (see {doc}`jep/14273-shard-map` for more rationale).

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
jax.errors.JaxRuntimeError: INVALID_ARGUMENT: CopyArrays only supports
destination device list of the same size as the array device lists.
```

#### How this can happen

- This error can appear in a multi-host setting (i.e.,
  `jax.process_count() > 1`)
  where users try to index into a sharded array (e.g., `x[0]`) with the
  intention of grabbing what is semantically a replica. Please see
  [Appendix A](#appendix-a) for more details.

#### How to fix

Instead of `x[0]`, use one of these approaches:

- **Access local data directly**: Use `.addressable_shards[0].data` to get the
  local shard without triggering global resharding.
- **Explicit resharding**: Use `jax.device_put(x, sharding)` with an appropriate
  `NamedSharding` to explicitly control how data is distributed.

### Using `jax.stages.Lowered` returned by `jax.pmap(f).lower(*args)`

Because of the default call path of a `jax.stages.Lowered` object, we miss the
conversion from host-local arrays to global arrays to pass into the underlying
`jax.shard_map(f)` as well as the conversion back from global arrays to
host-local arrays for the output. This can lead to unexpected behavior in the
multi-host setting. In this case, we recommend users call
`jax.experimental.multihost_utils`'s `host_local_array_to_global_array` on
inputs and `global_array_to_host_local_array` on outputs of `.compile()(*args)`
to perform the necessary conversions.

### `JaxRuntimeError: INTERNAL: Core halted unexpectedly`

#### Example

```
jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly: Assertion args:
0x00000000 0x00000000 0x00000000 INTERNAL: Accelerator device halted
prematurely, perhaps due to an on-device check-failure. Node 0 halted
unexpectedly at tag:pc
TensorCoreSequencer:1:0x160 (from TensorCoreSequencer:1:0x208): scheckne:
```

#### How this can happen

- This error typically occurs in multi-host settings when process
  synchronization barriers are not properly aligned. The new `jax.pmap`
  implementation may have different synchronization semantics compared to the
  old implementation.

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

In multi-process JAX programs (i.e., `jax.process_count() > 1`), arrays might
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

For the best support, we recommend migrating from `jax.pmap` to
`jax.jit(jax.shard_map)`. `jax.shard_map` allows you to treat your entire device
cluster as a single computational fabric.

While the new `jax.pmap` is itself built on `shard_map`, migrating your code
gives you explicit control over data distribution and collective operations.
Migrating involves updates to three primary areas:

### 1. The pmapped function itself (Rank-preserving vs. Rank-reducing)

#### Update your mapped function

The "mapped function" is the function you pass to `jax.pmap` or `jax.shard_map`
(often via a decorator). When migrating, the biggest change within the function
body itself is how array ranks and shapes are handled. While it's possible that
very few if any changes are needed, you should carefully verify any
rank-sensitive logic.

`jax.pmap` is a **rank-reducing map**: it "unstacks" each array along the mapped
axis. For example, if you map over a `(8, 128)` array on 8 devices, the code
inside `jax.pmap` sees an array of shape `(128,)`.

In contrast, `jax.shard_map` is a **rank-preserving map**: it slices or
"unconcatenates" the array into blocks. Using the same example on a mesh of size
8, the code inside `jax.shard_map` sees an array of shape `(1, 128)`.

- **Rank adjustments**: Because `shard_map` slices the array, keeping an
  explicit dimension for each mapped axis instead of unstacking it, you may
  need to adjust how you treat those dimensions.

  ```python
  # pmap style (rank-reduced)
  def mapped_fn(x):
    # x has shape (128,)
    return jnp.dot(x, weights)

  # shard_map style (rank-preserving)
  def mapped_fn(x):
    # x has shape (1, 128)
    # Option 1: restores pmap rank
    return jnp.dot(x.squeeze(0), weights)

    # Option 2: use matmul (handles the leading dimension naturally)
    # return jnp.matmul(x, weights)

    # Option 3: indexing
    # return jnp.dot(x[0], weights)
  ```

Many JAX functions are sensitive to array rank and may behave differently or
raise errors when moving from `pmap` to `shard_map`. Be particularly careful
with reductions (e.g., `jnp.sum`, `jnp.mean`, `jnp.max`) when the `axis` is not
specified, linear algebra operations (`jnp.dot`, `jnp.matmul`, `jnp.einsum`),
shape manipulations (`jnp.reshape`, `jnp.transpose`, `jnp.squeeze`,
`jnp.expand_dims`), and higher-level neural network layers (e.g., in Flax or
Equinox) that expect specific input ranks for batch or feature dimensions.

- **Broadcasting vs. Stacking**: In `pmap`, "unmapped" inputs (marked with
  `None` in `in_axes`) were implicitly replicated. In `shard_map`, you specify
  this via `jax.P()`. The mapped function in `shard_map` sees the _full_
  replicated shape of these inputs, just like `pmap` did.

#### Rewriting `pmap` to `jit(shard_map)`

Once you have made any necessary rank adjustments, you can rewrite your
`jax.pmap` calls as `jax.jit(jax.shard_map(...))`. This transition involves a
few key components that differ from the implicit world of `pmap`:

- **`Mesh`**: Unlike `pmap` which assumes a linear arrangement of devices,
  `shard_map` requires an explicit `Mesh` object to define your device topology
  and axis names.
- **`in_specs` and `out_specs`**: These replace `in_axes` and `out_axes`.
  Instead of just specifying integer axes, you use `jax.P` (PartitionSpec) to
  explicitly map array dimensions to named mesh axes. This gives you precise
  control over how data is sliced (tiled) for inputs and assembled for outputs.
- **`jax.jit` wrapper**: While `pmap` is itself a compiled transform,
  `shard_map` is often used as a building block. Wrapping it in `jax.jit` is
  required to trigger the SPMD (Single Program Multiple Data) lowering and
  compilation that enables efficient parallel execution across the mesh.

Below are a number of examples of how to rewrite `jax.pmap` using
`jax.jit(jax.shard_map(...))` after first defining a `Mesh` object.

```python
from functools import partial
import jax
from jax.sharding import Mesh

# Define device topology: 8 devices logically arranged as a 1D vector named 'i'.
# This serves as the global context for axis names, similar to 'axis_name' in
# pmap.
mesh = jax.make_mesh(shape=(8,), axis_names=('i',))
```

**Basic Map**

```python
# pmap style: rank-reducing
# x_global: f32[8, 128]
@jax.pmap
def f(x):
  # x: f32[128]
  return x * 2
# output: f32[8, 128]

# shard_map style: rank-preserving
# x_global: f32[8, 128]
@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=jax.P('i'), out_specs=jax.P('i'))
def f(x):
  # x: f32[1, 128] (if logically x_global was (8, 128) and mesh size is 8)
  return x * 2
# output: f32[8, 128]
```

**Unmapped axes and replicated outputs**

```python
# pmap style
# x: f32[8, 128], y: f32[128]
@partial(jax.pmap, in_axes=(0, None), out_axes=None)
def f(x, y):
  # x: f32[128], y: f32[128]
  return x + y
# output: f32[128] (replicated)

# shard_map style
# x_global: f32[8, 128], y_replicated: f32[128]
@jax.jit
@partial(
    jax.shard_map, mesh=mesh, in_specs=(jax.P('i'), jax.P()), out_specs=jax.P()
)
def f(x, y):
  # x: f32[1, 128], y: f32[128]
  return x + y
# output: f32[128] (replicated)
```

**Multiple axes of parallelism**

```python
# Analogy to pmap(pmap(f, 'i'), 'j')
# mesh2d: 4 devices for 'i', 2 devices for 'j'
mesh2d = jax.make_mesh(shape=(4, 2), axis_names=('i', 'j'))

# nested pmap
# x: f32[4, 2, 128]
@partial(jax.pmap, axis_name='i')
@partial(jax.pmap, axis_name='j')
def f(x):
  # x: f32[128]
  return jax.lax.psum(x, ('i', 'j'))
# output: f32[4, 2, 128] (if out_axes=0)

# shard_map
# x_global: f32[4, 2, 128]
@jax.jit
@partial(
    jax.shard_map, mesh=mesh2d, in_specs=jax.P('i', 'j'), out_specs=jax.P()
)
def f(x):
  # x: f32[1, 1, 128]
  return jax.lax.psum(x, ('i', 'j'))
# output: f32[128] (replicated)
```

**Buffer donation**

```python
# pmap style
# donate_argnums specifies which inputs can be overwritten in-place
f = jax.pmap(func, donate_argnums=(0,))

# shard_map style: donate_argnums goes on the jit wrapper
# The underlying shard_map itself just handles the sharding layout
f = jax.jit(jax.shard_map(func, mesh=mesh, ...), donate_argnums=(0,))
```

#### Collectives

Collective operations like `jax.lax.psum` still use
`axis_name`, but they now operate over named mesh axes defined in your `Mesh`
object. Note that in `shard_map`, you must choose an `out_specs` that is
consistent with your collective (e.g., if you `psum` over `'i'`, an
`out_specs` of `jax.P()` implies you want a replicated result).

### 2. Input data preparation

Preparing data for `jax.jit(jax.shard_map)` requires a shift in how you think
about data distribution. While `jax.pmap` often handled sharding implicitly
based on array shapes and `in_axes`, `shard_map` asks you to be explicit about
how global data is sliced and placed across your device mesh. This means you
must directly provide arrays with a `sharding` that matches the `mesh` and
`in_specs` of your `shard_map` call; unlike `pmap`, `shard_map` will not
implicitly reshard inputs and will instead raise a **hard error** (e.g.,
[`ValueError: Received incompatible
devices`](#valueerror-received-incompatible-devices-)).
This involves new considerations for data locality, sharding layouts, and
multi-host orchestration.

#### Host-local vs. Global Views

Migration often starts with how you currently load data.

- **Host-local Array**: An array stored only on the devices attached to the
  current process. This is the standard `pmap` pattern where each host
  independently loads a subset of the dataset (e.g., using
  `jax.process_index()` to calculate an offset).
- **Global Array**: The entire logical dataset across all devices in the `Mesh`.
  `shard_map` (via `jax.jit`) expects this global view.

#### Addressability and Topology

The relationship between these views depends on your hardware setup.

- **Single-host**: All devices are connected to one process. A "global"
  array and a "fully addressable" array are effectively the same thing because
  the process can "see" every shard.
- **Multi-host**: Devices are spread across multiple processes (e.g., a
  TPU Pod). Each process only "sees" its local devices.
- **Fully Addressable**: A global array is **fully addressable** if the current
  process can access all of its shards. In multi-host settings, global arrays
  are typically **not fully addressable**; each process only sees the
  "host-local" part. You can query this state using the
  `x.is_fully_addressable` property.

#### Shardings

You define how global arrays are distributed across devices using
`jax.NamedSharding`. When using `shard_map`, it is critical that the **input
array's sharding explicitly matches** the `mesh` and `in_specs` you pass to the
`shard_map` call. If the physical distribution of your data does not align with
the logical distribution expected by `shard_map`, JAX will have to reshard the
data (potentially involving expensive communication) before the parallel
computation can begin.

- **NamedSharding vs. PmapSharding**:
  - `PmapSharding` is the legacy internal representation for `pmap`. It is
    inherently **rank-reducing** and tied to the implicit device axis of
    `pmap`.
  - `NamedSharding` is the modern, flexible representation used with `jit`
    and `shard_map`. It is **rank-preserving** and uses a `Mesh` and
    `PartitionSpec` to logically map array dimensions to device axes.
- **SingleDeviceSharding**: While `shard_map` is about distributed data,
  `jax.SingleDeviceSharding` remains a core part of the system. It is used
  for arrays that live entirely on one device, such as host-local data or the
  results of unshared computations.

#### The Migration Pattern: "Stitching"

In `pmap`, JAX implicitly handled the split across hosts. With `shard_map`, you
must be explicit. The standard pattern is to load **host-local** data (just as
you did for `pmap`) and then use
`jax.make_array_from_process_local_data` to "stitch" that local data into a
single global (but partially addressable) `jax.Array` before passing it to your
sharded computation.

```python
import jax
import jax.numpy as jnp
import numpy as np

# 1. Define your mesh and sharding (logical view)
mesh = jax.make_mesh((jax.process_count(),), ('batch',))
sharding = jax.NamedSharding(mesh, jax.P('batch'))

# 2. Load host-local data (as you would for pmap)
# Example: each process loads a different subset of a dataset
local_batch_size = 32
start_idx = jax.process_index() * local_batch_size
local_data = np.arange(start_idx, start_idx + local_batch_size).reshape(
    local_batch_size, 10
)

# 3. Stitch into a global jax.Array
# The resulting array will have global shape (32 * num_processes, 10)
global_batch = jax.make_array_from_process_local_data(sharding, local_data)

print(f"Process {jax.process_index()} local shape: {local_data.shape}")
print(f"Global array shape: {global_batch.shape}")
```

> [!NOTE]
> `jax.make_array_from_process_local_data` requires that the `local_data` shape
> on each process matches the expected shard size derived from the `sharding`.

### 3. Output consumption

While `pmap` returns a value that is often treated as a stack of per-device
outputs (sometimes requiring a `concatenate` to use as a single array),
`shard_map` returns a single `jax.Array`.

#### Global View

The output is already a single logical array sharded across
devices. You can immediately perform global operations on it (like
`jnp.mean(output)`) within a `jax.jit` context.

#### The `unreplicate` Anti-pattern

As described in [Appendix A](#appendix-a), there is a common pattern where
arrays are **physically sharded** across devices despite being **logically
replicated** (i.e., every shard contains the same data).

In the legacy `pmap` implementation, users would frequently call
`flax.jax_utils.unreplicate(output)` (equivalent to `output[0]`) to retrieve
what they assumed was a cheap local replica.

- **The issue**: JAX does not track semantic replication for sharded arrays.
  When you call `x[0]` on an array sharded along its leading axis, JAX must
  assume the first shard contains unique data that needs to be broadcast to the
  entire mesh to satisfy indexing semantics. This triggers a **global gather**,
  causing significant performance regressions.
- **Recommendation**: Avoid creating physically sharded replicas. If you must
  work with them, use `x.addressable_shards[0].data` to access the local replica
  without triggering communication. See [Appendix A](#appendix-a) for a detailed
  technical breakdown.

#### Host access

To get the data back to the host process, you use standard
JAX patterns like `device_get` or simple indexing.

### Related documentation

To help with migration, we recommend reviewing the following documentation based
on your needs:

- **{doc}`sharded-computation`**: Start here for a high-level introduction to
  parallel programming in JAX. This tutorial covers all three sharding modes
  (automatic, explicit, and manual) with a comparison table, explains key
  concepts like data sharding and `NamedSharding`, and demonstrates how each
  mode handles a simple neural network layer. This is the best starting point
  for understanding the overall landscape of parallelism in JAX.

- **{doc}`notebooks/Distributed_arrays_and_automatic_parallelization`**: Read
  this for a deeper understanding of `jax.Array` and automatic parallelization
  via `jax.jit`. This notebook explains how sharded data works, how computation
  follows data placement, and how to use `jax.lax.with_sharding_constraint` to
  guide the compiler. It includes practical neural network examples with batch
  data parallelism and model tensor parallelism.

- **{doc}`notebooks/shard_map`**: This is the comprehensive guide for manual
  parallelism with `jax.shard_map`. It explains the difference between
  rank-reducing maps (like `vmap`) and rank-preserving maps (like `shard_map`),
  how to control input splitting and output assembly with `in_specs` and
  `out_specs`, and includes a detailed collectives tutorial covering `psum`,
  `all_gather`, `psum_scatter`, and more. If you're migrating complex `pmap`
  code with explicit collectives, this is essential reading.

- **{doc}`notebooks/explicit-sharding`**: Explore this for the newest sharding
  mode where sharding becomes part of the JAX-level type system. With explicit
  sharding, sharding propagation happens at trace time and shardings are
  queryable via `jax.typeof(x)`. This mode provides more control than automatic
  sharding while still using a global-view programming model. It's particularly
  useful when you want deterministic sharding behavior without resorting to
  fully manual parallelism.

- **{doc}`jep/14273-shard-map`**: Read the original design document for
  `shard_map`. This JEP (JAX Enhancement Proposal) provides the technical
  rationale for the API, detailed comparisons with `pmap` and `xmap`, and
  explains the fundamental concepts of rank-reducing vs. rank-preserving maps
  over array axes.

(appendix-a)=

## Appendix A: More details about `int` indexing into sharded arrays.

### What should `x[0]` return?

In **NumPy**, `x[0]` returns a rank-reduced array representing the first slice
along the first dimension. For example, if `x = np.ones((8, 3, 4))`, then `x[0]`
returns an array of shape `(3, 4)`.

In **JAX** (`jax.numpy`), `x[0]` semantically works the same way: it returns the
rank-reduced slice of the logical array `x`. However, performance depends on how
`x` is sharded or replicated across devices. Consider an array `x` with shape
`(8, 3, 4)` distributed across 8 devices (using `jax.P`):

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
    If `x` is sharded along the first dimension, `x[0]` results in shape
    `(3, 4)` with partition spec `jax.P(None, None)`.
    - **The Issue:** Because the first dimension is sharded, the data for
      `x[0]` physically resides _only_ on the first device. To satisfy the
      output sharding `jax.P(None, None)` (which implies replication), JAX
      must broadcast the data from the first device to all other devices. This
      requires **communication**; JAX will gather the _entire_ array of shape
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
device (it is **logically replicated**), JAX sees an array with
`shape (8, 3, 4)` and spec `jax.P('x', None, None)`—that is, the data is
**physically sharded** along its leading dimension.

**JAX does not track semantic replication.** It does not "know" that the shard
on device 1 is identical to the shard on device 0. Therefore, when you call
`x[0]`, JAX must satisfy the strict semantics of array indexing: it must
retrieve the first slice and, because the output is typically expected to be
available for subsequent JIT-ted operations, it must often ensure that result
is replicated across the mesh.

This triggers a **global gather (or broadcast)** of the entire array to all
devices before slicing. What the user assumes is a constant-time "ignore the
extra copies" operation actually becomes a serialized communication bottleneck
(visible as `_gather` operations in a stack trace).

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
_logical_ global array. In the multi-host setting, we will see a more drastic
version of the performance issues described above as all the hosts gather the
entire array to each device before slicing. In certain cases, users can even
face hard errors (e.g., `INVALID_ARGUMENT: CopyArrays only support...`).

In multi-host settings (e.g., 4 hosts × 2 devices = 8 devices total):

1. A global array with shape `(8, ...)` and `jax.P('x')` has each slice
   distributed across all 8 devices spanning all hosts.

2. When you call `x[0]`, JAX needs to slice the first element and reshard the
   result so it's available to all hosts.

3. The `CopyArrays` operation in XLA requires source and destination to have the
   same device count. But each host only sees its _local_ subset of devices (2
   in this example), not all 8. When JAX tries to create a resharded array, the
   device list mismatch triggers the error.

<!--* freshness: { reviewed: '2026-01-09' } *-->
