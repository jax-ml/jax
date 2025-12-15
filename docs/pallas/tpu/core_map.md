# Pallas Core-specific Programming

In this guide, we explore using `pl.core_map` to write Pallas kernels. Compared with `pallas_call`, `core_map` offers a few key characteristics:

* **Per-core level programming**: You write code for an TPU/GPU core, not for a JAX device. This gives you full control over what runs on every core, or how cores communicate and distribute work among one another.

* **Collectives**: `core_map` explicitly models physical cores, so inter-core communication can be expressed safely.

* **Platform generic**: `core_map` programming model works for TPU (TensorCore and SparseCore) and GPU with minimal boilerplate changes.

This guide focuses on TPU. For how to use `core_map` on GPU to achieve higher thread flexibility, check out our [Pallas GPU `core_map` tutorial](https://docs.jax.dev/en/latest/pallas/gpu/reference.html#using-core-map).

## Environment setup

Modern accelerators often have multiple cores under a device. For recent TPU chips (v4, v5p), every JAX device may contains 2 TensorCores (aka. a [Megacore](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#chips)). Some TPUs (v5p, v6e, 7x) also contain [SparseCores](https://openxla.org/xla/sparsecore#specifications_at_a_glance), each of which consists of many subcores.

This guide was written on a v5p chip, which contains 4 devices (2 TensorCores each) and 4 SparseCores, each with 16 subcores.


```python
from functools import partial

import jax
from jax.sharding import NamedSharding
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


num_devices = jax.local_device_count()
assert num_devices > 1, "Please run this notebook with more than one device."

tpu_info = pltpu.get_tpu_info()  # This notebook only runs on TPU.
print(f"Running on {num_devices} TPU {tpu_info.chip_version} devices.")
```

    Running on 4 TPU v5p devices.


In addition to the typical TPU device mesh, you need to make a mesh of cores. Consider this as an addition dimension called `core`, with length 2, in addition to the 4-device mesh you work with. That is 8 cores in total.


```python
# Mesh of devices
mesh = jax.make_mesh((jax.device_count(),), ('device',))
print(mesh)

# Mesh of cores, within a JAX device
tc_mesh = pltpu.create_tensorcore_mesh('core')
print(tc_mesh)

num_devices = mesh.size
num_cores = len(tc_mesh.devices)
print(f"There are {num_devices} devices, and {num_cores} cores each.")
```

    Mesh('device': 4, axis_types=(Explicit,))
    TensorCoreMesh(devices=array([TensorCore(id=0), TensorCore(id=1)], dtype=object), axis_names=('core',))
    There are 4 devices, and 2 cores each.


## A simple per-core kernel

`pl.core_map` allows you to write per-core local code, just as `jax.shard_map` allows you to write per-device code.

In the example kernel below, each core has its own VMEM and semaphore allocations. As with normal kernel, you can initiate copies between HBM and VMEM refs using `pltpu.async_copy`.

**Communication between cores**

Before communicating between cores, it is good practice to perform a barrier (using `pltpu.semaphore_signal`) to ensure resources have been allocated and both cores are at the same point during the program.

Once the cores are synchronized, use `pltpu.make_async_remote_copy` to send data between them. The `device_id` keyword argument generically allows sending to any core on any device, but if you just pass in `{'core': other_core_id}`, it will perform a intra-device inter-core copy (the other axis names are held constant).



```python
# This runs on every core
def swap_cores_kernel(in_hbm, out_hbm,
                      in_vmem, scratch_vmem, out_vmem,
                      sem, send_sem, recv_sem):
  core_index = jax.lax.axis_index('core')
  num_cores = jax.lax.axis_size('core')
  slc_size = in_hbm.shape[-1] // num_cores
  slc = pl.ds(core_index * slc_size, slc_size)

  # Copy in a core-dependent slice of the input
  pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()

  # A barrier to make sure all cores have entered run_scoped.
  # You won't need this if not doing inter-core communications.
  dst_core = (core_index + 1) % num_cores
  sem0 = pltpu.get_barrier_semaphore()
  pltpu.semaphore_signal(sem0, 1, device_id={'core': dst_core})
  pltpu.semaphore_wait(sem0, 1)

  # Swap data between core 0 and core 1
  the_copy = pltpu.make_async_remote_copy(
      in_vmem, scratch_vmem, send_sem, recv_sem, device_id={'core': dst_core},
  )
  the_copy.start()
  the_copy.wait()

  # Core-local compute
  out_vmem[...] = scratch_vmem[...] * 2

  # Copy out the output
  pltpu.async_copy(out_vmem, out_hbm.at[:, slc], sem).wait()

```

Once you have the local kernel:

 * Start your top-level JAX code with HBM refs, and allocate output refs if needed.

 * Use `pl.core_map`, which takes the TensorCore mesh, to start per-core programming.

    * You will need `collective_id` for the barrier semaphore.

 * Inside `pl.core_map`, invoke `pl.run_scoped` to allocate per-core scratch spaces (VMEM and semaphores) and run the local kernel.


```python
input_shape = (32, 256)
local_vmem_shape = (32 // num_devices, 256 // num_cores)
in_spec = jax.P('device', None)
sharding = NamedSharding(mesh, in_spec)

@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec,
         check_vma=False)
def swap_cores(x):
  # Get buffers out of the input and output
  x_hbm_ref = jax.new_ref(x)
  o_hbm_ref = jax.new_ref(jax.lax.empty(x.shape, x.dtype))

  @pl.core_map(tc_mesh, compiler_params=pltpu.CompilerParams(collective_id=0))
  def _():
    pl.run_scoped(
        partial(swap_cores_kernel, x_hbm_ref, o_hbm_ref),
        *([pltpu.VMEM(local_vmem_shape, x.dtype)] * 3),  # VMEM allocations
        *([pltpu.SemaphoreType.DMA] * 3),                # semaphores
    )
  return o_hbm_ref[...]


x = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
x = jax.device_put(x, sharding)
y = swap_cores(x)

np.testing.assert_array_equal(y[:, 128:], x[:, :128] * 2)
np.testing.assert_array_equal(y[:, :128], x[:, 128:] * 2)
```

### Save the boilerplate

You can use the `pl.kernel` decorator to wrap boilerplate such as `core_map`, `run_scoped`, and output buffer allocation.

Note that this should run inside any `jax.shard_map` you may have at the top level.


```python
@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def swap_cores(x):
  scratch_shapes = [pltpu.VMEM(local_vmem_shape, x.dtype)] * 3 + [pltpu.SemaphoreType.DMA] * 3
  return pl.kernel(swap_cores_kernel, out_shape=x, mesh=tc_mesh,
                   scratch_shapes=scratch_shapes,
                   compiler_params=pltpu.CompilerParams(collective_id=0))(x)

y = swap_cores(x)
np.testing.assert_array_equal(y[:, 128:], x[:, :128] * 2)
np.testing.assert_array_equal(y[:, :128], x[:, 128:] * 2)
```

## Pipelining with `core_map`

Note that the kernel above only does simple copies and compute, without automatic pipelining via Pallas `grid` and `BlockSpec`. To do pipelining inside `core_map`, use `pltpu.emit_pipeline` inside the core-local kernel.

**Automatically parallelize work amongst cores**

The simple way is to annotate a block axis as `pltpu.PARALLEL`, and Pallas will automatically parallelize work along this axis. Both `pl.pallas_call` and `pltpu.emit_pipeline` supports this, via arguments `core_axis` and `dimension_semantics`. The `pallas_call` example is [in another guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpus-in-megacore-configuration), and the `emit_pipeline` case is shown below.

When the `PARALLEL` annotation is provided, the corresponding grid dimension will be logically split and executed on separate cores. (The exact semantics of which grid dimensions are executed on which core is guaranteed).

**Scratch shapes allocation**

Note that in the example below, the top level `pl.run_scoped` (wrapped inside `kernel`) did not allocate any VMEM scratch buffers. Instead, `pltpu.emit_pipeline` allocates its own scratch buffers in VMEM and use them for its multiple buffering.



```python
def add_one_body(in_vmem, out_vmem):
  out_vmem[...] = in_vmem[...] + 1

input_shape = (1024, 1024)
in_spec = jax.P('device', None)

def add_one_kernel(x_hbm_ref, o_hbm_ref):
  in_shape = x_hbm_ref.shape
  pltpu.emit_pipeline(
      add_one_body,
      grid=(in_shape[0] // 8, in_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i, j),
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i, j),
      )],
      core_axis_name='core',
      dimension_semantics=(pltpu.PARALLEL, pltpu.ARBITRARY),
  )(x_hbm_ref, o_hbm_ref)


@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def add_one(x):
  return pl.kernel(add_one_kernel, out_shape=x, mesh=tc_mesh, scratch_shapes=[])(x)


x = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
x = jax.device_put(x, NamedSharding(mesh, in_spec))
y = add_one(x)

np.testing.assert_array_equal(y, x + 1)
```

## Scalar prefetch

The code below extends the kernel above but uses [scalar prefetch and dynamic block indexing](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html) to select a specific sub-slice of the input.

This involves pre-allocating an SMEM buffer (via the `pl.run_scoped` call inside `kernel`) and populating the buffer using a `sync_copy` before the pipeline starts. Close over the dynamic index value inside the `index_map` to use it.

**Manually delegate work amongst cores**

The code example below also shows how `core_map` allows you to customize exactly how the work is split between cores, without relying on the automatic API shown above.

To achieve that, customize your `index_map` to use the core index to work on different slices on different cores.



```python
input_shape = (1024, 1024)
in_spec = jax.P('device', None)
output_shape = (1024, 512)

def indexed_add_one_kernel(in_refs, out_refs, i_smem_ref):
  (x_hbm_ref, i_hbm_ref), o_hbm_ref = in_refs, out_refs
  in_shape = x_hbm_ref.shape
  pltpu.sync_copy(i_hbm_ref, i_smem_ref)

  core_idx = jax.lax.axis_index('core')
  core_slc_size = in_shape[0] // num_cores
  i_map = lambda i: core_idx * core_slc_size // 8 + i  # split work among cores
  j_map = lambda j: i_smem_ref[0] // 128 + j           # use the prefetched offset

  pltpu.emit_pipeline(
      add_one_body,
      grid=(core_slc_size // 8, output_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i_map(i), j_map(j)),
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(8, 128), index_map=lambda i, j: (i_map(i), j),
      )]
  )(x_hbm_ref, o_hbm_ref)


@jax.jit
@partial(jax.shard_map, mesh=mesh,
         in_specs=(in_spec, jax.P()), out_specs=in_spec, check_vma=False)
def indexed_add_one(x, index):
  out_shape = jax.ShapeDtypeStruct((x.shape[0], x.shape[1] // 2), x.dtype)
  return pl.kernel(indexed_add_one_kernel,
                   out_shape=out_shape, mesh=tc_mesh,
                   scratch_shapes=[pltpu.SMEM((1,), jnp.int32)])((x, index))


xs = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
xs = jax.device_put(xs, NamedSharding(mesh, in_spec))
idx = 256
y = indexed_add_one(xs, jnp.array([idx]))

np.testing.assert_array_equal(y, xs[:, idx:(idx+512)] + 1)
```

## Mapping over SparseCores

TPU v5p contains 4 [SparseCores](https://openxla.org/xla/sparsecore), which are specialized for sparse memory access and operations. This guide will not dive into the full capabilities of SparseCore, but rather show how to run a program on SparseCore with the same semantics and minimal changes from the TensorCore code.

Start with knowing the basic SparseCore specs of your chip, and create a `VectorSubcoreMesh` for vector operations. Note that each SparseCore has 16 (or other number) subcores on TPU v5p, and `core_map` will run your code SPMD on each of them.


```python
sc_info = pltpu.get_tpu_info().sparse_core
assert sc_info is not None
print(sc_info)

sc_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core", subcore_axis_name="subcore",
    num_cores=sc_info.num_cores
)
sc_num_cores = sc_info.num_cores
sc_num_subcores = sc_info.num_subcores
```

    SparseCoreInfo(num_cores=4, num_subcores=16, num_lanes=8)


The code below is very similar to the `add_one_kernel` we wrote earlier, except for a few differences:

1. You need to split the work amongst all subcores, so a few lines to compute the specific slice for each subcore.

1. SparseCore register computation allows smaller slices (`4x16` max for int32), so you need nested loops to iterate the slice during computation phase.


```python
input_shape = (4096, 128)
SC_REG_OP_SHAPE = (4, 16)

def sc_add_one_body(in_vmem, out_vmem):
  @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
  def _reg_loop_0(c0):
    @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
    def _reg_loop_1(c1):
      slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
      out_vmem[slc] = in_vmem[slc] + 1


def sc_add_one_kernel(x_hbm_ref, o_hbm_ref):
  in_shape = x_hbm_ref.shape
  core_idx = jax.lax.axis_index('core')
  subcore_idx = jax.lax.axis_index("subcore")
  cm_idx = core_idx * sc_num_subcores + subcore_idx  # index on the core_map
  slc_size = in_shape[0] // (sc_num_subcores * sc_num_cores)
  index_map = lambda i, j: (
      pl.ds(pl.multiple_of(cm_idx * slc_size + i * 8, 8), 8), j)

  pltpu.emit_pipeline(
      sc_add_one_body,
      grid=(slc_size // 8, in_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )]
  )(x_hbm_ref, o_hbm_ref)


@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=in_spec, out_specs=in_spec, check_vma=False)
def sc_add_one(x):
  return pl.kernel(sc_add_one_kernel, out_shape=x, mesh=sc_mesh, scratch_shapes=[])(x)


x = jax.random.randint(jax.random.key(0), input_shape, 0, 64, jnp.int32)
x = jax.device_put(x, NamedSharding(mesh, in_spec))
y = sc_add_one(x)

np.testing.assert_array_equal(y, x + 1)
```
