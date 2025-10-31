# Pallas Core-specifc Programming

In this guide, we explore using `pl.core_map` to write Pallas kernels. Compared with `pallas_call`, `core_map` offers a few key characteristics:

* **Per-core level programming**: You write code for an TPU/GPU core, not for a JAX device. This is crucial if you want to specifically control a core, or how cores communicate and distribute work among one another.

* **Flexible pipelining**: You have the option to write pipelining communications on your own, instead of relying on Pallas grids and specs. This is helpful if your pipeline diverges from the standard "copy-in, compute & copy-out" pattern.

* **Collectives**: Since `core_map` allows inter-core communications, it is especially helpful when writing collectives on the core level.

This guide focuses on TPU. For how to use `core_map` on GPU to achieve higher thread flexibility, check out our [Pallas GPU `core_map` tutorial](https://docs.jax.dev/en/latest/pallas/gpu/reference.html#using-core-map).

## Environment setup

Modern accelerators often have multiple cores under a device. For TPU chips higher than v4, every JAX device by default contains two TensorCores (aka. a [Megacore](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#chips)). They also contain a [SparseCore](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#sparsecore), consisting of many subcores.

This guide was written on a v5p chip, which contains 4 devices (2 TensorCores each) and a SparseCore of 16 subcores.


```python
from functools import partial

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np


num_devices = jax.local_device_count()
assert num_devices > 1, "Please run this notebook with more than one device."
assert "TPU" in jax.devices()[0].device_kind, "Please run this notebook with TPU devices."
print(f"Running with {num_devices} {jax.devices()[0].device_kind} devices.")
```

    Running with 4 TPU v5 devices.


In addition to the typical TPU device mesh, you need to make a mesh of cores. Consider this as an addition dimension called "core", with length 2, in addition to the 4-device mesh you work with. That is 8 cores in total.


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

    Mesh('device': 4, axis_types=(Auto,))
    TensorCoreMesh(devices=array([TensorCore(id=0), TensorCore(id=1)], dtype=object), axis_names=('core',))
    There are 4 devices, and 2 cores each.


## A simple per-core kernel

`pl.core_map` allows you to write per-core local code, just as `jax.shard_map` allows you to write per-device code.

In the example kernel below, each core has its own VMEM and semaphore allocations. As with normal kernel, you can initiate copy between HBM and VMEM refs using `async_copy`.

**Communication amongst cores**

Before making a inter-core communication, you may need to do a global barrier signal (`pltpu.semaphore_signal`), to make sure all the destination semaphores have been properly initialized.

After that, use `pltpu.make_async_remote_copy` to send the actual data. The `device_id` allows you to specify the destination using only the axis coordinate(s) that are different from the source.



```python
# This runs on every core
def swap_cores_kernel(in_hbm, out_hbm,
                      in_vmem, scratch_vmem, out_vmem,
                      sem, send_sem, recv_sem):
  core_index = jax.lax.axis_index('core')
  num_cores = jax.lax.axis_size('core')
  slc_size = in_hbm.shape[-1] // num_cores
  slc = pl.ds(core_index * slc_size, slc_size)

  # A barrier to make sure all cores have entered run_scoped.
  # You won't need this if not doing inter-core communications.
  sem0 = pltpu.get_barrier_semaphore()
  for i in range(num_devices):
    for j in range(num_cores):
      pltpu.semaphore_signal(sem0, 1, device_id={'device': i, 'core': j})
  pltpu.semaphore_wait(sem0, num_devices * num_cores)

  # Copy in the input
  pltpu.async_copy(in_hbm.at[:, slc], in_vmem, sem).wait()

  # Swap data between core 0 and core 1
  dst_core = (core_index + 1) % num_cores
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

 * Wrap it with `pl.run_scoped` and HBM refs, so that the required scratch spaces (VMEM and semaphores) are allocated, one copy per core.

 * Call it inside a `pl.core_map`, which takes the TensorCore mesh.

    * You would need `collective_id` if there exists inter-core communications.

 * Allocate the output buffer and pass in the references of the input and output.


```python
input_shape = (32, 256)
local_vmem_shape = (32 // num_devices, 256 // num_cores)
input_pspec = P('device', None)
sharding = NamedSharding(mesh, input_pspec)

@jax.jit
@partial(jax.shard_map, mesh=mesh, in_specs=input_pspec, out_specs=input_pspec,
         check_vma=False)
def swap_cores(x):
  # Get buffers out of the input and output
  x_hbm_ref, o_hbm_ref = jax.tree.map(jax.new_ref, (x, jax.lax.empty(x.shape, x.dtype)))

  @pl.core_map(tc_mesh, compiler_params=pltpu.CompilerParams(collective_id=0))
  def _():
    pl.run_scoped(
        partial(swap_cores_kernel, x_hbm_ref, o_hbm_ref),
        *([pltpu.VMEM(local_vmem_shape, x.dtype)] * 3),  # VMEM allocations
        *([pltpu.SemaphoreType.DMA] * 3),          # semaphores
    )
  return o_hbm_ref[...]


x = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
x = jax.device_put(x, sharding)
y = swap_cores(x)

np.testing.assert_array_equal(y[:, 128:], x[:, :128] * 2)
np.testing.assert_array_equal(y[:, :128], x[:, 128:] * 2)
```

### Save the boilerplate

You could make a shortcut `kernel()` that wraps all the `shard_map`, `core_map` and `run_scoped` boilerplates.

Some similar APIs are currently available in Pallas package, such as `plgpu.kernel` and `plsc.kernel`. A unified API may be released soon.


```python
def kernel(kernel_body, core_mesh, scratch_shapes,
           in_specs, out_specs, out_shape=None, out_dtype=None,
           compiler_params=None):
  @jax.jit
  @partial(jax.shard_map, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
  def run(x):
    y = jax.lax.empty(out_shape if out_shape else x.shape,
                      out_dtype if out_dtype else x.dtype)
    def pl_kernel(hbm_refs):
      @pl.core_map(core_mesh, compiler_params=compiler_params)
      def _():
        pl.run_scoped(partial(kernel_body, *hbm_refs), *scratch_shapes)
    _, y = pl.run_state(pl_kernel)((x, y))
    return y
  return run

scratch_shapes = [pltpu.VMEM(local_vmem_shape, x.dtype)] * 3 + [pltpu.SemaphoreType.DMA] * 3
y = kernel(swap_cores_kernel, tc_mesh, scratch_shapes, input_pspec, input_pspec,
           compiler_params=pltpu.CompilerParams(collective_id=0))(x)

np.testing.assert_array_equal(y[:, 128:], x[:, :128] * 2)
np.testing.assert_array_equal(y[:, :128], x[:, 128:] * 2)
```

## Pipelining with `core_map`

Note that the kernel above only does simple copies and computes, without automatic pipelining via Pallas `grid` and `BlockSpec`. To do pipelining inside `core_map`, use `pltpu.emit_pipeline` inside the core-local kernel.

**Parallelize work per core**

Since you are programming on the core level, you get to customize exactly how the work is splitted amongst cores. To do that, you need to:

1. Provide an `index_map` function that, given the iteration indices, return *the slice* of the input data that shall be passed in.

1. On `BlockSpec`, wrap the corresponding dimension with `pl.BoundedSlice`, indicating the `index_map` function would return a slice instead of a iteration index on that dimension.

This manual approach gives you the full control to work splitting, in contrast to `pl.pallas_call`, [which automatically parallelizes an array axis over cores under the hood](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpus-in-megacore-configuration).

**Scratch shapes allocation**

Note that in the example below, the top level `pl.run_scoped` (wrapped inside `kernel`) did not allocate any scratch buffer. Instead, the VMEM scratch shapes were allocated within `pltpu.emit_pipeline`, which called another `pl.run_scoped` within.



```python
def add_one_body(in_vmem, out_vmem):
  out_vmem[...] = in_vmem[...] + 1

input_shape = (1024, 1024)
input_pspec = P('device', None)

def add_one_kernel(x_hbm_ref, o_hbm_ref):
  core_idx = jax.lax.axis_index('core')
  in_shape = x_hbm_ref.shape
  core_slc_size = in_shape[0] // num_cores  # The slice that this core will work on
  index_map = lambda i, j: (
      pl.ds(pl.multiple_of(core_idx * core_slc_size + i * 8, 8), 8), j)

  pltpu.emit_pipeline(
      add_one_body,
      grid=(core_slc_size // 8, in_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128), index_map=index_map,
      )]
  )(x_hbm_ref, o_hbm_ref)


x = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
x = jax.device_put(x, NamedSharding(mesh, input_pspec))
y = kernel(add_one_kernel, tc_mesh, [], input_pspec, input_pspec)(x)

np.testing.assert_array_equal(y, x + 1)
```

## Scalar prefetch

The code below extended the kernel above but uses [scalar prefetch and dynamic block indexing](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html) to select a specific sub-slice of the input.

This involves pre-allocating an SMEM buffer (via the `pl.run_scoped` call inside `kernel`) and fill the buffer using a `sync_copy` before the pipeline starts. The dynamic index value is then being closed-over inside the `index_map` API.


```python
input_shape = (1024, 1024)
input_pspec = P('device', None)
output_shape = (1024, 512)

def indexed_add_one_kernel(in_refs, out_refs, i_smem_ref):
  (x_hbm_ref, i_hbm_ref), o_hbm_ref = in_refs, out_refs
  in_shape = x_hbm_ref.shape
  pltpu.sync_copy(i_hbm_ref, i_smem_ref)

  core_idx = jax.lax.axis_index('core')
  core_slc_size = in_shape[0] // num_cores
  i_map = lambda i: pl.ds(pl.multiple_of(core_idx * core_slc_size + i * 8, 8), 8)
  j_map = lambda j: pl.ds(pl.multiple_of((i_smem_ref[0] + j * 128), 128), 128)

  pltpu.emit_pipeline(
      add_one_body,
      grid=(core_slc_size // 8, output_shape[1] // 128),
      in_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), pl.BoundedSlice(128)),
          index_map=lambda i, j: (i_map(i), j_map(j)),
      )],
      out_specs=[pl.BlockSpec(
          block_shape=(pl.BoundedSlice(8), 128),
          index_map=lambda i, j: (i_map(i), j),
      )]
  )(x_hbm_ref, o_hbm_ref)


xs = jax.random.normal(jax.random.key(0), input_shape, jnp.float32)
xs = jax.device_put(xs, NamedSharding(mesh, input_pspec))
idx = 256
y = kernel(indexed_add_one_kernel, tc_mesh, [pltpu.SMEM((1,), jnp.int32)],
            in_specs=((input_pspec, P()),), out_specs=input_pspec,
            out_shape=(input_shape[0] // num_devices, input_shape[1] // 2),
            out_dtype=jnp.float32,
           )((xs, jnp.array([idx])),)

np.testing.assert_array_equal(y, xs[:, idx:(idx+512)] + 1)
```

## Mapping over SparseCores

TPU v4 and above includes a [SparseCore](https://openxla.org/xla/sparsecore), which is specialized in sparse memory access and operations. This guide will not dive into the capabilities of SparseCore, but rather show how to run a program on SparseCore with same semantics and minimal changes from the TensorCore code.

Start with knowing the basic SparseCore specs of your chip, and create a `VectorSubcoreMesh` for vector operations. Note that each SparseCore has 16 (or other number) subcores, and `core_map` will map your code on each of them.


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


The code below is very similar from the `add_one_kernel` we wrote earlier, except for a few differences:

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
      out_vmem.at[*slc][...] = in_vmem.at[*slc][...] + 1


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


x = jax.random.randint(jax.random.key(0), input_shape, 0, 64, jnp.int32)
x = jax.device_put(x, NamedSharding(mesh, input_pspec))
y = kernel(sc_add_one_kernel, sc_mesh, [], input_pspec, input_pspec)(x)

np.testing.assert_array_equal(y, x + 1)
```
