# GPU performance tips

<!--* freshness: { reviewed: '2025-03-20' } *-->

This document focuses on performance tips for neural network workloads

## Matmul precision

On recent GPU generations, such as the Nvidia A100 generation or later, it can
be a good idea to perform most computations in `bfloat16` precision. For
example, if using [Flax](https://github.com/google/flax), instantiate `Dense`
layers using `flax.linen.Dense(..., dtype=jax.numpy.bfloat16)`. Here are some
code examples:
* In the [Flax LM1B
  example](https://github.com/google/flax/tree/main/examples/lm1b), `Dense`
  modules are [instantiated with a configurable
  dtype](https://github.com/google/flax/blob/fd8fd76a4af5307a61f85bac98feab9b26d60db8/examples/lm1b/models.py#L188)
  which [defaults](https://github.com/google/flax/blob/fd8fd76a4af5307a61f85bac98feab9b26d60db8/examples/lm1b/configs/default.py#L112) to
  [bfloat16](https://github.com/google/flax/blob/c0087535d7f2e5bfcbf2a7be6825b9f5055a54c6/examples/lm1b/train.py#L431).
* In [MaxText](https://github.com/google/maxtext), `DenseGeneral` modules are
  also [instantiated with a configurable
  dtype](https://github.com/google/maxtext/blob/07dc6ce27ced1246407d0de311d4a0d6a9fd46d8/MaxText/layers.py#L592)
  that [defaults to
  bfloat16](https://github.com/google/maxtext/blob/07dc6ce27ced1246407d0de311d4a0d6a9fd46d8/MaxText/configs/base.yml#L41).

## XLA performance flags

```{note}
  JAX-Toolbox also has a page on [NVIDIA XLA performance FLAGS](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md).
```

The existence and exact behavior of XLA flags may be `jaxlib`-version dependent.

As of `jaxlib==0.4.18` (released [Oct 6
2023](https://pypi.org/project/jaxlib/#history)), setting these XLA flags can
improve performance. Some are related to communication between GPUs, and so are
only relevant when running computations on multiple devices, while others are
related to code generation on each device.

Some of these may be set by default in future releases.

These flags can be set via the `XLA_FLAGS` shell environment variable. For
example, we can add this to the top of a Python file:
```python
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
```

For more examples, see also [XLA Flags recommended for Pax
training on Nvidia GPUs](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/rosetta/projects/pax/README.md#xla-flags).


### Code generation flags

* **--xla_gpu_triton_gemm_any** Use the Triton-based GEMM (matmul) emitter for
  any GEMM that it supports. The default value is False.

## Communication tips

### Auto and manual PGLE

The Profile Guided Latency Estimator (PGLE) workflow measures the actual running time
of compute and collectives, the the profile information is fed back into XLA compiler
for a better scheduling decision.

The Profile Guided Latency Estimator can be used manually or automatically. In the auto mode
JAX will collect profile information and recompile a module in a single run. While
in manual mode you need to run a task twice, the first time to collect and save profiles
and the second to compile and run with provided data.

**Important**: the JAX profiler, which is used by both of the PGLE workflows documented
below, cannot co-exist with the NVIDIA Nsight Systems profiler. This limitation can be
avoided by using the JAX compilation cache, as described below.

### Auto PGLE
The auto PGLE can be turned on by setting the following environment variables:

Mandatory:
```bash
export JAX_ENABLE_PGLE=true

# For JAX version <= 0.5.0 make sure to include:
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
```

Optional:
```bash
export JAX_PGLE_PROFILING_RUNS=3
export JAX_PGLE_AGGREGATION_PERCENTILE=85

# Right now the auto PGLE profile collection doesn't work with command buffer.
# If the command buffer is enabled, Auto PGLE will disable it during profile
# collection and enable it back after the recompilation. If you need to have a
# consistent command buffer logic with and with PGLE profile you can disable it
# manually:
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_enable_command_buffer=''"
```

Or in the JAX this can be set as the following:

```
import jax
from jax._src import config

with config.enable_pgle(True), config.pgle_profiling_runs(1):
  # Run with the profiler collecting performance information.
  train_step()
  # Automatically re-compile with PGLE profile results
  train_step()
  ...
```

You can control amount of reruns used to collect profile data by changing `JAX_PGLE_PROFILING_RUNS`.
Increasing this parameter would lead to better profile information, but it will also increase the
amount of non-optimized training steps.

Decreasing the `JAX_PGLE_AGGREGATION_PERCENTILE` parameter might help in case when performance between steps is too noisy to filter out a non-relevant measures.

**Attention:** Auto PGLE doesn't work for pre-compiled modules. Since JAX need to recompile the module during execution the auto PGLE will not work neither for AoT nor for the following case:

```
import jax
from jax._src import config

train_step_compiled = train_step().lower().compile()

with config.enable_pgle(True), config.pgle_profiling_runs(1):
  train_step_compiled()
  # No effect since module was pre-compiled.
  train_step_compiled()
```

#### Collecting NVIDIA Nsight Systems profiles when using AutoPGLE
[jax#24910](https://github.com/jax-ml/jax/pull/24910) (JAX v0.5.1 and newer) added a
new JAX configuration option, `JAX_COMPILATION_CACHE_EXPECT_PGLE`, which tells JAX to
attempt to load PGLE-optimized compiled functions from the persistent compilation
cache.

This allows a two-step process, where the first step writes a PGLE-optimized function
to the cache:
```bash
export JAX_ENABLE_COMPILATION_CACHE=yes          # not strictly needed, on by default
export JAX_COMPILATION_CACHE_DIR=/root/jax_cache
JAX_ENABLE_PGLE=yes python my-model.py
```
And the second step uses Nsight Systems and loads the PGLE-optimized function from the
cache:
```bash
JAX_COMPILATION_CACHE_EXPECT_PGLE=yes nsys profile python my-model.py
```
See also [this page](
https://docs.jax.dev/en/latest/persistent_compilation_cache.html#pitfalls) for more
information about the persistent compilation cache and possible pitfalls.

### Manual PGLE

If you still want to use a manual Profile Guided Latency Estimator the workflow in XLA/GPU is:

- 1. Run your workload once, with async collectives and latency hiding scheduler enabled.

You could do so by setting:

```bash
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
```

- 2. Collect and post process a profile by using JAX profiler, saving the extracted instruction latencies into a binary protobuf file.

```python
import os
from etils import epath
import jax
from jax.experimental import profiler as exp_profiler

# Define your profile directory
profile_dir = 'gs://my_bucket/profile'
jax.profiler.start_trace(profile_dir)

# run your workflow
# for i in range(10):
#   train_step()

# Stop trace
jax.profiler.stop_trace()
profile_dir = epath.Path(profile_dir)
directories = profile_dir.glob('plugins/profile/*/')
directories = [d for d in directories if d.is_dir()]
rundir = directories[-1]
logging.info('rundir: %s', rundir)

# Post process the profile
fdo_profile = exp_profiler.get_profiled_instructions_proto(os.fspath(rundir))

# Save the profile proto to a file.
dump_dir = rundir / 'profile.pb'
dump_dir.parent.mkdir(parents=True, exist_ok=True)
dump_dir.write_bytes(fdo_profile)

```

After this step, you will get a `profile.pb` file under the `rundir` printed in the code.

- 3. Run the workload again feeding that file into the compilation.

You need to pass the `profile.pb` file to the `--xla_gpu_pgle_profile_file_or_directory_path` flag.

```bash
 export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_pgle_profile_file_or_directory_path=/path/to/profile/profile.pb"
```

To enable logging in the XLA and check if the profile is good, set the logging level to include `INFO`:

```bash
export TF_CPP_MIN_LOG_LEVEL=0
```

Run the real workflow, if you found these loggings in the running log, it means the profiler is used in the latency hiding scheduler:

```
2023-07-21 16:09:43.551600: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:478] Using PGLE profile from /tmp/profile/plugins/profile/2023_07_20_18_29_30/profile.pb
2023-07-21 16:09:43.551741: I external/xla/xla/service/gpu/gpu_hlo_schedule.cc:573] Found profile, using profile guided latency estimator
```

#### Flags

* **--xla_gpu_enable_latency_hiding_scheduler** This flag enables latency hiding
  schedulers to overlap asynchronous communication with computation efficiently.
  The default value is False.
* **--xla_gpu_memory_limit_slop_factor** This flag serves as a multiplier applied
  to the total available memory, creating a threshold that guides the Latency Hiding
  Scheduler (LHS) in balancing memory reduction and latency hiding optimizations.
  The default value is 95.

  This factor effectively establishes a memory limit for compiler passes, determining
  when the scheduler should prioritize:
    1. Memory reduction: When memory usage approaches or exceeds the calculated threshold.
    2. Latency hiding: When memory usage is below the threshold, allowing for more
       aggressive optimizations that may temporarily increase memory usage but improve
       overall performance.

  By adjusting this factor, users can fine-tune the trade-off between memory efficiency
  and performance optimizations.
* **--xla_gpu_all_gather_combine_threshold_bytes**
  **--xla_gpu_reduce_scatter_combine_threshold_bytes**
  **--xla_gpu_all_reduce_combine_threshold_bytes**
  These flags tune when to combine multiple small
  `AllGather`/`ReduceScatter`/`AllReduce` into one big
  `AllGather`/`ReduceScatter`/`AllReduce` to reduce time spent on cross-device
  communication. For example, for the `AllGather`/`ReduceScatter` thresholds
  on a Transformer-based workload, consider tuning them high enough so as to
  combine at least a Transformer Layer's weight `AllGather`/`ReduceScatter`. By
  default, the `combine_threshold_bytes` is set to 256.

### Pipeline Parallelism on GPU

XLA implements SPMD-based pipeline parallelism optimizations. This is a scaling technique
where the forward and backward pass are split into multiple pipeline stages.
Each device (or device group) processes the result of the previous
pipeline stage (or the pipeline input) and sends its partial result to the next
stage until the end of the pipeline is reached. This optimization works best
when the latency of the computation is larger than communication. At compile
time, the operations will be rearranged to overlap communication with
computation.

For an optimized schedule, we recommend these XLA flags:
```
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_command_buffer=''
--xla_disable_hlo_passes=collective-permute-motion
--xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE
```

The following JAX example demonstrates a pattern where communication operations
are scheduled to overlap with computations. In this example we will illustrate
how to set up an optimized pipeline parallelism scheduling using 4 GPUs that
form a communication ring (device 0 -> device 1 -> device 2 -> device 3 ->
device 0). We refer to the pattern `0 -> 1 -> 2 -> 3` as the forward edge, and
`3 -> 0` as the back edge.

```
# Imports and setup
import functools
import jax
from jax import sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.random

NUM_DEVICES = 4
NUM_MICROBATCHES = 5
NUM_CIRC_REPEATS = 2
CONTRACTING_DIM_SIZE = 4096
NON_CONTRACTING_DIM_SIZE = 8192
COMPUTE_INTENSITY = 32

# Creates a collective permute for the "forward edge".
# 0->1, 1->2, ... (N-2)->(N-1)
def shift_right(arr):
  padding = [[1, 0]] + [[0, 0]] * (arr.ndim - 1)
  # Use lax.slice to guarantee the gradient is a pad.
  return jax.lax.slice(jnp.pad(arr, padding), [0] * arr.ndim, arr.shape)


# Creates a collective permute for the "back edge".
# (N-1)->0
def cycle_back(arr):
  padding = [[0, NUM_DEVICES - 1]] + [[0, 0]] * (arr.ndim - 1)
  return jax.lax.slice(
      jnp.pad(arr, padding),
      [NUM_DEVICES - 1] + [0] * (arr.ndim - 1),
      (NUM_DEVICES - 1 + arr.shape[0],) + arr.shape[1:],
  )


def select_on_first_device(then_value, else_value):
  assert then_value.shape == else_value.shape
  is_first_device = jax.lax.broadcasted_iota("int32", then_value.shape, 0) == 0
  return jnp.where(is_first_device, then_value, else_value)


def select_on_last_device(then_value, else_value):
  assert then_value.shape == else_value.shape
  is_last_device = (
      jax.lax.broadcasted_iota("int32", then_value.shape, 0) == NUM_DEVICES - 1
  )
  return jnp.where(is_last_device, then_value, else_value)


def select_on_first_cycle(i, then_value, else_value):
  assert then_value.shape == else_value.shape
  is_first_cycle = i < NUM_MICROBATCHES
  return jnp.where(is_first_cycle, then_value, else_value)


def while_body(carry, i):
  """Body of the pipeline while loop."""
  weights, input_buffer, output_buffer, fwd_edge_data, bwd_edge_data = carry

  # Read input data from input buffer.
  input_data = jax.lax.dynamic_slice(
      input_buffer,
      (0, (i + 0) % NUM_MICROBATCHES, 0, 0),
      (NUM_DEVICES, 1, CONTRACTING_DIM_SIZE, NON_CONTRACTING_DIM_SIZE),
  )

  # Collective permute on the "forward edge" shifts data to the next stage.
  fwd_edge_data = shift_right(fwd_edge_data)

  # Select compute argument based on device and pipeline cycle.
  compute_argument = select_on_first_device(
      select_on_first_cycle(i, input_data, bwd_edge_data),
      fwd_edge_data,
  ).reshape((NUM_DEVICES, CONTRACTING_DIM_SIZE, NON_CONTRACTING_DIM_SIZE))

  # A few matmuls to simulate compute.
  tmp = compute_argument
  for _ in range(COMPUTE_INTENSITY):
    tmp = jax.lax.dot_general(weights, tmp, (((2,), (1,)), ((0,), (0,))))
  compute_result = tmp.reshape(
      (NUM_DEVICES, 1, CONTRACTING_DIM_SIZE, NON_CONTRACTING_DIM_SIZE)
  )

  # Read data from buffer to pass it to the first device of the pipeline on the
  # "back edge".
  bwd_edge_data = jax.lax.dynamic_slice(
      output_buffer,
      (0, (1 + i) % NUM_MICROBATCHES, 0, 0),
      (NUM_DEVICES, 1, CONTRACTING_DIM_SIZE, NON_CONTRACTING_DIM_SIZE),
  )

  # Collective permute on the "back edge" passes data to the first device.
  bwd_edge_data = cycle_back(bwd_edge_data)

  # Update output buffer. We do this after reading from it to avoid the data
  # dependency.
  output_buffer = jax.lax.dynamic_update_slice(
      output_buffer,
      compute_result,
      (0, (2 + i) % NUM_MICROBATCHES, 0, 0),
  )

  fwd_edge_data = compute_result
  carry = (
      weights,
      input_buffer,
      output_buffer,
      fwd_edge_data,
      bwd_edge_data,
  )
  return carry, i


@functools.partial(jax.jit, static_argnames=["mesh"])
def entry_computation(weights, input_buffer, mesh):

  # Init output buffer.
  output_buffer = jnp.zeros_like(input_buffer)

  # Init dummy data for forward and backward edge passed through the while loop.
  dummy_data = jnp.zeros(
      shape=(NUM_DEVICES, 1, CONTRACTING_DIM_SIZE, NON_CONTRACTING_DIM_SIZE)
  ).astype(jnp.float32)
  dummy_data = jax.device_put(
      dummy_data,
      sharding.NamedSharding(
          mesh, sharding.PartitionSpec("the_one_and_only_axis")
      ),
  )

  # Start pipeline.
  carry = weights, input_buffer, output_buffer, dummy_data, dummy_data
  num_iterations = NUM_CIRC_REPEATS * NUM_MICROBATCHES + NUM_DEVICES - 1
  carry, _ = jax.lax.scan(while_body, carry, xs=jnp.arange(num_iterations))
  _, _, output_buffer, _, _ = carry

  return output_buffer


def main(_):

  # Expect constant number of devices.
  assert NUM_DEVICES == jax.local_device_count()

  # Create mesh.
  mesh = sharding.Mesh(
      mesh_utils.create_device_mesh([NUM_DEVICES]),
      axis_names=["the_one_and_only_axis"],
  )

  # Init weights.
  weights = 1.0 / CONTRACTING_DIM_SIZE
  weights = jax.lax.broadcast_in_dim(
      weights,
      shape=(NUM_DEVICES, CONTRACTING_DIM_SIZE, CONTRACTING_DIM_SIZE),
      broadcast_dimensions=(),
  )
  weights = jax.device_put(
      weights,
      sharding.NamedSharding(
          mesh, sharding.PartitionSpec("the_one_and_only_axis")
      ),
  )

  # Init random input and replicate it across all devices.
  random_key = jax.random.key(0)
  input_buffer = jax.random.uniform(
      random_key,
      shape=(
          NUM_MICROBATCHES,
          CONTRACTING_DIM_SIZE,
          NON_CONTRACTING_DIM_SIZE,
      ),
  )
  input_buffer = jax.lax.broadcast_in_dim(
      input_buffer,
      shape=(
          NUM_DEVICES,
          NUM_MICROBATCHES,
          CONTRACTING_DIM_SIZE,
          NON_CONTRACTING_DIM_SIZE,
      ),
      broadcast_dimensions=[1, 2, 3],
  )
  input_buffer = jax.device_put(
      input_buffer,
      sharding.NamedSharding(
          mesh, sharding.PartitionSpec("the_one_and_only_axis")
      ),
  )

  # Run computation.
  output_buffer = entry_computation(weights, input_buffer, mesh)
  print(f"output_buffer = \n{output_buffer}")
```
## NCCL flags

These Nvidia NCCL flag values may be useful for single-host multi-device
computations on Nvidia GPUs:

```python
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })
```

These NCCL flags could improve single-host communication speed. These flags
don't seem useful for multi-host communication yet.

## Multi-Process

We recommend using one process per GPU and not one per node.  In some
cases, this can speed up jitted computation. The
{func}`jax.distributed.initialize` API will automatically understand
that configuration when run under SLURM. However, this only a rule of
thumb and it may be useful to test both one process per GPU and one
process per node on your use case.
