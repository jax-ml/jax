GPU memory allocation
=====================

**JAX will preallocate 75% of the total GPU memory when the first JAX
operation is run.** Preallocating minimizes allocation overhead and memory
fragmentation, but can sometimes cause out-of-memory (OOM) errors. If your JAX
process fails with OOM, the following environment variables can be used to
override the default behavior:

``XLA_PYTHON_CLIENT_PREALLOCATE=false``
  This disables the preallocation behavior.  JAX will instead allocate GPU
  memory as needed, potentially decreasing the overall memory usage.  However,
  this behavior is more prone to GPU memory fragmentation, meaning a JAX program
  that uses most of the available GPU memory may OOM with preallocation
  disabled.

``XLA_PYTHON_CLIENT_MEM_FRACTION=.XX``
  If preallocation is enabled, this makes JAX preallocate XX% of
  the total GPU memory, instead of the default 75%. Lowering the
  amount preallocated can fix OOMs that occur when the JAX program starts.

``XLA_PYTHON_CLIENT_ALLOCATOR=platform``
  This makes JAX allocate exactly what is needed on demand, and deallocate
  memory that is no longer needed (note that this is the only configuration that
  will deallocate GPU memory, instead of reusing it). This is very slow, so is
  not recommended for general use, but may be useful for running with the
  minimal possible GPU memory footprint or debugging OOM failures.


Common causes of OOM failures
-----------------------------

**Running multiple JAX processes concurrently.**
  Either use :code:`XLA_PYTHON_CLIENT_MEM_FRACTION` to give each process an
  appropriate amount of memory, or set
  :code:`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

**Running JAX and GPU TensorFlow concurrently.**
  TensorFlow also preallocates by default, so this is similar to running
  multiple JAX processes concurrently.

  One solution is to use CPU-only
  TensorFlow (e.g. if you're only doing data loading with TF). You can prevent
  TensorFlow from using the GPU with the command
  :code:`tf.config.experimental.set_visible_devices([], "GPU")`

  Alternatively, use :code:`XLA_PYTHON_CLIENT_MEM_FRACTION` or
  :code:`XLA_PYTHON_CLIENT_PREALLOCATE`. There are
  also similar options to configure TensorFlow's GPU memory allocation
  (`gpu_memory_fraction
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L36>`_
  and `allow_growth
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L40>`_
  in TF1, which should be set in a :code:`tf.ConfigProto` passed to
  :code:`tf.Session`. See
  `Using GPUs: Limiting GPU memory growth
  <https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_
  for TF2).

**Running JAX on the display GPU.**
  Use :code:`XLA_PYTHON_CLIENT_MEM_FRACTION` or
  :code:`XLA_PYTHON_CLIENT_PREALLOCATE`.

**Disabling rematerialization HLO pass**
  Sometimes disabling the automatic rematerialization HLO pass is favorable to avoid 
  poor remat choices by the compiler. The pass can be enable/disable by setting
  :code:`jax.config.update('enable_remat_opt_pass', True)` or 
  :code:`jax.config.update('enable_remat_opt_pass', False)` respectively. Enabling or
  disabling the automatic remat pass produces different trade-offs between compute and 
  memory. Note however, that the algorithm is basic and you can often get better 
  trade-off between compute and memory by disabling the automatic remat pass and doing
  it manually with `the jax.remat API <https://docs.jax.dev/en/latest/jep/11830-new-remat-checkpoint.html>`_


Experimental features
---------------------

Features here are experimental and must be tried with caution.

``TF_GPU_ALLOCATOR=cuda_malloc_async``
  This replace XLA's own BFC memory allocator with `cudaMallocAsync
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>`_.
  This will remove the big fixed pre-allocation and use a memory pool that grows.
  The expected benefit is no need to set `XLA_PYTHON_CLIENT_MEM_FRACTION`.

  The risk are:

  - that memory fragmentation is different, so if you are close to the
    limit, the exact OOM case due to fragmentation will be different.
  - The allocation time won't be all paid at the start, but be incurred
    when the memory pool need to be increased. So you could
    experience less speed stability at the start and for benchmarks
    it will be even more important to ignore the first few iterations.

  The risks can be mitigated by pre-allocating a signigicant chunk and
  still get the benefit of having a growing memory pool. This can be
  done with `TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=N`. If N is `-1`
  it will preallocate the same as what was allocatedy by
  default. Otherwise, it is the size in bytes that you want to
  preallocate.
