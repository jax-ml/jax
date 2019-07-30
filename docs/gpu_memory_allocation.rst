GPU memory allocation
=====================

**JAX will preallocate 90% of currently-available GPU memory when the first JAX
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
  currently-available GPU memory, instead of the default 90%. Lowering the
  amount preallocated can fix OOMs that occur when the JAX program starts.

``XLA_PYTHON_CLIENT_ALLOCATOR=platform``
  This makes JAX allocate exactly what is needed on demand. This is very slow,
  so is not recommended for general use, but may be useful for debugging OOM
  failures.


Common causes of OOM failures
-----------------------------

**Running multiple JAX processes concurrently.**
  Either use XLA_PYTHON_CLIENT_MEM_FRACTION to give each process an appropriate
  amount of memory, or set XLA_PYTHON_CLIENT_PREALLOCATE=false.

**Running JAX and GPU TensorFlow concurrently.**
  TensorFlow also preallocates by default, so this is similar to running
  multiple JAX processes concurrently. One solution is to use CPU-only
  TensorFlow (e.g. if you're only doing data loading with TF). Alternatively,
  use XLA_PYTHON_CLIENT_MEM_FRACTION or XLA_PYTHON_CLIENT_PREALLOCATE. There are
  also similar options to configure TensorFlow's GPU memory allocation
  (`gpu_memory_fraction
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L36>`_
  and `allow_growth
  <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto#L40>`_
  in TF1, which should be set in a tf.ConfigProto passed to tf.Session. See
  `Using GPUs: Limiting GPU memory growth
  <https://www.tensorflow.org/beta/guide/using_gpu#limiting_gpu_memory_growth>`_
  for TF2).

**Running JAX on the display GPU.**
  Use XLA_PYTHON_CLIENT_MEM_FRACTION or XLA_PYTHON_CLIENT_PREALLOCATE.
