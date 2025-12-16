Pseudo-Random Number Generation
===============================

Pallas TPU implements several APIs for generating pseudorandom numbers inside of a kernel with varying tradeoffs in portability and efficiency. For maximum portability, consider using `jax.random` functions directly. Pallas also exposes the hardware PRNG contained on TPUs which are the fastest to compute but the underlying implementation can vary between hardware generations.

Using the ``jax.random`` API
----------------------------

Pallas supports a subset of operations in the ``jax.random`` API. These functions are guaranteed to produce bitwise-equal results compared to calling these functions in JAX outside of Pallas when given the same key. Only ``threefry2x32`` keys are supported.

The following random sampling functions are currently supported:

* :func:`jax.random.bits`
* :func:`jax.random.uniform`
* :func:`jax.random.bernoulli`
* :func:`jax.random.normal`

The following utility functions are supported:

* :func:`jax.random.key`
* :func:`jax.random.fold_in`
* :func:`jax.random.wrap_key_data`

PRNG keys can be generated inside of the kernel using :func:`jax.random.key`. However, the more likely scenario is that a key will be passed into the kernel from the caller. In such a case, the key can be passed into the kernel via VMEM as follows:

.. code-block:: python

  def body(key_ref, o_ref):
    key = key_ref[...]
    o_ref[...] = jax_random.uniform(
        key, shape=o_ref[...].shape, minval=0.0, maxval=1.0
    )

  threefry_key = jax_random.key(0, impl="threefry2x32")

  # We generate a threefry key outside of the kernel and pass it in via VMEM.
  result = pl.pallas_call(
      body,
      in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
      out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32)
  )(threefry_key)

.. note::

  In terms of performance concerns, generating random numbers inside of a kernel helps reduce memory bandwidth usage as it is cheaper to pass in a key than a large array of random numbers. However, ``threefry2x32`` is a vector-heavy algorithm that involves dozens of chained bitwise operations. This can become a bottleneck and lead to low accelerator usage as it does not utilize the matrix multiply unit (MXU) where the majority of FLOP/s are.

Using the hardware PRNG
-----------------------

TPUs implement a sequential (rather than counter-based) PRNG natively in hardware that is much faster to compute than using a software-implemented PRNG such as ``threefry2x32``. However, JAX random APIs assume a stateless, counter-based PRNG so Pallas introduces its own stateful PRNG API to offer equivalent functionality.

.. warning::

  The underlying implementation of the hardware PRNG varies between TPU generations, so it is best practice to not depend on its exact behavior. For a more stable PRNG implemented in software, it is recommended to use the ``threefry2x32`` implementation.


Stateful Random Number Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the Pallas PRNG in stateful mode is the most native and efficient method for generative random numbers. First, the PRNG seed should be set using ``pltpu.prng_seed(N)``, where N is an integer seed.

Afterwards, you can call any number of stateful sampling functions which are equivalent to the corresponding JAX version but lack the ``key`` argument:

* ``pltpu.stateful_uniform``: the stateful equivalent to :func:`jax.random.uniform`
* ``pltpu.stateful_normal``: the stateful equivalent to :func:`jax.random.normal`
* ``pltpu.stateful_bernoulli``: the stateful equivalent to :func:`jax.random.bernoulli`

Generating any random number updates the internal state of the PRNG and subsequent calls will generate different numbers. Unlike in JAX, there is no need to ``split`` or ``fold_in`` keys and pass them into the sampling functions.

For example, the following kernel generates a set of uniform numbers from 0 to 1:

.. code-block:: python

  from jax.experimental.pallas import tpu as pltpu

  def kernel_body(o_ref):
    pltpu.prng_seed(0)
    o_ref[...] = pltpu.stateful_uniform(shape=o_ref.shape, minval=0.0, maxval=1.0)

  pl.pallas_call(kernel_body,
                 out_shape=jax.ShapeDtypeStruct((256, 256), jnp.float32))

Note that in kernels with a grid, the seed should only be set on the first iteration, or else the random numbers generated in each program instance will be identical due to resetting the seed.

Stateless Generation 
^^^^^^^^^^^^^^^^^^^^

Pallas offers an intermediate API between the stateless API described previously and the stateless ``jax.random`` API and allows you to use the hardware PRNG in a stateless manner. In order to do so, convert a JAX key into a special Pallas-typed key via ``pltpu.to_pallas_key(key)`` and pass this key into the kernel via SMEM. Once the key is dereferenced inside the kernel, it can be passed into supported sampling functions from ``jax.random`` to produce random numbers. Compared to the stateless API, there is an overhead of computing and setting a seed every time the random number generator is invoked.

For example, the following kernel draws uniform numbers using the hardware PRNG:

.. code-block:: python

  def body(key_ref, o_ref):
    o_ref[...] = jax.random.uniform(
        key_ref[...], shape=o_ref[...].shape
    )

  rbg_key = jax_random.key(0, impl="threefry2x32")
  key = pltpu.to_pallas_key(rbg_key)
  o_shape = jax.ShapeDtypeStruct((8, 128), dtype)
  result = pl.pallas_call(
      body,
      in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)],
      out_shape=o_shape,
  )(key)

For larger kernels with a grid, :func:`jax.random.fold_in` can be used on the ``program_id`` to generate a unique key for each program instance.  


Block-invariant sampling
------------------------

Block-invariant sampling is a method for generating random numbers in blocks that is invariant to the block sizes and iteration order used. For example, you may wish to generate identical sets of random numbers between two kernels (such as a forwards and backwards pass), but the two kernels may have different block sizes chosen after tuning.

Pallas providers a helper function (``pltpu.sample_block``) that allows one to guarantee identical random numbers drawn over different block and grid settings. The first step is to select a ``tile_size``, which is a tile that divides all block sizes you wish to be invariant to. For example, ``tile_size=(16, 128)`` would work for block sizes of ``(32, 128)`` and ``(16, 256)``. The larger the tile size, the more efficient the sampling process will be, so the greatest common divisor between all potential block sizes is the best choice.

Next, call ``pltpu.sample_block`` with the following arguments:

.. code-block:: python

  pltpu.sample_block(
    sampler_function,  # A JAX random function, such as `jax.random.uniform`.
    global_key,  # A global key shared across all blocks.
    block_size,  # The local block size to generate.
    tile_size,  # The tile size.
    total_size,  # The total shape of the generated array across all blocks.
    block_index,  # The block index into total_size. Usually this is the current program instance.
    **sampler_kwargs  # Keyword arguments to sampler_function
  )

For example, the following snippet generates identical numbers over a `(16, 128)` block shape, and a `(32, 256)` block shape with a transposed grid iteration order:

.. code-block:: python

  def make_kernel_body(index_map):
    def body(key_ref, o_ref):
      key = key_ref[...]
      samples = pltpu.sample_block(
          jax.random.uniform,
          key,
          block_size=o_ref[...].shape,
          tile_size=(16, 128),
          total_size=(64, 512),
          block_index=index_map(pl.program_id(0), pl.program_id(1)),
          minval=0.0,
          maxval=1.0)
      o_ref[...] = samples
    return body

  global_key = pltpu.to_pallas_key(jax_random.key(0))
  o_shape = jnp.ones((64, 512), dtype=jnp.float32)
  key_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  out_spec = pl.BlockSpec((16, 128), lambda i, j: (i, j))
  result_16x128 = pl.pallas_call(
      make_kernel_body(index_map=lambda i, j: (i, j)),
      out_shape=o_shape,
      in_specs=[key_spec],
      out_specs=out_spec,
      grid=(4, 4),
  )(global_key)

  out_spec = pl.BlockSpec((32, 256), lambda i, j: (j, i))
  result_32x256_transposed = pl.pallas_call(
      make_kernel_body(index_map=lambda i, j: (j, i)),
      in_specs=[key_spec],
      out_shape=o_shape,
      out_specs=out_spec,
      grid=(2, 2),
  )(global_key)

