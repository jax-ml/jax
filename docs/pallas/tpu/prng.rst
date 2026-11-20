Pseudo-Random Number Generation
===============================

Pallas TPU implements several APIs for generating pseudorandom numbers inside of a kernel with varying tradeoffs in portability and efficiency. For maximum portability, consider using ``jax.random`` functions directly. Pallas also exposes the hardware PRNG contained on TPUs which is the fastest to compute, although the underlying implementation can vary between hardware generations.

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

PRNG keys can be generated inside of the kernel using :func:`jax.random.key`. However, the more likely scenario is that a key will be passed into the kernel from the caller. With ``pl.kernel``, the key arrives in HBM and can be staged into VMEM via ``scratch_types`` and ``pltpu.sync_copy`` as follows:

.. code-block:: python

  import jax
  from jax import random as jax_random
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  import jax.numpy as jnp

  threefry_key = jax_random.key(0, impl="threefry2x32")
  out_type = jax.ShapeDtypeStruct((256, 256), jnp.float32)
  mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=1)

  @pl.kernel(
      out_type=out_type,
      mesh=mesh,
      scratch_types=[
          pltpu.VMEM.like(threefry_key),
          pltpu.VMEM(out_type.shape, out_type.dtype),
      ],
  )
  def body(key_hbm_ref, o_hbm_ref, key_vmem_ref, o_vmem_ref):
    pltpu.sync_copy(key_hbm_ref, key_vmem_ref)
    key = key_vmem_ref[...]
    o_vmem_ref[...] = jax_random.uniform(
        key, shape=o_vmem_ref.shape, minval=0.0, maxval=1.0
    )
    pltpu.sync_copy(o_vmem_ref, o_hbm_ref)

  result = body(threefry_key)

.. note::

  In terms of performance concerns, generating random numbers inside of a kernel helps reduce memory bandwidth usage as it is cheaper to pass in a key than a large array of random numbers. However, ``threefry2x32`` is a vector-heavy algorithm that involves dozens of chained bitwise operations. This can become a bottleneck and lead to low accelerator usage as it does not utilize the matrix multiply unit (MXU) where the majority of FLOP/s are.

Using the hardware PRNG
-----------------------

TPUs implement a sequential (rather than counter-based) PRNG natively in hardware that is much faster to compute than using a software-implemented PRNG such as ``threefry2x32``. However, JAX random APIs assume a stateless, counter-based PRNG so Pallas introduces its own stateful PRNG API to offer equivalent functionality.

.. warning::

  The underlying implementation of the hardware PRNG varies between TPU generations, so it is best practice to not depend on its exact behavior. For a more stable PRNG implemented in software, it is recommended to use the ``threefry2x32`` implementation.


Stateful Random Number Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the Pallas PRNG in stateful mode is the most native and efficient method for generating random numbers. First, the PRNG seed should be set using ``pltpu.prng_seed(N)``, where ``N`` is an integer seed.

Afterwards, you can call any number of stateful sampling functions which are equivalent to the corresponding JAX version but lack the ``key`` argument:

* ``pltpu.stateful_uniform``: the stateful equivalent to :func:`jax.random.uniform`
* ``pltpu.stateful_normal``: the stateful equivalent to :func:`jax.random.normal`
* ``pltpu.stateful_bernoulli``: the stateful equivalent to :func:`jax.random.bernoulli`

Generating any random number updates the internal state of the PRNG and subsequent calls will generate different numbers. Unlike in JAX, there is no need to ``split`` or ``fold_in`` keys and pass them into the sampling functions.

For example, the following pipelined kernel generates uniform numbers from 0 to 1 across blocks, seeding the PRNG once on the first pipeline step:

.. code-block:: python

  out_type = jax.ShapeDtypeStruct((256, 256), jnp.float32)
  mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=1)

  @pl.kernel(out_type=out_type, mesh=mesh)
  def sample_uniform(o_hbm_ref):
    def step_body(o_vmem_ref):
      @pl.when((pl.program_id(0) == 0) & (pl.program_id(1) == 0))
      def _():
        pltpu.prng_seed(0)

      o_vmem_ref[...] = pltpu.stateful_uniform(
          shape=o_vmem_ref.shape, minval=0.0, maxval=1.0
      )

    pltpu.emit_pipeline(
        step_body,
        grid=(2, 2),
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
    )(o_hbm_ref)

  result = sample_uniform()

Note that in pipelined kernels with a grid, the seed should only be set on the first iteration, or else the random numbers generated in each program instance will be identical due to resetting the seed.

Stateless Generation
^^^^^^^^^^^^^^^^^^^^

Pallas offers an intermediate API between the stateful API described previously and the stateless ``jax.random`` API that allows you to use the hardware PRNG in a stateless manner. Convert a JAX key into a special Pallas-typed key via ``pltpu.to_pallas_key(key)`` and copy this key from HBM into SMEM inside ``pl.kernel``. Once the key is dereferenced in SMEM, it can be passed into supported sampling functions from ``jax.random`` to produce random numbers. Compared to the stateful API, there is a small overhead of computing and setting a seed every time the random number generator is invoked.

For example, the following kernel draws uniform numbers using the hardware PRNG:

.. code-block:: python

  rbg_key = jax_random.key(0, impl="rbg")
  key = pltpu.to_pallas_key(rbg_key)
  out_type = jax.ShapeDtypeStruct((8, 128), jnp.float32)
  mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=1)

  @pl.kernel(
      out_type=out_type,
      mesh=mesh,
      scratch_types=[
          pltpu.SMEM.like(key),
          pltpu.VMEM(out_type.shape, out_type.dtype),
      ],
  )
  def body(key_hbm_ref, o_hbm_ref, key_smem_ref, o_vmem_ref):
    pltpu.sync_copy(key_hbm_ref, key_smem_ref)
    o_vmem_ref[...] = jax.random.uniform(
        key_smem_ref[...], shape=o_vmem_ref.shape
    )
    pltpu.sync_copy(o_vmem_ref, o_hbm_ref)

  result = body(key)

For larger pipelined kernels with a grid, :func:`jax.random.fold_in` can be used on ``pl.program_id`` to generate a unique key for each pipeline step.


Block-invariant sampling
------------------------

Block-invariant sampling is a method for generating random numbers in blocks that is invariant to the block sizes and iteration order used. For example, you may wish to generate identical sets of random numbers between two kernels (such as a forwards and backwards pass), but the two kernels may have different block sizes chosen after tuning.

Pallas provides a helper function (``pltpu.sample_block``) that guarantees identical random numbers drawn over different block and grid settings. The first step is to select a ``tile_size``, which is a tile that divides all block sizes you wish to be invariant to. For example, ``tile_size=(16, 128)`` works for block sizes of ``(32, 128)`` and ``(16, 256)``. The larger the tile size, the more efficient the sampling process will be, so the greatest common divisor between all potential block sizes is the best choice.

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

For example, the following snippet generates identical numbers over a ``(16, 128)`` block shape, and a ``(32, 256)`` block shape with a transposed grid iteration order:

.. code-block:: python

  def run_block_invariant_sample(global_key, block_shape, grid, index_map):
    out_type = jax.ShapeDtypeStruct((64, 512), jnp.float32)
    mesh = pltpu.TensorCoreMesh(axis_name="core", num_cores=1)

    @pl.kernel(
        out_type=out_type,
        mesh=mesh,
        scratch_types=[pltpu.SMEM.like(global_key)],
    )
    def kernel(key_hbm_ref, o_hbm_ref, key_smem_ref):
      pltpu.sync_copy(key_hbm_ref, key_smem_ref)

      def step_body(o_vmem_ref):
        o_vmem_ref[...] = pltpu.sample_block(
            jax.random.uniform,
            key_smem_ref[...],
            block_size=o_vmem_ref.shape,
            tile_size=(16, 128),
            total_size=(64, 512),
            block_index=index_map(pl.program_id(0), pl.program_id(1)),
            minval=0.0,
            maxval=1.0,
        )

      pltpu.emit_pipeline(
          step_body,
          grid=grid,
          out_specs=pl.BlockSpec(block_shape, index_map),
      )(o_hbm_ref)

    return kernel(global_key)

  global_key = pltpu.to_pallas_key(jax_random.key(0, impl="rbg"))
  result_16x128 = run_block_invariant_sample(
      global_key, (16, 128), grid=(4, 4), index_map=lambda i, j: (i, j)
  )
  result_32x256_transposed = run_block_invariant_sample(
      global_key, (32, 256), grid=(2, 2), index_map=lambda i, j: (j, i)
  )
