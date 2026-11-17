``jax.experimental.pallas.tpu`` module
======================================

.. automodule:: jax.experimental.pallas.tpu

Classes
-------

.. autosummary::
   :toctree: _autosummary

   ChipVersion
   CompilerParams
   CoreType
   GridDimensionSemantics
   LoweringException
   MemorySpace
   PrefetchScalarGridSpec
   SemaphoreType
   SideEffectType
   TensorCoreMesh
   Tiling
   TpuInfo
   dma_semaphore

Functions
---------

.. autosummary::
   :toctree: _autosummary

   load
   store

   bitcast
   einshape
   reciprocal
   roll
   stochastic_round

   join_bits
   pack_elementwise
   pack_fp6
   split_bits
   unpack_elementwise
   unpack_fp6

Communication
-------------

.. autosummary::
   :toctree: _autosummary

   async_copy
   async_remote_copy
   make_async_copy
   make_async_remote_copy
   sync_copy

Matrix multiplication
---------------------

.. autosummary::
   :toctree: _autosummary

   conv
   matmul_acc_lhs
   matmul_lhs_fifo
   matmul_pop
   matmul_pop_fifo
   matmul_push_rhs

Pipelining
----------

.. autosummary::
   :toctree: _autosummary

   BufferedRef
   BufferedRefBase
   BufferType
   PipelineStep
   emit_pipeline
   emit_pipeline_with_allocations
   emit_pipeline_with_async_prefetch

Pseudorandom Number Generation
------------------------------

.. autosummary::
   :toctree: _autosummary

   prng_random_bits
   prng_seed
   sample_block
   stateful_bernoulli
   stateful_bits
   stateful_normal
   stateful_uniform
   to_pallas_key

Interpret Mode
--------------

.. autosummary::
   :toctree: _autosummary

   force_tpu_interpret_mode
   InterpretParams
   reset_tpu_interpret_mode_state
   set_tpu_interpret_mode

Miscellaneous
-------------

.. autosummary::
   :toctree: _autosummary

   annotate
   core_barrier
   get_barrier_semaphore
   get_tpu_info
   get_tpu_info_for_chip
   has_memory_space
   is_tpu_device
   run_on_first_core
   touch
   trace_value
   with_memory_space_constraint

Aliases
-------

.. autosummary::
   :toctree: _autosummary

   ACC
   CMEM
   HBM
   SEMAPHORE
   SMEM
   VMEM
   VMEM_SHARED

   ARBITRARY
   CORE_PARALLEL
   PARALLEL
   SUBCORE_PARALLEL
