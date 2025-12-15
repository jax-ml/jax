``jax.experimental.pallas.tpu`` module
======================================

.. automodule:: jax.experimental.pallas.tpu

Classes
-------

.. autosummary::
   :toctree: _autosummary

   ChipVersion
   CompilerParams
   GridDimensionSemantics
   MemorySpace
   PrefetchScalarGridSpec
   SemaphoreType
   TpuInfo

Communication
-------------

.. autosummary::
   :toctree: _autosummary

   async_copy
   async_remote_copy
   make_async_copy
   make_async_remote_copy
   sync_copy

Pipelining
----------

.. autosummary::
   :toctree: _autosummary

   BufferedRef
   BufferedRefBase
   emit_pipeline
   emit_pipeline_with_allocations
   get_pipeline_schedule
   make_pipeline_allocations


Pseudorandom Number Generation
------------------------------

.. autosummary::
   :toctree: _autosummary

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

   core_barrier
   get_barrier_semaphore
   get_tpu_info
   is_tpu_device
   run_on_first_core
   with_memory_space_constraint

