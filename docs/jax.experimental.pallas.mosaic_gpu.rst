``jax.experimental.pallas.mosaic_gpu`` module
=============================================

.. automodule:: jax.experimental.pallas.mosaic_gpu

Classes
-------

.. autosummary::
   :toctree: _autosummary

   Barrier
   BlockSpec
   CompilerParams
   MemorySpace
   Layout
   SwizzleTransform
   TilingTransform
   TransposeTransform
   WGMMAAccumulatorRef

Functions
---------

.. autosummary::
   :toctree: _autosummary

   as_torch_kernel
   kernel
   layout_cast
   set_max_registers
   planar_snake

Loop-like functions
-------------------

.. autosummary::
   :toctree: _autosummary

   emit_pipeline
   emit_pipeline_warp_specialized
   nd_loop
   dynamic_scheduling_loop

Synchronization
---------------

.. autosummary::
   :toctree: _autosummary

   barrier_arrive
   barrier_wait
   semaphore_signal_parallel
   SemaphoreSignal

Asynchronous copies
-------------------

.. autosummary::
   :toctree: _autosummary

   commit_smem
   copy_gmem_to_smem
   copy_smem_to_gmem
   wait_smem_to_gmem

Hopper-specific functions
-------------------------

.. autosummary::
   :toctree: _autosummary

   wgmma
   wgmma_wait

Blackwell-specific functions
----------------------------

.. autosummary::
   :toctree: _autosummary

   tcgen05_mma
   tcgen05_commit_arrive
   async_load_tmem
   async_store_tmem
   wait_load_tmem
   commit_tmem
   try_cluster_cancel
   query_cluster_cancel

Multimem operations
-------------------

.. autosummary::
   :toctree: _autosummary

   multimem_store
   multimem_load_reduce

Aliases
-------

.. autosummary::
   :toctree: _autosummary

   ACC
   GMEM
   SMEM
