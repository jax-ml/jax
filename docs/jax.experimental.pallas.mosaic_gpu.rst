``jax.experimental.pallas.mosaic_gpu`` module
=============================================

.. automodule:: jax.experimental.pallas.mosaic_gpu

Classes
-------

.. autosummary::
   :toctree: _autosummary

   Barrier
   BlockSpec
   ClusterBarrier
   CompilerParams
   CopyPartition
   MemorySpace
   Layout
   LoweringSemantics
   OOBFillMode
   PeerMemRef
   RefType
   RefUnion
   Replicated
   SemaphoreType
   ShapeDtypeStruct
   SwizzleTransform
   Tiling
   TilingTransform
   TMEMLayout
   TraceScope
   Transform
   TryClusterCancelResult
   WarpMesh
   WGMMAAccumulatorRef

Functions
---------

.. autosummary::
   :toctree: _autosummary

   as_torch_kernel
   broadcasted_iota
   cluster_ref
   find_swizzle
   inline_mgpu
   kernel
   layout_cast
   load
   multicast_ref
   print_layout
   remote_ref
   set_max_registers
   planar_snake
   store
   warp_map

Reductions
----------

.. autosummary::
   :toctree: _autosummary

   max
   min
   prod
   sum

Atomic operations
-----------------

.. autosummary::
   :toctree: _autosummary

   atomic_add
   atomic_and
   atomic_max
   atomic_min
   atomic_or
   atomic_xor

Loop-like functions
-------------------

.. autosummary::
   :toctree: _autosummary

   emit_pipeline
   emit_pipeline_warp_specialized
   PipelinePipeline
   nd_loop
   NDLoopInfo
   dynamic_scheduling_loop

Synchronization
---------------

.. autosummary::
   :toctree: _autosummary

   alloc_semaphore
   barrier_arrive
   barrier_arrive_and_wait
   barrier_test
   barrier_wait
   get_barrier_semaphore
   griddepcontrol_launch_dependents
   griddepcontrol_wait
   semaphore_signal
   semaphore_signal_multicast
   semaphore_signal_parallel
   semaphore_wait
   SemaphoreSignal

Asynchronous copies
-------------------

.. autosummary::
   :toctree: _autosummary

   async_prefetch
   async_store_smem
   commit_smem
   commit_smem_to_gmem_group
   copy_gmem_to_smem
   copy_smem_to_gmem
   wait_gmem_to_smem
   wait_smem_to_gmem
   im2col_ref

Ampere-specific functions
-------------------------

.. autosummary::
   :toctree: _autosummary

   mma
   Layout.MMA_LHS
   Layout.MMA_RHS
   Layout.MMA_ACC

Hopper-specific functions
-------------------------

.. autosummary::
   :toctree: _autosummary

   wgmma
   wgmma_accumulator_load
   wgmma_wait

Blackwell-specific functions
----------------------------

.. autosummary::
   :toctree: _autosummary

   tcgen05_mma
   tcgen05_commit_arrive
   async_copy_scales_to_tmem
   async_copy_smem_to_tmem
   async_copy_sparse_metadata_to_tmem
   async_load_tmem
   async_store_tmem
   format_tcgen05_sparse_metadata
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

Interpret Mode
--------------

.. autosummary::
   :toctree: _autosummary

   force_gpu_interpret_mode
   InterpretGPUParams
   set_gpu_interpret_mode

Aliases
-------

.. autosummary::
   :toctree: _autosummary

   ACC
   GMEM
   SMEM
   TMEM
   REGS
