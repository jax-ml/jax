``jax.experimental.pallas.mosaic_gpu`` module
=============================================

.. automodule:: jax.experimental.pallas.mosaic_gpu

Classes
-------

.. autosummary::
   :toctree: _autosummary

   Barrier
   GPUBlockSpec
   GPUCompilerParams
   GPUMemorySpace
   Layout
   SwizzleTransform
   TilingTransform
   TransposeTransform
   WGMMAAccumulatorRef

Functions
---------

.. autosummary::
   :toctree: _autosummary

   barrier_arrive
   barrier_wait
   commit_smem
   copy_gmem_to_smem
   copy_smem_to_gmem
   emit_pipeline
   layout_cast
   set_max_registers
   wait_smem_to_gmem
   wgmma
   wgmma_wait

Aliases
-------

.. autosummary::
   :toctree: _autosummary

   ACC
   GMEM
   SMEM
