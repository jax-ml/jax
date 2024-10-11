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
   SwizzleTransform
   TilingTransform
   TransposeTransform
   WGMMAAccumulatorRef

Functions
---------

.. autosummary::
   :toctree: _autosummary

   copy_gmem_to_smem
   copy_smem_to_gmem
   wait_barrier
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
