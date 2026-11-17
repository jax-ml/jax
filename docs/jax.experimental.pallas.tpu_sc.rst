``jax.experimental.pallas.tpu_sc`` module
=========================================

.. automodule:: jax.experimental.pallas.tpu_sc

Classes
-------

.. autosummary::
   :toctree: _autosummary

   Indices
   PackFormat
   ScalarSubcoreMesh
   SparseCorePushStreamWarning
   VectorSubcoreMesh

Functions
---------

.. autosummary::
   :toctree: _autosummary

   bitcast
   pack
   parallel_loop
   sort_key_val
   unpack

Memory operations
-----------------

.. autosummary::
   :toctree: _autosummary

   addupdate
   addupdate_compressed
   addupdate_scatter
   fetch_and_add
   load_expanded
   load_gather
   slice_subcore_vmem_from_shared
   store_compressed
   store_scatter

Reductions and scans
--------------------

.. autosummary::
   :toctree: _autosummary

   all_reduce_ffs
   all_reduce_population_count
   cummax
   cumsum
   scan_count

Miscellaneous
-------------

.. autosummary::
   :toctree: _autosummary

   get_sparse_core_info
   subcore_barrier
