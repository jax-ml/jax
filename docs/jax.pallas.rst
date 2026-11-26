``jax.pallas`` module
=====================

.. automodule:: jax.pallas

Backends
--------

.. toctree::
    :maxdepth: 1

    Pallas TPU (TensorCore) <jax.pallas.tpu>
    Pallas TPU (SparseCore) <jax.pallas.tpu_sc>
    Pallas MGPU <jax.pallas.mosaic_gpu>

Libraries
---------

.. toctree::
    :maxdepth: 1

    Fuser <jax.pallas.fuser>

Classes
-------

.. autosummary::
  :toctree: _autosummary

  Blocked
  BlockSpec
  BoundedSlice
  Buffered
  CompilerParams
  CostEstimate
  Element
  Indirect
  MemoryRef
  MemorySpace
  RevisitMode
  Slice
  Squeezed
  semaphore

Functions
---------

.. autosummary::
  :toctree: _autosummary

  kernel
  pallas_call
  program_id
  num_programs

  cdiv
  dslice
  ds
  empty
  empty_like
  empty_ref_like
  pad_ref

  broadcast_to
  debug_check
  debug_print
  delay
  get_global
  loop
  multiple_of
  run_scoped
  run_state
  select_ref
  when
  with_scoped

Utilities
---------

.. autosummary::
  :toctree: _autosummary

  align_to
  next_power_of_2
  strides_from_shape

Cost estimation and lowering
----------------------------

.. autosummary::
  :toctree: _autosummary

  estimate_cost
  lower_as_mlir
  pallas_export_experimental

Debugging
---------

.. autosummary::
  :toctree: _autosummary

  enable_debug_checks
  enable_poison_buffers

Synchronization
---------------

.. autosummary::
  :toctree: _autosummary

  semaphore_read
  semaphore_signal
  semaphore_wait

Aliases
-------

.. autosummary::
  :toctree: _autosummary

  ANY
  HOST
  BlockDim
  no_block_spec
  squeezed
