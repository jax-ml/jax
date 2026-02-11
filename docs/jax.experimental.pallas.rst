``jax.experimental.pallas`` module
==================================

.. automodule:: jax.experimental.pallas

Backends
--------

.. toctree::
    :maxdepth: 1

    Pallas TPU (TensorCore) <jax.experimental.pallas.tpu>
    Pallas MGPU <jax.experimental.pallas.mosaic_gpu>
    Triton <jax.experimental.pallas.triton>

Classes
-------

.. autosummary::
  :toctree: _autosummary

  BlockSpec
  GridSpec
  Slice

Functions
---------

.. autosummary::
  :toctree: _autosummary

  core_map
  kernel
  pallas_call
  program_id
  num_programs

  cdiv
  dslice
  empty
  empty_like

  broadcast_to
  debug_check
  debug_print
  dot
  get_global
  loop
  multiple_of
  run_scoped
  when
  with_scoped

Synchronization
---------------

.. autosummary::
  :toctree: _autosummary

  semaphore_read
  semaphore_signal
  semaphore_wait
