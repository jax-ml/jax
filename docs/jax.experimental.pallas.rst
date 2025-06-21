``jax.experimental.pallas`` module
==================================

.. automodule:: jax.experimental.pallas

Backends
--------

.. toctree::
    :maxdepth: 1

    jax.experimental.pallas.mosaic_gpu
    jax.experimental.pallas.triton
    jax.experimental.pallas.tpu

Classes
-------

.. autosummary::
  :toctree: _autosummary

  BlockSpec
  GridSpec
  Slice

  MemoryRef

Functions
---------

.. autosummary::
  :toctree: _autosummary

  pallas_call
  program_id
  num_programs

  load
  store
  swap

  atomic_and
  atomic_add
  atomic_cas
  atomic_max
  atomic_min
  atomic_or
  atomic_xchg
  atomic_xor
  broadcast_to
  debug_print
  dot
  max_contiguous
  multiple_of
  run_scoped
  when
