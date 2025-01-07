``jax.ffi`` module
==================

.. automodule:: jax.ffi

.. autosummary::
  :toctree: _autosummary

  ffi_call
  ffi_lowering
  pycapsule
  register_ffi_target
  register_ffi_type_id


``jax.extend.ffi`` module (deprecated)
======================================

The ``jax.extend.ffi`` module has been moved to ``jax.ffi``, and that import
path should be used instead, but these functions remain documented here while
the legacy import is being deprecated.

.. automodule:: jax.extend.ffi

.. autosummary::
  :toctree: _autosummary

  ffi_call
  ffi_lowering
  pycapsule
  register_ffi_target
