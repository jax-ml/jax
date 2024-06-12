``jax.export`` module
=====================

.. automodule:: jax.export

Classes
-------

.. autosummary::
  :toctree: _autosummary

  Exported
  DisabledSafetyCheck

Functions
---------

.. autosummary::
  :toctree: _autosummary

  export
  deserialize
  minimum_supported_serialization_version
  maximum_supported_serialization_version
  default_lowering_platform

Functions related to shape polymorphism
---------------------------------------

.. autosummary::
  :toctree: _autosummary

  symbolic_shape
  symbolic_args_specs
  is_symbolic_dim
  SymbolicScope

Constants
---------

.. data:: jax.export.minimum_supported_serialization_version

   The minimum supported serialization version; see :ref:`export-serialization-version`.

.. data:: jax.export.maximum_supported_serialization_version

   The maximum supported serialization version; see :ref:`export-serialization-version`.
