
``jax.debug`` module
====================

.. currentmodule:: jax.debug

.. automodule:: jax.debug

Runtime value debugging utilities
---------------------------------

:doc:`debugging/print_breakpoint` describes how to make use of JAX's runtime value
debugging features.

.. autosummary::
  :toctree: _autosummary

  callback
  print
  breakpoint

Sharding debugging utilities
----------------------------

Functions that enable inspecting and visualizing array shardings inside (and outside)
staged functions.

.. autosummary::
  :toctree: _autosummary

  inspect_array_sharding
  visualize_array_sharding
  visualize_sharding
