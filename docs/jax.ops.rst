``jax.ops`` module
==================

.. currentmodule:: jax.ops

.. automodule:: jax.ops

.. _syntactic-sugar-for-ops:

The functions ``jax.ops.index_update``, ``jax.ops.index_add``, etc., which were
deprecated in JAX 0.2.22, have been removed. Please use the
:attr:`jax.numpy.ndarray.at` property on JAX arrays instead.

Segment reduction operators
---------------------------

.. autosummary::
  :toctree: _autosummary

    segment_max
    segment_min
    segment_prod
    segment_sum
