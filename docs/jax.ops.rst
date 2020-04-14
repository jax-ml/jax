
jax.ops package
=================

.. currentmodule:: jax.ops

.. automodule:: jax.ops


Indexed update operators
------------------------

JAX is intended to be used with a functional style of programming, and hence
does not support NumPy-style indexed assignment directly. Instead, JAX provides
pure alternatives, namely :func:`jax.ops.index_update` and its relatives.

.. autosummary::
  :toctree: _autosummary

    index
    index_update
    index_add
    index_mul
    index_min
    index_max

Other operators
---------------

.. autosummary::
  :toctree: _autosummary

    segment_sum
