
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


Syntactic sugar for indexed update operators
--------------------------------------------

JAX also provides an alternate syntax for these indexed update operators.
Specifically, JAX ndarray types have a property ``at``, which can be used as
follows (where ``idx`` can be an arbitrary index expression).

====================  ===================================================
Alternate syntax      Equivalent expression
====================  ===================================================
``x.at[idx].set(y)``  ``jax.ops.index_update(x, jax.ops_index[idx], y)``
``x.at[idx].add(y)``  ``jax.ops.index_add(x, jax.ops_index[idx], y)``
``x.at[idx].mul(y)``  ``jax.ops.index_mul(x, jax.ops_index[idx], y)``
``x.at[idx].min(y)``  ``jax.ops.index_min(x, jax.ops_index[idx], y)``
``x.at[idx].max(y)``  ``jax.ops.index_max(x, jax.ops_index[idx], y)``
====================  ===================================================

Note that none of these expressions modify the original `x`; instead they return
a modified copy of `x`.

Other operators
---------------

.. autosummary::
  :toctree: _autosummary

    segment_sum
