
jax.ops package
=================

.. currentmodule:: jax.ops

.. automodule:: jax.ops

.. _syntactic-sugar-for-ops:

Indexed update operators
------------------------

JAX is intended to be used with a functional style of programming, and
does not support NumPy-style indexed assignment directly. Instead, JAX provides
alternative pure functional operators for indexed updates to arrays.

JAX array types have a property ``at``, which can be used as
follows (where ``idx`` is a NumPy index expression).

=========================  ===================================================
Alternate syntax           Equivalent in-place expression
=========================  ===================================================
``x.at[idx].get()``        ``x[idx]``
``x.at[idx].set(y)``       ``x[idx] = y``
``x.at[idx].add(y)``       ``x[idx] += y``
``x.at[idx].multiply(y)``  ``x[idx] *= y``
``x.at[idx].divide(y)``    ``x[idx] /= y``
``x.at[idx].power(y)``     ``x[idx] **= y``
``x.at[idx].min(y)``       ``x[idx] = np.minimum(x[idx], y)``
``x.at[idx].max(y)``       ``x[idx] = np.maximum(x[idx], y)``
=========================  ===================================================

None of these expressions modify the original `x`; instead they return
a modified copy of `x`. However, inside a :py:func:`jit` compiled function,
expressions like ``x = x.at[idx].set(y)`` are guaranteed to be applied in-place.

By default, JAX assumes that all indices are in-bounds. There is experimental
support for giving more precise semantics to out-of-bounds indexed accesses,
via the ``mode`` parameter to functions such as ``get`` and ``set``. Valid
values for ``mode`` include ``"clip"``, which means that out-of-bounds indices
will be clamped into range, and ``"fill"``/``"drop"``, which are aliases and
mean that out-of-bounds reads will be filled with a scalar ``fill_value``,
and out-of-bounds writes will be discarded.


Indexed update functions (deprecated)
-------------------------------------

The following functions are aliases for the ``x.at[idx].set(y)``
style operators. Use the ``x.at[idx]`` operators instead.

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

    segment_max
    segment_min
    segment_prod
    segment_sum
