``jax.experimental.rebindable`` module
======================================

.. automodule:: jax.experimental.rebindable

``rebindable`` stages a kernel call so that its declared hyperparameters (e.g.
tile sizes) can be inspected and rebound in an already traced program, without
retracing the program. Typical use:

1. Wrap a kernel with :func:`rebindable`, declaring its hyperparameters.
2. Trace the program with ``jax.jit(f).trace(*args)`` and list its call sites
   with :func:`extract_rebindables`. Each :class:`RebindableSite` exposes the
   ``rebindable`` that runs (function, hyperparameters, operand types, ``key``)
   and the ``ctx`` it was called in (mesh, ``xla_metadata``, ...).
3. Benchmark a site alone with :meth:`RebindableSite.call` on operands of its
   types, e.g. on a submesh where each shard sees the site's per-shard types.
4. Apply the winners with :func:`rebind` and lower the result; only
   the rebound kernels are retraced.

Rebindables are never differentiated (call them from the rules of a
:func:`jax.custom_vjp` instead) and cannot be nested.

.. currentmodule:: jax.experimental.rebindable

API
---

.. autosummary::
  :toctree: _autosummary

  rebindable
  extract_rebindables
  rebind
  Rebindable
  RebindableSite
