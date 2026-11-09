.. _jax-601:

JAX 601: internals
==================

How JAX works inside. These pages are for the curious, for contributors,
and for anyone extending JAX at the lowest level; nothing here is needed to
*use* JAX.

1. :doc:`jaxpr` — the jaxpr language: the intermediate representation that
   tracing produces, its grammar, and how to read it.
2. :doc:`jax-primitives` — how primitive operations work: what JAX requires
   of a primitive, and defining new ones with ``jax.extend.core.Primitive``.
3. :doc:`../autodidax` — JAX core from scratch: build tracing, jaxprs,
   autodiff, and jit in pure Python, one layer at a time.
4. :doc:`../autodidax2_part1` — Autodidax2, part 1: a from-scratch rebuild
   reflecting JAX's current internals.

.. toctree::
   :hidden:
   :maxdepth: 1

   jaxpr
   jax-primitives
   ../autodidax
   ../autodidax2_part1
