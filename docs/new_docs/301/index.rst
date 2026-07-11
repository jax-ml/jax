:orphan:
:nosearch:

JAX 301: advanced autodiff and extending JAX
============================================

The 101 docs introduced :func:`jax.grad`; the pages here go as deep as you
like: the machinery underneath (JVPs, VJPs, Jacobians, Hessians), how
autodiff interacts with sharding, defining your own derivative rules and
even your own types, autodiff with mutable state, and controlling the
memory/compute tradeoff of differentiation.

1. :doc:`cookbook` — the autodiff cookbook: Hessian-vector products,
   Jacobians with ``jacfwd``/``jacrev``, the ``jvp``/``vjp`` machinery and
   how it's composed, and differentiation with complex numbers.
2. :doc:`sharding-ad` — how autodiff interacts with sharding: cotangent
   shardings as a function of primal shardings, and controlling
   backward-pass communication with ``unreduced`` and ``reduced``, in both
   explicit and manual (``shard_map``) modes.
3. :doc:`custom-derivatives` — defining custom derivative rules with hijax
   primitives, the recommended approach: one primitive can carry rules for
   both modes, plus linearization, batching, and more.
4. :doc:`custom-jvp-vjp` — the classic decorators, customizing one
   differentiation mode at a time: still fully supported, and often the most
   convenient tool for simple cases.
5. :doc:`refs` — autodiff with mutable arrays: plumbing values out of
   backward passes, in-place gradient accumulation with ``with_refs``, and
   differentiating with respect to refs.
6. :doc:`remat` — gradient checkpointing with ``jax.checkpoint``: what
   autodiff saves versus recomputes, name-based policies, offloading, and
   per-function control with ``custom_remat``.
7. :doc:`hijax-types` — defining entirely new JAX types with hijax, with
   their own derivatives, batching, and sharding behavior.

.. toctree::
   :hidden:
   :maxdepth: 1

   cookbook
   sharding-ad
   custom-derivatives
   custom-jvp-vjp
   refs
   remat
   hijax-types
