:orphan:
:nosearch:

JAX 301: advanced autodiff and extending JAX
============================================

The 101 docs introduced :func:`jax.grad`; the pages here go as deep as you
like: the machinery underneath (JVPs, VJPs, Jacobians, Hessians), how
autodiff interacts with sharding, defining your own derivative rules and
even your own types, autodiff with mutable state, and controlling the
memory/compute tradeoff of differentiation.

1. :doc:`cookbook` — cooking with autodiff's fundamental ingredients, the
   ``jvp`` and ``vjp`` machinery: recipes for Hessian-vector products and
   full Jacobians, spiced up by mixing in ``vmap`` and complex numbers.
2. :doc:`vjp-objects` — the VJP object as a pytree, splitting the forward
   and backward passes into separately compiled functions run on your own
   schedule, and excluding argument values (like weights) from the saved
   state with ``saveable_args``.
3. :doc:`sharding-ad` — how autodiff interacts with sharding: cotangent
   shardings as a function of primal shardings, and controlling
   backward-pass communication with ``unreduced`` and ``reduced``, in both
   explicit and manual (``shard_map``) modes.
4. :doc:`custom-derivatives` — defining custom derivative rules with hijax
   primitives, the recommended approach: one primitive can carry rules for
   both modes, plus linearization, batching, and more.
5. :doc:`custom-jvp-vjp` — the classic decorators, customizing one
   differentiation mode at a time: still fully supported, and often the most
   convenient tool for simple cases.
6. :doc:`refs` — autodiff with mutable arrays: plumbing values out of
   backward passes, in-place gradient accumulation with ``with_refs``, and
   differentiating with respect to refs.
7. :doc:`remat` — gradient checkpointing with ``jax.checkpoint``: what
   autodiff saves versus recomputes, name-based policies, offloading, and
   per-function control with ``custom_remat``.
8. :doc:`hijax-types` — defining entirely new JAX types with hijax, with
   their own derivatives, batching, and sharding behavior.

.. toctree::
   :hidden:
   :maxdepth: 1

   cookbook
   vjp-objects
   sharding-ad
   custom-derivatives
   custom-jvp-vjp
   refs
   remat
   hijax-types
