(whats-new)=
# Newly documented

Features documented for the first time. See also the
{doc}`changelog`.

## August 2026

- **Autodiff and sharding.** How cotangent shardings follow from primal
  shardings in explicit mode, and controlling backward-pass communication with
  *unreduced* and *reduced* shardings. See {doc}`301/sharding-ad`.

- **Refs: mutable arrays.** {func}`jax.new_ref <jax.ref.new_ref>` creates
  an array *ref* that can be read and written in place, composing with
  transformations. Basics in {ref}`jax-101-refs`; in-place updates under
  `jit` in {ref}`jax-201-jit-refs`; autodiff with refs in
  {ref}`jax-301-refs`.

- **First-class VJP objects.** The callable returned by {func}`jax.vjp` is
  a pytree. A recipe for getting the forward and backward passes as separate
  functions. See {doc}`301/vjp-objects`.

- **`saveable_args`** on `jax.vjp`. Exclude argument values (like weights) from
  what a VJP saves. See {ref}`jax-301-saveable-args`.

- **Custom derivatives with hijax primitives.** One primitive can carry
  rules for both differentiation modes and batching. More capable alternative
  to `jax.custom_vjp` and `jax.custom_jvp`.  See
  {doc}`301/custom-derivatives`.

- **Backward-pass logging.** Plumb data out of backward passes, e.g. for
  gradient diagnostics. See {ref}`jax-301-bwd-logging`.

- **Structured residuals.** Organize what the forward pass saves for the
  backward pass. See {ref}`jax-301-structured-residuals`.

- **New JAX types with hijax.** Define new types with their own tangent
  types, batching behaviors, and sharding. Consume with your own hijax
  primitives.  See {doc}`301/hijax-types`.

- **Fault tolerance.** Surviving machine failures in multi-host jobs with
  `jax.live_devices`. See {doc}`501/fault-tolerance`.

- **Compiler control.** Compilation effort levels per `jit`-compiled
  function, and per-operation XLA metadata. See
  {ref}`jax-201-compiler-flags` and {ref}`jax-201-xla-metadata`.

- **Complex numbers and differentiation.** A better explanation of what JVPs
  and VJPs mean over $\mathbb{C}$, and where `grad`'s conjugation convention
  comes from. See {ref}`jax-301-complex`.
