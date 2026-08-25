(whats-new)=
# Newly documented

Features documented for the first time. See also the
{doc}`changelog`.

## August 2026

- **{doc}`Autodiff and sharding <301/sharding-ad>`.** How cotangent shardings
  follow from primal shardings in explicit mode, and controlling backward-pass
  communication with *unreduced* and *reduced* shardings.

- **{ref}`Refs: mutable arrays <jax-101-refs>`.** {func}`jax.new_ref
  <jax.ref.new_ref>` creates an array *ref* that can be read and written in
  place, composing with transformations. In-place updates under `jit` in
  {ref}`jax-201-jit-refs`; autodiff with refs in {ref}`jax-301-refs`.

- **{doc}`First-class VJP objects <301/vjp-objects>`.** The callable returned
  by {func}`jax.vjp` is a pytree. A recipe for getting the forward and
  backward passes as separate functions.

- **{ref}`saveable_args <jax-301-saveable-args>`** on `jax.vjp`. Exclude
  argument values (like weights) from what a VJP saves.

- **{doc}`Custom derivatives with hijax primitives <301/custom-derivatives>`.**
  One primitive can carry rules for both differentiation modes and batching.
  More capable alternative to `jax.custom_vjp` and `jax.custom_jvp`.

- **{ref}`Backward-pass logging <jax-301-bwd-logging>`.** Plumb data out of
  backward passes, e.g. for gradient diagnostics.

- **{ref}`Structured residuals <jax-301-structured-residuals>`.** Organize
  what the forward pass saves for the backward pass.

- **{doc}`New JAX types with hijax <301/hijax-types>`.** Define new types
  with their own tangent types, batching behaviors, and sharding. Consume
  with your own hijax primitives.

- **{doc}`Fault tolerance <501/fault-tolerance>`.** Surviving machine
  failures in multi-host jobs with `jax.live_devices`.

- **{doc}`Compiler control <201/controlling-xla>`.** Compilation effort
  levels per `jit`-compiled function, and per-operation XLA metadata.

- **{ref}`Complex numbers and differentiation <jax-301-complex>`.** A better
  explanation of what JVPs and VJPs mean over $\mathbb{C}$, and where
  `grad`'s conjugation convention comes from.
