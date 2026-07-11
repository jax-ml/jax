---
nosearch: true
---

(jax-401-pallas)=
# Pallas: custom kernels in JAX

<!--* freshness: { reviewed: '2026-07-10' } *-->

Most of the time, you don't write kernels: you write `jax.numpy` programs,
and the compiler decides how to turn them into device code. Usually it does
well. But sometimes you know something the compiler doesn't — a fusion it
won't find, a memory-access pattern it won't discover (think
FlashAttention), a sparsity structure it can't exploit — and the way to get
that last factor of performance is to take over and write the kernel
yourself.

Pallas is JAX's kernel language: an extension of JAX that lets you write
custom kernels for GPU and TPU, with fine-grained control over the generated
code while keeping JAX tracing and the `jax.numpy` API. Kernels are written
as functions over `Ref`s in fast on-chip memory, launched over a grid with
`pl.kernel`, and they compose with the rest of JAX — you can `jit`, `vmap`,
and differentiate around them.

Pallas has its own extensive documentation site, which we won't duplicate
here:

**[pallas.jax.dev](https://pallas.jax.dev)**

Good entry points:

* {doc}`Pallas quickstart </pallas/quickstart>` — kernels, `Ref`s, grids,
  and `BlockSpec`s, on both GPU and TPU.
* {doc}`Pipelining </pallas/pipelining>` and
  {doc}`grids and BlockSpecs </pallas/grid_blockspec>` — the core concepts
  for expressing how data is carved up and streamed through on-chip memory.
* {doc}`The TPU backend guide </pallas/tpu/index>` and
  {doc}`the Mosaic GPU backend guide </pallas/gpu/index>` — per-platform
  details, lowering paths, and platform-specific features.
* The {mod}`jax.experimental.pallas` module API reference.

Note that Pallas is experimental and changes frequently: expect sharp
edges, and see the Pallas changelog for recent developments.
