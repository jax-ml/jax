(jax-401-pallas)=
# Pallas: custom kernels in JAX

<!--* freshness: { reviewed: '2026-07-10' } *-->

Most of the time, you don't write kernels: you write `jax.numpy` programs,
and the compiler decides how to turn them into device code. Usually it does
well. But sometimes you know something the compiler doesn't: a fusion it
won't find, a memory-access pattern it won't discover (as in
FlashAttention), or a sparsity structure it can't exploit. Then the way to
get the remaining performance is to write the kernel yourself.

Pallas is JAX's kernel language: an extension of JAX that lets you write
custom kernels for GPU and TPU, with fine-grained control over the generated
code while keeping JAX tracing and the `jax.numpy` API. Kernels are written
as functions over `Ref`s in fast on-chip memory, launched over a grid with
`pl.pallas_call` (or over a mesh of cores with `pl.kernel`), and they compose
with the rest of JAX: you can `jit`, `vmap`,
and differentiate around them.

Pallas has its own extensive documentation site, which we won't duplicate
here:

**[pallas.jax.dev](https://pallas.jax.dev)**

Good entry points:

* {doc}`Quickstart for GPU </pallas/gpu/quickstart>` and
  {doc}`Quickstart for TPU </pallas/tpu/quickstart>` — writing your first custom
  kernels on GPU and TPU.
* {doc}`Refs </pallas/refs>`,
  {doc}`grids and BlockSpecs </pallas/grid_blockspec>`, and
  {doc}`pipelining </pallas/pipelining>` — the core concepts for working with
  mutable memory buffers and streaming data through on-chip memory.
* {doc}`The TPU backend guide </pallas/tpu/index>` and
  {doc}`the Mosaic GPU backend guide </pallas/gpu/index>` — per-platform
  details, lowering paths, and platform-specific features.
* The {mod}`jax.experimental.pallas` module API reference.

Pallas is experimental and changes frequently; see the
{ref}`Pallas changelog <pallas-changelog>` for recent developments.
