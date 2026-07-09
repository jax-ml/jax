:orphan:
:nosearch:

JAX 201: performance and scaling
================================

The :doc:`JAX 101 docs </new_docs/101/index>` covered how to *express*
computations: arrays and :mod:`jax.numpy`, transformations like
:func:`jax.grad` and :func:`jax.vmap`, pytrees, randomness, and state. None of
that made anything fast. These pages are about performance: compiling your
code, measuring it, scaling it from one chip to thousands, and tuning it.

**The core arc**, meant to be read in order, runs from compiling on one
device to programming many:

1. :doc:`jit` — just-in-time compilation with :func:`jax.jit`: how it works,
   tracing and retracing, static arguments, caching, in-place updates with
   refs and buffer donation, and asynchronous dispatch.
2. :doc:`aot` — ahead-of-time lowering and compilation, for inspecting or
   controlling each stage of the ``jit`` pipeline.
3. :doc:`control-flow` — expressing conditionals and loops so they can be
   compiled: constraints on Python control flow under ``jit``, and the
   structured alternatives like ``lax.cond`` and ``lax.scan``.
4. :doc:`placement` — where arrays live: meshes as the unit of placement
   (single-device meshes included), committed vs. uncommitted arrays, and
   moving data between meshes.
5. :doc:`sharding` — distributed arrays and automatic parallelization: the
   global-view programming model that scales one program to many devices,
   plus its zoomed-in sibling, device-local layout.
6. :doc:`shard-map` — the full tutorial for ``sharding``'s manual mode:
   per-device programming with explicit collectives, for complete control
   over how computation and communication are partitioned.
7. :doc:`callbacks` — calling back to host Python from compiled code with
   ``pure_callback`` and ``io_callback``, including how callbacks interact
   with sharded data.

**Diagnostics and tuning** pages are there for when you need them, in any
order:

8. :doc:`profiling` — how to measure: benchmarking pitfalls, capturing and
   reading profiler traces (including of distributed code), and device
   memory profiling.
9. :doc:`debugging` — printing and inspecting runtime values inside compiled
   code, and the debugging flags every JAX user should know.
10. :doc:`slow-compilation` — a field guide to diagnosing slow tracing and
    compilation: diagnostic flags, reading the logs, and the Python patterns
    that defeat JAX's caches.
11. :doc:`precision` — controlling matmul precision: dot algorithms, the
    classic ``Precision`` levels, and global defaults.
12. :doc:`controlling-xla` — steering the compiler directly: XLA flags per
    function or process-wide, and attaching metadata to operations.
13. :doc:`gpu-memory` — how JAX allocates GPU memory, and what to do about
    out-of-memory failures.

A couple of performance topics — computation/communication overlap and
memory spaces — will be added here as those parts of JAX stabilize.

.. toctree::
   :hidden:
   :maxdepth: 1

   jit
   aot
   control-flow
   placement
   sharding
   shard-map
   callbacks
   profiling
   debugging
   slow-compilation
   precision
   controlling-xla
   gpu-memory
