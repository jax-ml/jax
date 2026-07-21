:orphan:
:nosearch:

JAX 401: kernels and FFI
========================

Sometimes the compiler isn't enough: you want to write the device code
yourself, or call code that lives outside JAX entirely. This section covers
both escape hatches.

1. :doc:`pallas` — Pallas, JAX's kernel language for writing custom GPU and
   TPU kernels; mostly a map to the extensive Pallas documentation at
   `pallas.jax.dev <https://pallas.jax.dev>`_.
2. :doc:`cute-dsl` — writing high-performance GPU kernels with NVIDIA's
   CuTe DSL and calling them from JAX.
3. :doc:`ffi` — the foreign function interface: wrapping external C++/CUDA
   code as a JAX operation, and teaching it to work with ``jit``, ``vmap``,
   ``grad``, and sharding.

.. toctree::
   :hidden:
   :maxdepth: 1

   pallas
   cute-dsl
   ffi
