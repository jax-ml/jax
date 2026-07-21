:orphan:
:nosearch:

JAX documentation
=================

JAX is a Python library for high-performance numerical computing and machine
learning: a NumPy-like interface, function transformations for
differentiation and batching, and compilation to CPU, GPU, and TPU at any
scale.

The documentation is organized into levels. Start at 101 and read in order,
or jump to the level that matches what you're trying to do:

* :doc:`JAX 101 <101/index>` — **expressing computations**: arrays and
  :mod:`jax.numpy`, transformations (:func:`jax.grad`, :func:`jax.vmap`),
  how tracing works, pytrees, random numbers, and state.

* :doc:`JAX 201 <201/index>` — **performance and scaling**: compiling with
  :func:`jax.jit`, ahead-of-time compilation, control flow, data placement,
  sharding and automatic parallelization, per-device programming with
  ``shard_map``, callbacks, and the diagnostics toolbox: profiling,
  debugging, compilation time, numerical precision, and GPU memory.

* :doc:`JAX 301 <301/index>` — **advanced autodiff and extending JAX**: the
  autodiff cookbook (JVPs, VJPs, Jacobians, Hessians), autodiff with
  sharding, custom derivative rules, autodiff with mutable state, gradient
  checkpointing, and defining new JAX types.

* :doc:`JAX 401 <401/index>` — **kernels and FFI**: writing custom GPU and
  TPU kernels with Pallas, and calling external code through the foreign
  function interface.

* :doc:`JAX 501 <501/index>` — **systems topics**: multi-controller JAX
  across many hosts, distributed data loading, fault tolerance, exporting
  and serialization, the persistent compilation cache, and transfer guards.

.. toctree::
   :hidden:
   :maxdepth: 1

   101/index
   201/index
   301/index
   401/index
   501/index
