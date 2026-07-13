:orphan:
:nosearch:

JAX 501: systems topics
=======================

Everything so far assumed one Python process driving some devices. These
pages are about running JAX as part of a larger system: many processes and
hosts, long-lived and restartable jobs, artifacts that outlive the process
that created them, and the machinery that keeps it all fast and safe.

1. :doc:`multiprocess` — multi-controller JAX: running one process per host,
   process-spanning meshes and arrays, runtime-level pipeline parallelism
   with ``jax.device_put``, and building global arrays from per-process
   data.
2. :doc:`data-loading` — distributed data loading: getting each batch's
   shards onto the right hosts, for data-parallel and model-parallel
   workloads.
3. :doc:`fault-tolerance` — fault-tolerant distributed JAX: surviving
   machine failures with ``jax.live_devices``, barrier semantics, and
   recovery, with worked training examples.
4. :doc:`export` — exporting and serializing staged-out computations for
   later or cross-platform execution, and :doc:`shape-polymorphism` —
   exporting with symbolic shapes so one artifact serves many input sizes.
5. :doc:`compilation-cache` — the persistent compilation cache: skipping
   recompilation across process restarts and sharing compiled artifacts
   across nodes.
6. :doc:`transfer-guard` — logging or disallowing unintended host-device
   transfers.

Smaller notes
-------------

A few systems topics are small enough to cover right here.

**Concurrency and threads.** JAX has limited support for Python concurrency:
it's fine to call JAX APIs like :func:`jax.jit` or :func:`jax.grad` from
multiple Python threads, but you must not use threads to manipulate JAX
trace values *inside* a traced function — the likely outcome is a mysterious
error. In multi-controller JAX (:doc:`multiprocess`) threading has a further
hazard: every process must enqueue the same operations in the same order on
a given device, and threads can schedule work in different orders in
different processes, causing non-deterministic crashes. The
:func:`jax.thread_guard` context manager helps detect this: once set, an
error is raised if a JAX operation is issued from a thread other than the
one where the guard was set.

**Multi-process coordination helpers.** The
:mod:`jax.experimental.multihost_utils` module collects small utilities for
multi-controller programs: ``sync_global_devices`` (a named cross-process
barrier), ``broadcast_one_to_all`` (make process 0's value everyone's
value), ``process_allgather`` (gather a value from every process),
``assert_equal`` (check that a value agrees across processes), and
``host_local_array_to_global_array`` / ``global_array_to_host_local_array``
(convert between per-host and global views of arrays).

**Compatibility policies for long-lived artifacts.** Serialized artifacts
outlive the process that made them, so it's worth knowing what's promised:
exported modules have explicit compatibility windows (see the compatibility
guarantees in :doc:`export`); persistent compilation cache entries make no
cross-version promises, but their keys include the jaxlib version, so
version changes cause cache misses rather than misbehavior; and JAX's
general API stability rules are described in the
:doc:`API compatibility policy </api_compatibility>`.

.. toctree::
   :hidden:
   :maxdepth: 1

   multiprocess
   data-loading
   fault-tolerance
   export
   shape-polymorphism
   compilation-cache
   transfer-guard
