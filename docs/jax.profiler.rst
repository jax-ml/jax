.. currentmodule:: jax.profiler

``jax.profiler`` module
=======================

.. automodule:: jax.profiler

Tracing and time profiling
--------------------------

:doc:`/201/profiling` describes how to make use of JAX's tracing and time profiling
features.

.. autosummary::
  :toctree: _autosummary

  start_server
  start_trace
  stop_trace
  trace
  annotate_function
  TraceAnnotation
  StepTraceAnnotation
  register_subprocess


Device memory profiling
-----------------------

See :ref:`jax-201-memory-profiling` for an introduction to JAX's device memory
profiling features.

.. autosummary::
  :toctree: _autosummary

  device_memory_profile
  save_device_memory_profile
