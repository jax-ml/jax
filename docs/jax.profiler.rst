.. currentmodule:: jax.profiler

jax.profiler module
===================

.. automodule:: jax.profiler

Tracing and time profiling
--------------------------

:doc:`profiling` describes how to make use of JAX's tracing and time profiling
features.

.. autosummary::
  :toctree: _autosummary

  start_server
  trace_function
  TraceContext


Device memory profiling
-----------------------

See :doc:`device_memory_profiling` for an introduction to JAX's device memory
profiling features.

.. autosummary::
  :toctree: _autosummary

  device_memory_profile
  save_device_memory_profile
  