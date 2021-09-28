jax.lib package
===============
The `jax.lib` package is a set of internal tools and types for bridging between
JAX's Python frontend and its XLA backend.

jax.lib.xla_bridge
------------------

.. currentmodule:: jax.lib.xla_bridge

.. autosummary::
  :toctree: _autosummary

  constant
  default_backend
  device_count
  get_backend
  get_compile_options
  local_device_count
  process_index

jax.lib.xla_client
------------------

.. currentmodule:: jaxlib.xla_client

.. autosummary::
   :toctree: _autosummary

jax.lib.xla_extension
---------------------

.. currentmodule:: jaxlib.xla_extension

.. autosummary::
   :toctree: _autosummary

   Device
   CpuDevice
   GpuDevice
   TpuDevice
