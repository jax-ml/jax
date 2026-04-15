.. _jax:

.. This target is required to prevent the Sphinx build error "Unknown target name: jax".
.. The custom directive list_config_options imports JAX to extract real configuration
.. data, which causes Sphinx to look for a target named "jax". This dummy target
.. satisfies that requirement while allowing the actual JAX import to work.

Configuration Options
=====================

JAX provides various configuration options to customize its behavior. These options control everything from numerical precision to debugging features.

How to Use Configuration Options
--------------------------------

JAX configuration options can be set in several ways:

1. **Environment variables** (set before running your program):

   .. code-block:: bash

      export JAX_ENABLE_X64=True
      python my_program.py

2. **Runtime configuration** (in your Python code):

   .. code-block:: python

      import jax

      # Sets a flag's value
      jax.config.enable_x64.set(True)

      # Gets a flag's value
      jax.config.enable_x64.value

  Older versions of JAX expressed this as:

   .. code-block:: python

      import jax

      # Sets a flag's value (DEPRECATED)
      jax.config.update('jax_enable_x64', True)

      # Gets a flag's value (DEPRECATED). The deprecate flag getter has a jax_
      # prefix that is not present in the updated API.
      jax.config.jax_enable_x64

  Prefer the undeprecated forms for new code.

  Some configuration options can also be set locally for a single thread using a context manager:

   .. code-block:: python

      import jax

      with jax.config.enable_x64(True):
          # Code runs with x64 enabled
          pass

   A thread-local value, if set, takes precedence over the global value.

3. **Command-line flags** (using Abseil):

   .. code-block:: python

      # In your code:
      import jax
      jax.config.parse_flags_with_absl()

   .. code-block:: bash

      # When running:
      python my_program.py --jax_enable_x64=True

Common Configuration Options
----------------------------

Here are some of the most frequently used configuration options:

- ``jax_enable_x64`` -- Enable 64-bit floating-point precision
- ``jax_disable_jit`` -- Disable JIT compilation for debugging
- ``jax_debug_nans`` -- Check for and raise errors on NaNs
- ``jax_platforms`` -- Control which backends (CPU/GPU/TPU) JAX will initialize
- ``jax_numpy_rank_promotion`` -- Control automatic rank promotion behavior
- ``jax_default_matmul_precision`` -- Set default precision for matrix multiplication operations

.. raw:: html

   <div style="margin-top: 30px;"></div>

All Configuration Options
-------------------------

Below is a complete list of all available JAX configuration options:

.. list_config_options::
