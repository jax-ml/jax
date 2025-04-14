.. _user-guides:

User guides
===========

User guides are deeper dives into particular topics within JAX
that become relevant as your JAX project matures into larger
or deployed codebases.

.. toctree::
   :maxdepth: 1
   :caption: Debugging and performance

   notebooks/thinking_in_jax
   profiling
   device_memory_profiling
   debugging/index
   gpu_performance_tips
   persistent_compilation_cache

.. toctree::
   :maxdepth: 1
   :caption: Interfaces

   pytrees
   errors
   aot
   export/index
   transfer_guard

.. toctree::
   :maxdepth: 1
   :caption: Custom operations

   pallas/index
   ffi

.. toctree::
   :caption: Example applications
   :maxdepth: 1

   notebooks/neural_network_with_tfds_data
   notebooks/Neural_Network_and_Data_Loading
   notebooks/vmapped_log_probs
