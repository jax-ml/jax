.. _advanced_guides:

Resources and Advanced Guides
=============================

This section contains examples and tutorials on more advanced topics,
such as multi-core computation, automatic differentiation, and custom
operations.

.. toctree::
   :caption: Parallel computation
   :maxdepth: 1

   notebooks/Distributed_arrays_and_automatic_parallelization
   notebooks/explicit-sharding
   notebooks/shard_map
   notebooks/layout
   notebooks/host-offloading
   multi_process
   distributed_data_loading
   notebooks/colocated-python

.. toctree::
   :caption: Machine learning
   :maxdepth: 1

   the-training-cookbook

.. toctree::
   :caption: Automatic differentiation
   :maxdepth: 1

   notebooks/autodiff_cookbook
   notebooks/Custom_derivative_rules_for_Python_code
   notebooks/autodiff_remat
   advanced-autodiff

.. toctree::
   :maxdepth: 1
   :caption: Errors and debugging

   errors
   debugging
   debugging/index
   debugging/flags
   transfer_guard

.. toctree::
   :maxdepth: 1
   :caption: Pytrees

   custom_pytrees

.. toctree::
   :maxdepth: 1
   :caption: Performance optimizations

   persistent_compilation_cache
   buffer_donation
   gpu_performance_tips

.. toctree::
   :maxdepth: 1
   :caption: Performance benchmarking and profiling

   benchmarking
   profiling
   device_memory_profiling

.. toctree::
   :caption: Non-functional programming
   :maxdepth: 1

   array_refs

.. toctree::
   :caption: External Callbacks
   :maxdepth: 1

   external-callbacks

.. toctree::
   :caption: FFI
   :maxdepth: 1

   ffi

.. toctree::
   :caption: Modeling workflows
   :maxdepth: 1

   gradient-checkpointing
   aot
   export/index

.. toctree::
   :caption: Example applications
   :maxdepth: 1

   notebooks/neural_network_with_tfds_data
   notebooks/Neural_Network_and_Data_Loading
   notebooks/vmapped_log_probs

.. toctree::
   :caption: Deep dives
   :maxdepth: 1

   notebooks/convolutions
   xla_flags
   jax-primitives
   jaxpr
