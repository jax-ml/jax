JAX reference documentation
===========================

JAX is Autograd_ and XLA_, brought together for high-performance numerical computing and machine learning research.
It provides composable transformations of Python+NumPy programs: differentiate, vectorize,
parallelize, Just-In-Time compile to GPU/TPU, and more.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notebooks/quickstart
   notebooks/thinking_in_jax
   notebooks/Common_Gotchas_in_JAX

.. toctree::
   :maxdepth: 2

   jax-101/index

.. toctree::
   :maxdepth: 1
   :caption: Reference Documentation

   faq
   async_dispatch
   jaxpr
   notebooks/convolutions
   pytrees
   type_promotion
   errors
   glossary
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Advanced JAX Tutorials

   notebooks/autodiff_cookbook
   notebooks/vmapped_log_probs
   notebooks/neural_network_with_tfds_data
   notebooks/Custom_derivative_rules_for_Python_code
   notebooks/How_JAX_primitives_work
   notebooks/Writing_custom_interpreters_in_Jax
   notebooks/Neural_Network_and_Data_Loading
   notebooks/XLA_in_Python
   notebooks/maml
   notebooks/score_matching
   notebooks/xmap_tutorial
   multi_process

.. toctree::
   :maxdepth: 1
   :caption: Notes

   deprecation
   concurrency
   gpu_memory_allocation
   profiling
   device_memory_profiling
   rank_promotion_warning
   custom_vjp_update

.. toctree::
   :maxdepth: 2
   :caption: Developer documentation

   contributing
   developer
   jax_internal_api
   autodidax

.. toctree::
   :maxdepth: 3
   :caption: API documentation

   jax


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Autograd: https://github.com/hips/autograd
.. _XLA: https://www.tensorflow.org/xla
