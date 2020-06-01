JAX reference documentation
===============================

Composable transformations of Python+NumPy programs: differentiate, vectorize,
JIT to GPU/TPU, and more.

For an introduction to JAX, start at the
`JAX GitHub page <https://github.com/google/jax>`_.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   notebooks/quickstart
   notebooks/autodiff_cookbook
   notebooks/vmapped_log_probs
   Training a Simple Neural Network, with Tensorflow Datasets Data Loading <https://github.com/google/jax/blob/master/docs/notebooks/neural_network_with_tfds_data.ipynb>


.. toctree::
   :maxdepth: 1
   :caption: Advanced JAX Tutorials

   notebooks/Common_Gotchas_in_JAX
   notebooks/Custom_derivative_rules_for_Python_code
   notebooks/JAX_pytrees
   notebooks/XLA_in_Python
   notebooks/How_JAX_primitives_work
   notebooks/Writing_custom_interpreters_in_Jax.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Notes

   CHANGELOG
   faq
   jaxpr
   async_dispatch
   concurrency
   gpu_memory_allocation
   profiling
   pytree_arguments
   rank_promotion_warning
   type_promotion

.. toctree::
   :maxdepth: 2
   :caption: Developer documentation

   developer
   jax_internal_api

.. toctree::
   :maxdepth: 3
   :caption: API documentation

   jax


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
