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
   Training a Simple Neural Network, with PyTorch Data Loading <https://github.com/google/jax/blob/master/docs/notebooks/Neural_Network_and_Data_Loading.ipynb>


.. toctree::
   :maxdepth: 1
   :caption: Advanced JAX Tutorials

   notebooks/Common_Gotchas_in_JAX
   notebooks/XLA_in_Python
   notebooks/How_JAX_primitives_work
   notebooks/Writing_custom_interpreters_in_Jax.ipynb
   Training a Simple Neural Network, with Tensorflow Datasets Data Loading <https://github.com/google/jax/blob/master/docs/notebooks/neural_network_with_tfds_data.ipynb>
   notebooks/maml
   notebooks/score_matching
   notebooks/vmapped_log_probs

.. toctree::
   :maxdepth: 1
   :caption: Notes

   async_dispatch
   concurrency
   gpu_memory_allocation
   profiling
   rank_promotion_warning

.. toctree::
   :maxdepth: 2
   :caption: Developer documentation

   developer

.. toctree::
   :maxdepth: 3
   :caption: API documentation

   jax


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
