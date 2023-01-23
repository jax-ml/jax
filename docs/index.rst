JAX: High-Performance Array Computing
=====================================

JAX is Autograd_ and XLA_, brought together for high-performance numerical computing.

.. grid:: 3
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Familiar API
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers. 

   .. grid-item-card:: Transformations
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.

   .. grid-item-card:: Run Anywhere
      :columns: 12 6 6 4
      :class-card: key-ideas
      :shadow: None

      The same code executes on multiple backends, including CPU, GPU, & TPU

.. note::
   JAX 0.4.1 introduces new parallelism APIs, including breaking changes to :func:`jax.experimental.pjit` and a new unified ``jax.Array`` type.
   Please see `Distributed arrays and automatic parallelization <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_ tutorial and the :ref:`jax-array-migration`
   guide for more information.


.. grid:: 3

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting Started
      :columns: 12 6 6 4
      :link: beginner-guide
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` User Guides
      :columns: 12 6 6 4
      :link: user-guides
      :link-type: ref
      :class-card: user-guides

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Developer Docs
      :columns: 12 6 6 4
      :link: contributor-guide
      :link-type: ref
      :class-card: developer-docs


Installation
------------
.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install "jax[cpu]"

    .. tab-item:: GPU (CUDA)

       .. code-block:: bash

          pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    .. tab-item:: TPU (Google Cloud)

       .. code-block:: bash

          pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

For more information about supported accelerators and platforms, and for other
installation options, see the `Install Guide`_ in the project README.

Ecosystem
---------
JAX's accelerated array operations form the foundation for a growing ecosystem of tools.

For example, several Google Research groups have published domain specific tools including:

- `Flax <https://github.com/google/flax>`__: fully-featured neural networks in JAX
- `JAXopt <https://jaxopt.github.io>`__: differentiable optimization in JAX
- `BRAX <https://github.com/google/brax>`__: differentiable physics engine in JAX

In addition, DeepMind has open-sourced an `ecosystem of libraries around
JAX <https://deepmind.com/blog/article/using-jax-to-accelerate-our-research>`__,
including:

- `Haiku <https://github.com/deepmind/dm-haiku>`__: a sonnet-inspired JAX library for neural networks
- `Optax <https://github.com/deepmind/optax>`__: gradient processing and optimization in JAX
- `RLax <https://github.com/deepmind/rlax>`__: reinforcement learning in JAX
- `chex <https://github.com/deepmind/chex>`__: tools for working effectively with JAX
   
For many more examples, check out `Awesome JAX`_, a community-maintained census of the
wider JAX ecosystem.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   installation
   notebooks/quickstart
   notebooks/thinking_in_jax
   notebooks/Common_Gotchas_in_JAX
   faq

.. toctree::
   :hidden:
   :maxdepth: 1

   jax-101/index


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Further Resources

   user_guides
   advanced_guide
   contributor_guide
   notes
   jax


.. toctree::
   :hidden:
   :maxdepth: 1

   changelog
   glossary


.. _Autograd: https://github.com/hips/autograd
.. _XLA: https://www.tensorflow.org/xla
.. _Install Guide: https://github.com/google/jax#installation
.. _Awesome JAX: https://github.com/n2cholas/awesome-jax