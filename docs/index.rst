JAX: High performance array computing
=====================================

.. raw:: html
   :file: hero.html

.. grid:: 3
   :class-container: product-offerings
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Familiar API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX provides a familiar NumPy-style API for ease of adoption by researchers and engineers.

   .. grid-item-card:: Transformations
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      JAX includes composable function transformations for compilation, batching, automatic differentiation, and parallelization.

   .. grid-item-card:: Run anywhere
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      The same code executes on multiple backends, including CPU, GPU, & TPU

.. grid:: 3
    :class-container: color-cards

    .. grid-item-card:: :material-regular:`laptop_chromebook;2em` Installation
      :columns: 12 6 6 4
      :link: installation
      :link-type: ref
      :class-card: installation

    .. grid-item-card:: :material-regular:`rocket_launch;2em` Getting started
      :columns: 12 6 6 4
      :link: jax-101
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` API reference
      :columns: 12 6 6 4
      :link: jax
      :link-type: doc
      :class-card: jax-101

If you're looking to use JAX to train neural networks, check out the `JAX AI
Stack`_!

Documentation
-------------

The documentation is organized into levels. Start at 101 and read in order,
or jump to the level that matches what you're trying to do:

* :doc:`JAX 101 <101/index>` — **expressing computations**: arrays and
  :mod:`jax.numpy`, transformations (:func:`jax.grad`, :func:`jax.vmap`),
  how tracing works, pytrees, random numbers, and state.

* :doc:`JAX 201 <201/index>` — **performance and scaling**: compiling with
  :func:`jax.jit`, ahead-of-time compilation, control flow, data placement,
  sharding and automatic parallelization, per-device programming with
  ``shard_map``, callbacks, and the diagnostics toolbox: profiling,
  debugging, compilation time, numerical precision, and GPU memory.

* :doc:`JAX 301 <301/index>` — **advanced autodiff and extending JAX**: the
  autodiff cookbook (JVPs, VJPs, Jacobians, Hessians), autodiff with
  sharding, custom derivative rules, autodiff with mutable state, gradient
  checkpointing, and defining new JAX types.

* :doc:`JAX 401 <401/index>` — **kernels and FFI**: writing custom GPU and
  TPU kernels with Pallas, and calling external code through the foreign
  function interface.

* :doc:`JAX 501 <501/index>` — **systems topics**: multi-controller JAX
  across many hosts, distributed data loading, fault tolerance, exporting
  and serialization, the persistent compilation cache, and transfer guards.

* :doc:`JAX 601 <601/index>` — **internals**: the jaxpr language,
  primitives, and Autodidax, which builds JAX's core from scratch.

Already know JAX? See :doc:`whats-new` for the features these docs cover
for the first time.

Ecosystem
---------
JAX itself is narrowly-scoped and focuses on efficient array operations & program
transformations. Built around JAX is an evolving ecosystem of machine learning and
numerical computing tools; the following is just a small sample of what is out there:

.. grid:: 2
    :class-container: ecosystem-grid

    .. grid-item:: :material-outlined:`hub;2em` **Neural networks**

       - Flax_
       - Equinox_
       - Keras_

    .. grid-item:: :material-regular:`show_chart;2em` **Optimizers & solvers**

       - Optax_
       - Optimistix_
       - Lineax_
       - Diffrax_

    .. grid-item:: :material-outlined:`storage;2em` **Data loading**

       - Grain_
       - `TensorFlow Datasets`_
       - `Hugging Face Datasets`_

    .. grid-item:: :material-regular:`construction;2em` **Miscellaneous tools**

       - Orbax_
       - Chex_

    .. grid-item:: :material-regular:`lan;2em` **Probabilistic programming**

       - Blackjax_
       - Numpyro_
       - PyMC_

    .. grid-item:: :material-regular:`bar_chart;2em` **Probabilistic modeling**

       - `TensorFlow Probability`_
       - Distrax_

    .. grid-item:: :material-outlined:`animation;2em` **Physics & simulation**

       - `JAX MD`_
       - Brax_

    .. grid-item:: :material-regular:`language;2em` **LLMs**

       - MaxText_
       - AXLearn_
       - Levanter_
       - EasyLM_
       - Marin_


Many more JAX-based libraries have been developed; the community-run `Awesome JAX`_ page
maintains an up-to-date list.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Documentation

   whats-new
   101/index
   201/index
   301/index
   401/index
   501/index
   601/index
   notebooks/Common_Gotchas_in_JAX
   pallas/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Resources, guides, and references

   advanced_guides
   jax
   contributor_guide
   extensions
   notes
   about


.. toctree::
   :hidden:
   :maxdepth: 1

   faq
   changelog
   glossary

.. toctree::
   :hidden:
   :maxdepth: 2

   config_options

.. _Awesome JAX: https://github.com/n2cholas/awesome-jax
.. _AXLearn: https://github.com/apple/axlearn
.. _Blackjax: https://blackjax-devs.github.io/blackjax/
.. _Brax: https://github.com/google/brax/
.. _Chex: https://chex.readthedocs.io/
.. _Diffrax: https://docs.kidger.site/diffrax/
.. _Distrax: https://github.com/google-deepmind/distrax
.. _EasyLM: https://github.com/young-geng/EasyLM
.. _Equinox: https://docs.kidger.site/equinox/
.. _Flax: https://flax.readthedocs.io/
.. _Grain: https://github.com/google/grain
.. _Hugging Face Datasets: https://huggingface.co/docs/datasets/
.. _JAX MD: https://jax-md.readthedocs.io/
.. _JAX AI Stack: https://docs.jaxstack.ai/en/latest/getting_started.html
.. _Keras: https://keras.io/
.. _Levanter: https://github.com/stanford-crfm/levanter
.. _Marin: https://github.com/marin-community/marin
.. _Lineax: https://github.com/patrick-kidger/lineax
.. _MaxText: https://github.com/google/maxtext/
.. _Numpyro: https://num.pyro.ai/en/latest/index.html
.. _Optax: https://optax.readthedocs.io/
.. _Optimistix: https://github.com/patrick-kidger/optimistix
.. _Orbax: https://orbax.readthedocs.io/
.. _PyMC: https://www.pymc.io/
.. _TensorFlow Datasets: https://www.tensorflow.org/datasets
.. _TensorFlow Probability: https://www.tensorflow.org/probability
