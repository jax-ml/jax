JAX: High performance array computing
=====================================

.. raw:: html

   <script>
      /* Along with some CSS settings in style.css (look for `body:has(.hero)`)
         this will ensure that the menu sidebar is hidden on the main page. */
      if (window.innerWidth >= 960) {
         document.getElementById("__primary").checked = true;
      }
   </script>


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
      :link: beginner-guide
      :link-type: ref
      :class-card: getting-started

    .. grid-item-card:: :material-regular:`library_books;2em` User guides
      :columns: 12 6 6 4
      :link: user-guides
      :link-type: ref
      :class-card: user-guides

If you're looking to use JAX to train neural networks, start with the
`JAX AI Stack Tutorials`_, and then check out the `JAX AI Stack Examples`_
to see how JAX models can be implemented using the Flax_ framework.

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
   quickstart

.. toctree::
   :hidden:
   :maxdepth: 1

   tutorials

   notebooks/Common_Gotchas_in_JAX

   faq

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: More guides/resources

   user_guides
   advanced_guide
   contributor_guide
   extensions
   notes
   jax
   about


.. toctree::
   :hidden:
   :maxdepth: 1

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
.. _JAX AI Stack Tutorials: https://docs.jaxstack.ai/en/latest/tutorials.html
.. _JAX AI Stack Examples: https://docs.jaxstack.ai/en/latest/examples.html
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
