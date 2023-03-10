``jax.example_libraries`` module
================================

JAX provides some small, experimental libraries for machine learning. These
libraries are in part about providing tools and in part about serving as
examples for how to build such libraries using JAX. Each one is only <300 source
lines of code, so take a look inside and adapt them as you need!

.. note::
  Each mini-library is meant to be an *inspiration*, but not a prescription.

  To serve that purpose, it is best to keep their code samples minimal; so we
  generally **will not merge PRs** adding new features. Instead, please send your
  lovely pull requests and design ideas to more fully-featured libraries like
  `Haiku`_ or `Flax`_.


.. toctree::
    :maxdepth: 1

    jax.example_libraries.optimizers
    jax.example_libraries.stax

.. automodule:: jax.example_libraries


.. _Haiku: https://github.com/deepmind/dm-haiku
.. _Flax: https://github.com/google/flax
