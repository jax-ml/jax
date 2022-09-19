.. currentmodule:: jax.experimental

jax.experimental package
========================

``jax.experimental.optix`` has been moved into its own Python package
(https://github.com/deepmind/optax).

``jax.experimental.ann`` has been moved into ``jax.lax``.

Experimental Modules
--------------------

.. toctree::
    :maxdepth: 1

    jax.experimental.checkify
    jax.experimental.global_device_array
    jax.experimental.host_callback
    jax.experimental.maps
    jax.experimental.pjit
    jax.experimental.sparse
    jax.experimental.jet

Experimental APIs
-----------------

.. autosummary::
   :toctree: _autosummary

   enable_x64
   disable_x64

   jax.experimental.checkify.checkify
   jax.experimental.checkify.check
   jax.experimental.checkify.check_error
