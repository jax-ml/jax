.. currentmodule:: jax.experimental

``jax.experimental`` module
===========================

``jax.experimental.optix`` has been moved into its own Python package
(https://github.com/deepmind/optax).

``jax.experimental.ann`` has been moved into ``jax.lax``.

Experimental Modules
--------------------

.. toctree::
    :maxdepth: 1

    jax.experimental.checkify
    jax.experimental.compilation_cache
    jax.experimental.custom_dce
    jax.experimental.custom_partitioning
    jax.experimental.jet
    jax.experimental.key_reuse
    jax.experimental.mesh_utils
    jax.experimental.multihost_utils
    jax.experimental.pallas
    jax.experimental.pjit
    jax.experimental.serialize_executable
    jax.experimental.shard_map
    jax.experimental.sparse

Experimental APIs
-----------------

.. autosummary::
   :toctree: _autosummary

   enable_x64
   disable_x64
