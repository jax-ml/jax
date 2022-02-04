.. currentmodule:: jax

Public API: jax package
=======================

Subpackages
-----------

.. toctree::
    :maxdepth: 1

    jax.numpy
    jax.scipy
    jax.example_libraries
    jax.experimental
    jax.image
    jax.lax
    jax.nn
    jax.ops
    jax.random
    jax.tree_util
    jax.flatten_util
    jax.dlpack
    jax.profiler
    jax.config

.. toctree::
   :hidden:

   jax.lib

.. _jax-jit:

Just-in-time compilation (:code:`jit`)
--------------------------------------

.. autosummary::
  :toctree: _autosummary

    jit
    disable_jit
    ensure_compile_time_eval
    xla_computation
    make_jaxpr
    eval_shape
    device_put
    device_put_replicated
    device_put_sharded
    device_get
    default_backend
    named_call
    block_until_ready

.. _jax-grad:

Automatic differentiation
-------------------------

.. autosummary::
  :toctree: _autosummary

    grad
    value_and_grad
    jacfwd
    jacrev
    hessian
    jvp
    linearize
    linear_transpose
    vjp
    custom_jvp
    custom_vjp
    closure_convert
    checkpoint


Vectorization (:code:`vmap`)
----------------------------

.. autosummary::
  :toctree: _autosummary

    vmap
    numpy.vectorize

Parallelization (:code:`pmap`)
------------------------------

.. autosummary::
  :toctree: _autosummary

    pmap
    devices
    local_devices
    process_index
    device_count
    local_device_count
    process_count
