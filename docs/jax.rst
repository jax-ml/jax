.. currentmodule:: jax

Public API: jax package
=======================

Subpackages
-----------

.. toctree::
    :maxdepth: 1

    jax.numpy
    jax.scipy
    jax.experimental
    jax.image
    jax.lax
    jax.nn
    jax.ops
    jax.random
    jax.tree_util
    jax.dlpack
    jax.profiler

.. _jax-jit:

Just-in-time compilation (:code:`jit`)
--------------------------------------

.. autosummary::

    jit
    disable_jit
    xla_computation
    make_jaxpr
    eval_shape
    device_put
    device_put_replicated
    device_put_sharded
    default_backend
    named_call

.. _jax-grad:

Automatic differentiation
-------------------------

.. autosummary::

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

    vmap
    jax.numpy.vectorize

Parallelization (:code:`pmap`)
------------------------------

.. autosummary::

    pmap
    devices
    local_devices
    process_index
    device_count
    local_device_count
    process_count


.. autofunction:: jit
.. autofunction:: disable_jit
.. autofunction:: xla_computation
.. autofunction:: make_jaxpr
.. autofunction:: eval_shape
.. autofunction:: device_put
.. autofunction:: device_put_replicated
.. autofunction:: device_put_sharded
.. autofunction:: default_backend
.. autofunction:: named_call

.. autofunction:: grad
.. autofunction:: value_and_grad
.. autofunction:: jacfwd
.. autofunction:: jacrev
.. autofunction:: hessian
.. autofunction:: jvp
.. autofunction:: linearize
.. autofunction:: linear_transpose
.. autofunction:: vjp
.. autoclass:: custom_jvp

    .. automethod:: defjvp
    .. automethod:: defjvps

.. autoclass:: custom_vjp

    .. automethod:: defvjp

.. autofunction:: closure_convert

.. autofunction:: checkpoint

.. autofunction:: vmap
.. autofunction:: jax.numpy.vectorize
  :noindex:

.. autofunction:: pmap
.. autofunction:: devices
.. autofunction:: local_devices
.. autofunction:: process_index
.. autofunction:: device_count
.. autofunction:: local_device_count
.. autofunction:: process_count
