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

Just-in-time compilation (:code:`jit`)
--------------------------------------

.. autosummary::

    jit
    disable_jit
    xla_computation
    make_jaxpr
    eval_shape
    device_put
    named_call

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
    host_id
    host_ids
    device_count
    local_device_count
    host_count


.. autofunction:: jit
.. autofunction:: disable_jit
.. autofunction:: xla_computation
.. autofunction:: make_jaxpr
.. autofunction:: eval_shape
.. autofunction:: device_put
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

.. autofunction:: pmap
.. autofunction:: devices
.. autofunction:: local_devices
.. autofunction:: host_id
.. autofunction:: host_ids
.. autofunction:: device_count
.. autofunction:: local_device_count
.. autofunction:: host_count
