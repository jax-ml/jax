.. currentmodule:: jax

jax package
===========

Subpackages
-----------

.. toctree::
    :maxdepth: 1

    jax.numpy
    jax.scipy
    jax.experimental
    jax.lax
    jax.nn
    jax.ops
    jax.random
    jax.tree_util

Just-in-time compilation (:code:`jit`)
--------------------------------------

.. autofunction:: jit
.. autofunction:: disable_jit
.. autofunction:: xla_computation
.. autofunction:: make_jaxpr
.. autofunction:: eval_shape

Automatic differentiation
-------------------------

.. autofunction:: grad
.. autofunction:: value_and_grad
.. autofunction:: jacfwd
.. autofunction:: jacrev
.. autofunction:: hessian
.. autofunction:: jvp
.. autofunction:: linearize
.. autofunction:: vjp
.. autofunction:: custom_transforms
.. autofunction:: defjvp
.. autofunction:: defjvp_all
.. autofunction:: defvjp
.. autofunction:: defvjp_all
.. autofunction:: custom_gradient


Vectorization (:code:`vmap`)
----------------------------

.. autofunction:: vmap


Parallelization (:code:`pmap`)
------------------------------

.. autofunction:: pmap
.. autofunction:: devices
.. autofunction:: local_devices
.. autofunction:: host_id
.. autofunction:: host_ids
.. autofunction:: device_count
.. autofunction:: local_device_count
.. autofunction:: host_count

Tagging (:code:`collect` and :code:`inject`)
------------------------------

.. autofunction:: Scope
.. autofunction:: collect
.. autofunction:: inject
