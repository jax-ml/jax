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
    jax.ops
    jax.random
    jax.tree_util

Just-in-time compilation (:code:`jit`)
--------------------------------------

.. automodule:: jax
    :members: jit, disable_jit, xla_computation, make_jaxpr, eval_shape
    :undoc-members:
    :show-inheritance:

Automatic differentiation
-------------------------

.. automodule:: jax
    :members: grad, value_and_grad, jacfwd, jacrev, hessian, jvp, linearize, vjp, custom_transforms, defjvp, defjvp_all, defvjp, defvjp_all, custom_gradient
    :undoc-members:
    :show-inheritance:


Vectorization (:code:`vmap`)
----------------------------

.. automodule:: jax
    :members: vmap
    :undoc-members:
    :show-inheritance:


Parallelization (:code:`pmap`)
----------------------------

.. automodule:: jax
    :members: pmap
    :undoc-members:
    :show-inheritance:
