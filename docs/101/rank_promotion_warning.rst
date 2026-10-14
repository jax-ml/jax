Rank promotion warning
======================

`NumPy broadcasting rules
<https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>`_
allow the automatic promotion of arguments from one rank (number of array axes)
to another. This behavior can be convenient when intended but can also lead to
surprising bugs where a silent rank promotion masks an underlying shape error.

Here's an example of rank promotion:

>>> from jax import numpy as jnp
>>> x = jnp.arange(12).reshape(4, 3)
>>> y = jnp.array([0, 1, 0])
>>> x + y
Array([[ 0,  2,  2],
       [ 3,  5,  5],
       [ 6,  8,  8],
       [ 9, 11, 11]], dtype=int32)

To avoid potential surprises, you can configure :code:`jax.numpy` so that
expressions requiring rank promotion produce a warning or an error, or are
allowed just like in regular NumPy. The configuration option is named
:code:`jax_numpy_rank_promotion` and it can take the string values
:code:`allow`, :code:`warn`, and :code:`raise`. The default setting is
:code:`allow`, which allows rank promotion without warning or error.
The :code:`raise` setting raises an error on rank promotion, and :code:`warn`
raises a warning on the first occurrence of rank promotion.

You can set the option locally with the :func:`jax.numpy_rank_promotion`
context manager:

.. code-block:: python

   with jax.numpy_rank_promotion("warn"):
     z = x + y

You can also set it globally in several ways. One is by using
:code:`jax.config` in your code:

.. code-block:: python

  import jax
  jax.config.update("jax_numpy_rank_promotion", "warn")

Another is the environment variable :code:`JAX_NUMPY_RANK_PROMOTION`, for
example :code:`JAX_NUMPY_RANK_PROMOTION='warn'`. Finally, when using
:code:`absl-py` you can set the option with a command-line flag.
