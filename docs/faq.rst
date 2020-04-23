JAX Frequently Asked Questions
==============================

.. comment RST primer for Sphinx: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html
.. comment Some links referenced here. Use JAX_sharp_bits_ (underscore at the end) to reference


.. _JAX_sharp_bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
.. _How_JAX_primitives_work: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html

We are collecting here answers to frequently asked questions.
Contributions welcome!

Creating arrays with `jax.numpy.array` is slower than with `numpy.array`
------------------------------------------------------------------------

The following code is relatively fast when using NumPy, and slow when using
JAX's NumPy::

  import numpy as np
  np.array([0] * int(1e6))

The reason is that in NumPy the ``numpy.array`` function is implemented in C, while
the :func:`jax.numpy.array` is implemented in Python, and it needs to iterate over a long
list to convert each list element to an array element.

An alternative would be to create the array with original NumPy and then convert
it to a JAX array::

  from jax import numpy as jnp
  jnp.array(np.array([0] * int(1e6)))

`jit` changes the behavior of my function
-----------------------------------------

If you have a Python function that changes behavior after using :func:`jax.jit`, perhaps
your function uses global state, or has side-effects. In the following code, the
``impure_func`` uses the global ``y`` and has a side-effect due to ``print``::

    y = 0

    # @jit   # Different behavior with jit
    def impure_func(x):
      print("Inside:", y)
      return x + y

    for y in range(3):
      print("Result:", impure_func(y))

Without ``jit`` the output is::

    Inside: 0
    Result: 0
    Inside: 1
    Result: 2
    Inside: 2
    Result: 4

and with ``jit`` it is::

    Inside: 0
    Result: 0
    Result: 1
    Result: 2

For :func:`jax.jit`, the function is executed once using the Python interpreter, at which time the
``Inside`` printing happens, and the first value of ``y`` is observed. Then the function
is compiled and cached, and executed multiple times with different values of ``x``, but
with the same first value of ``y``.

Additional reading:

  * JAX_sharp_bits_


Gradients contain `NaN` where using ``where``
------------------------------------------------

If you define a function using ``where`` to avoid an undefined value, if you
are not careful you may obtain a ``NaN`` for reverse differentiation::

  def my_log(x):
    return np.where(x > 0., np.log(x), 0.)

  my_log(0.) ==> 0.  # Ok
  jax.grad(my_log)(0.)  ==> NaN

A short explanation is that during ``grad`` computation the adjoint corresponding
to the undefined ``np.log(x)`` is a ``NaN`` and when it gets accumulated to the
adjoint of the ``np.where``. The correct way to write such functions is to ensure
that there is a ``np.where`` *inside* the partially-defined function, to ensure
that the adjoint is always finite::

  def safe_for_grad_log(x):
    return np.log(np.where(x > 0., x, 1.)

  safe_for_grad_log(0.) ==> 0.  # Ok
  jax.grad(safe_for_grad_log)(0.)  ==> 0.  # Ok

The inner ``np.where`` may be needed in addition to the original one, e.g.::

  def my_log_or_y(x, y):
    """Return log(x) if x > 0 or y"""
    return np.where(x > 0., np.log(np.where(x > 0., x, 1.), y)


Additional reading:

  * `Issue: gradients through np.where when one of branches is nan <https://github.com/google/jax/issues/1052#issuecomment-514083352>`_.
  * `How to avoid NaN gradients when using where <https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf>`_.

Why do I get forward-mode differentiation error when I am trying to do reverse-mode differentiation?
-----------------------------------------------------------------------------------------------------

JAX implements reverse-mode differentiation as a composition of two operations:
linearization and transposition. The linearization step (see :func:`jax.linearize`)
uses the JVP rules to form the forward-computation of tangents along with the intermediate
forward computations of intermediate values on which the tangents depend.
The transposition step will turn the forward-computation of tangents
into a reverse-mode computation.

If the JVP rule is not implemented for a primitive, then neither the forward-mode
nor the reverse-mode differentiation will work, but the error given will refer
to the forward-mode because that is the one that fails.

You can read more details at How_JAX_primitives_work_.

