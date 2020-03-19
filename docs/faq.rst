JAX Frequently Asked Questions
==============================

We are collecting here answers to frequently asked questions.
Contributions welcome!

Gradients contain `NaN` where using ``where``
------------------------------------------------

If you define a function using ``where`` to avoid an undefined value, if you
are not careful you may obtain a `NaN` for reverse differentiation::

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

The inner ``np.where`` may be needed in addition to the original one, e.g.:

  def my_log_or_y(x, y):
    """Return log(x) if x > 0 or y"""
    return np.where(x > 0., np.log(np.where(x > 0., x, 1.), y)


Additional reading:

  * [Issue: gradients through np.where when one of branches is nan](https://github.com/google/jax/issues/1052#issuecomment-514083352)
  * [How to avoid NaN gradients when using ``where``](https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf)