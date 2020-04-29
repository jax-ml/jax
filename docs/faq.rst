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


Controlling data and computation placement on devices
-----------------------------------------------------

We describe first the principles of data and computation placement
in JAX.

In JAX the computation follows the data placement. JAX arrays
have two placement properties: the device where the data resides,
and whether it is **committed** to the device or not (we sometimes
say that the data is *sticky* to the device).

By default, JAX arrays are placed uncommitted on the default device
(``jax.devices()[0]``).

>>> from jax import numpy as jnp
>>> print(jnp.ones(3).device_buffer.device())
gpu:0

Computations involving uncommitted data are performed on the default
device and the results are uncommitted on the default device.

Data can also be placed explicitly on a device using :func:`jax.device_put`
with a ``device`` parameter, in which case if becomes **committed** to the device:

>>> from jax import device_put
>>> print(device_put(1, jax.devices()[2]).device_buffer.device())
gpu:2

Computations involving some committed inputs, will happen on the
committed device, and the result will be committed on the
same device. It is an error to invoke an operation on
arguments that are committed to more than one device.

You can also use :func:`jax.device_put` without a ``device`` parameter,
in which case the data is left as is if already on a device (whether
committed or not), or a Python value that is not on any device is
placed uncommitted on the default device.

Jitted functions behave as any other primitive operation
(will follow the data and will error if invoked on data
committed on more than one device).

(As of April 2020, :func:`jax.jit` has a `device` parameter
that affects slightly the device placement. That parameter
is experimental, is likely to be removed or changed, and
its use is not recommended.)

For a worked-out example, we recommend reading through
``test_computation_follows_data`` in
[multi_device_test.py](https://github.com/google/jax/blob/master/tests/multi_device_test.py).

.. comment We refer to the anchor below in JAX error messages

`Abstract tracer value encountered where concrete value is expected` error
--------------------------------------------------------------------------

If you are getting an error that a library function is called with
*"Abstract tracer value encountered where concrete value is expected"*, you may need to
change how you invoke JAX transformations. We give first an example, and
a couple of solutions, and then we explain in more detail what is actually
happening, if you are curious or the simple solution does not work for you.

Some library functions take arguments that specify shapes or axes,
such as the 2nd and 3rd arguments for :func:`jax.numpy.split`::

  # def np.split(arr, num_sections: Union[int, Sequence[int]], axis: int):
  np.split(np.zeros(2), 2, 0)  # works

If you try the following code::

  jax.jit(np.split)(np.zeros(4), 2, 0)

you will get the following error::

    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected (in jax.numpy.split argument 1).
    Use transformation parameters such as `static_argnums` for `jit` to avoid tracing input values.
    See `https://jax.readthedocs.io/en/latest/faq.html#abstract-tracer-value-where-concrete-value-is-expected-error`.
    Encountered value: Traced<ShapedArray(int32[], weak_type=True):JaxprTrace(level=-1/1)>

We must change the way we use :func:`jax.jit` to ensure that the ``num_sections``
and ``axis`` arguments use their concrete values (``2`` and ``0`` respectively).
The best mechanism is to use special transformation parameters
to declare some arguments to be static, e.g., ``static_argnums`` for :func:`jax.jit`::

  jax.jit(np.split, static_argnums=(1, 2))(np.zeros(4), 2, 0)

An alternative is to apply the transformation to a closure
that encapsulates the arguments to be protected, either manually as below
or by using ``functools.partial``::

  jax.jit(lambda arr: np.split(arr, 2, 0))(np.zeros(4))

**Note a new closure is created at every invocation, which defeats the
compilation caching mechanism, which is why static_argnums is preferred.**

To understand more subtleties having to do with tracers vs. regular values, and
concrete vs. abstract values, you may want to read `Different kinds of JAX values`_.

Different kinds of JAX values
------------------------------

In the process of transforming functions, JAX replaces some some function
arguments with special tracer values.
You could see this if you use a ``print`` statement::

  def func(x):
    print(x)
    return np.cos(x)

  res = jax.jit(func)(0.)

The above code does return the correct value ``1.`` but it also prints
``Traced<ShapedArray(float32[])>`` for the value of ``x``. Normally, JAX
handles these tracer values internally in a transparent way, e.g.,
in the numeric JAX primitives that are used to implement the
``jax.numpy`` functions. This is why ``np.cos`` works in the example above.

More precisely, a **tracer** value is introduced for the argument of
a JAX-transformed function, except the arguments identified by special
parameters such as ``static_argnums`` for :func:`jax.jit` or
``static_broadcasted_argnums`` for :func:`jax.pmap`. Typically, computations
that involve at least a tracer value will produce a tracer value. Besides tracer
values, there are **regular** Python values: values that are computed outside JAX
transformations, or arise from above-mentioned static arguments of certain JAX
transformations, or computed solely from other regular Python values.
These are the values that are used everywhere in absence of JAX transformations.

A tracer value carries an **abstract** value, e.g., ``ShapedArray`` with information
about the shape and dtype of an array. We will refer here to such tracers as
**abstract tracers**. Some tracers, e.g., those that are
introduced for arguments of autodiff transformations, carry ``ConcreteArray``
abstract values that actually include the regular array data, and are used,
e.g., for resolving conditionals. We will refer here to such tracers
as **concrete tracers**. Tracer values computed from these concrete tracers,
perhaps in combination with regular values, result in concrete tracers.
A **concrete value** is either a regular value or a concrete tracer.

Most often values computed from tracer values are themselves tracer values.
There are very few exceptions, when a computation can be entirely done
using the abstract value carried by a tracer, in which case the result
can be a regular value. For example, getting the shape of a tracer
with ``ShapedArray`` abstract value. Another example, is when explicitly
casting a concrete tracer value to a regular type, e.g., ``int(x)`` or
``x.astype(float)``.
Another such situation is for ``bool(x)``, which produces a Python bool when
concreteness makes it possible. That case is especially salient because
of how often it arises in control flow.

Here is how the transformations introduce abstract or concrete tracers:

  * :func:`jax.jit`: introduces **abstract tracers** for all positional arguments
    except those denoted by ``static_argnums``, which remain regular
    values.
  * :func:`jax.pmap`: introduces **abstract tracers** for all positional arguments
    except those denoted by ``static_broadcasted_argnums``.
  * :func:`jax.vmap`, :func:`jax.make_jaxpr`, :func:`xla_computation`:
    introduce **abstract tracers** for all positional arguments.
  * :func:`jax.jvp` and :func:`jax.grad` introduce **concrete tracers**
    for all positional arguments. An exception is when these transformations
    are within an outer transformation and the actual arguments are
    themselves abstract tracers; in that case, the tracers introduced
    by the autodiff transformations are also abstract tracers.
  * All higher-order control-flow primitives (:func:`lax.cond`, :func:`lax.while_loop`,
    :func:`lax.fori_loop`, :func:`lax.scan`) when they process the functionals
    introduce **abstract tracers**, whether or not there is a JAX transformation
    in progress.

All of this is relevant when you have code that can operate
only on regular Python values, such as code that has conditional
control-flow based on data::

    def divide(x, y):
      return x / y if y >= 1. else 0.

If we want to apply :func:`jax.jit`, we must ensure to specify ``static_argnums=1``
to ensure ``y`` stays a regular value. This is due to the boolean expression
``y >= 1.``, which requires concrete values (regular or tracers). The
same would happen if we write explicitly ``bool(y >= 1.)``, or ``int(y)``,
or ``float(y)``.

Interestingly, ``jax.grad(divide)(3., 2.)``, works because :func:`jax.grad`
uses concrete tracers, and resolves the conditional using the concrete
value of ``y``.

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

