JAX Frequently Asked Questions (FAQ)
====================================

.. comment RST primer for Sphinx: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html
.. comment Some links referenced here. Use `JAX - The Sharp Bits`_ (underscore at the end) to reference

.. _JAX - The Sharp Bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

We are collecting here answers to frequently asked questions.
Contributions welcome!

``jit`` changes the behavior of my function
--------------------------------------------

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
``Inside`` printing happens, and the first value of ``y`` is observed. Then, the function
is compiled and cached, and executed multiple times with different values of ``x``, but
with the same first value of ``y``.

Additional reading:

  * `JAX - The Sharp Bits`_

.. _faq-slow-compile:

``jit`` decorated function is very slow to compile
--------------------------------------------------

If your ``jit`` decorated function takes tens of seconds (or more!) to run the
first time you call it, but executes quickly when called again, JAX is taking a
long time to trace or compile your code.

This is usually a symptom of calling your function generating a large amount of
code in JAX's internal representation, typically because it makes heavy use of
Python control flow such as ``for`` loop. For a handful of loop iterations
Python is OK, but if you need _many_ loop iterations, you should rewrite your
code to make use of JAX's
`structured control flow primitives <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#Structured-control-flow-primitives>`_
(such as :func:`lax.scan`) or avoid wrapping the loop with ``jit`` (you can
still use ``jit`` decorated functions *inside* the loop).

If you're not sure if this is the problem, you can try running
:func:`jax.make_jaxpr` on your function. You can expect slow compilation if the
output is many hundreds or thousands of lines long.

Sometimes it isn't obvious how to rewrite your code to avoid Python loops
because your code makes use of many arrays with different shapes. The
recommended solution in this case is to make use of functions like
:func:`jax.numpy.where` to do your computation on padded arrays with fixed
shape. The JAX team is exploring a "masking" transformation to make such code
easier to write.

If your functions are slow to compile for another reason, please open an issue
on GitHub.

.. _faq-data-placement:

Controlling data and computation placement on devices
-----------------------------------------------------

Let's first look at the principles of data and computation placement in JAX.

In JAX, the computation follows data placement. JAX arrays
have two placement properties: 1) the device where the data resides;
and 2) whether it is **committed** to the device or not (the data is sometimes 
referred to as being *sticky* to the device).

By default, JAX arrays are placed uncommitted on the default device
(``jax.devices()[0]``), which is the first GPU by default. If no GPU is 
present, ``jax.devices()[0]`` is the first CPU. The default device can 
be set to "cpu" or "gpu" manually by setting the environment variable 
``JAX_PLATFORM_NAME`` or the absl flag ``--jax_platform_name``.

>>> from jax import numpy as jnp
>>> print(jnp.ones(3).device_buffer.device())  # doctest: +SKIP
gpu:0

Computations involving uncommitted data are performed on the default
device and the results are uncommitted on the default device.

Data can also be placed explicitly on a device using :func:`jax.device_put`
with a ``device`` parameter, in which case the data becomes **committed** to the device:

>>> import jax
>>> from jax import device_put
>>> print(device_put(1, jax.devices()[2]).device_buffer.device())  # doctest: +SKIP
gpu:2

Computations involving some committed inputs will happen on the
committed device and the result will be committed on the
same device. Invoking an operation on arguments that are committed 
to more than one device will raise an error.

You can also use :func:`jax.device_put` without a ``device`` parameter. If the data 
is already on a device (committed or not), it's left as-is. If the data isn't on any 
device—that is, it's a regular Python or NumPy value—it's placed uncommitted on the default 
device.

Jitted functions behave like any other primitive operations—they will follow the 
data and will show errors if invoked on data committed on more than one device.

``jnp.device_put(jnp.zeros(...), jax.devices()[1])`` or similar will actually create the
array of zeros on ``jax.devices()[1]``, instead of creating the array on the default
device then moving it. This is thanks to some laziness in array creation, which holds
for all the constant creation operations (``ones``, ``full``, ``eye``, etc).

(As of April 2020, :func:`jax.jit` has a `device` parameter that affects the device 
placement. That parameter is experimental, is likely to be removed or changed, 
and its use is not recommended.)

For a worked-out example, we recommend reading through
``test_computation_follows_data`` in
`multi_device_test.py <https://github.com/google/jax/blob/master/tests/multi_device_test.py>`_.

.. _faq-benchmark:

Benchmarking JAX code
---------------------

You just ported a tricky function from NumPy/SciPy to JAX. Did that actuallly
speed things up?

Keep in mind these important differences from NumPy when measuring the
speed of code using JAX:

1. **JAX code is Just-In-Time (JIT) compiled.** Most code written in JAX can be
   written in such a way that it supports JIT compilation, which can make it run
   *much faster* (see `To JIT or not to JIT`_). To get maximium performance from
   JAX, you should apply :func:`jax.jit` on your outer-most function calls.

   Keep in mind that the first time you run JAX code, it will be slower because
   it is being compiled. This is true even if you don't use ``jit`` in your own
   code, because JAX's builtin functions are also JIT compiled.
2. **JAX has asynchronous dispatch.** This means that you need to call
   ``.block_until_ready()`` to ensure that computation has actually happened
   (see :ref:`async-dispatch`).
3. **JAX by default only uses 32-bit dtypes.** You may want to either explicitly
   use 32-bit dtypes in NumPy or enable 64-bit dtypes in JAX (see
   `Double (64 bit) precision`_) for a fair comparison.
4. **Transferring data between CPUs and accelerators takes time.** If you only
   want to measure the how long it takes to evaluate a function, you may want to
   transfer data to the device on which you want to run it first (see
   :ref:`faq-data-placement`).

Here's an example of how to put together all these tricks into a microbenchmark
for comparing JAX versus NumPy, making using of IPython's convenient
`%time and %timeit magics`_::

    import numpy as np
    import jax.numpy as jnp
    import jax

    def f(x):  # function we're benchmarking (works in both NumPy & JAX)
      return x.T @ (x - x.mean(axis=0))

    x_np = np.ones((1000, 1000), dtype=np.float32)  # same as JAX default dtype
    %timeit f(x_np)  # measure NumPy runtime

    %time x_jax = jax.device_put(x_np)  # measure JAX device transfer time
    f_jit = jax.jit(f)
    %time f_jit(x_jax).block_until_ready()  # measure JAX compilation time
    %timeit f_jit(x_jax).block_until_ready()  # measure JAX runtime

When run with a GPU in Colab_, we see:

- NumPy takes 16.2 ms per evaluation on the CPU
- JAX takes 1.26 ms to copy the NumPy arrays onto the GPU
- JAX takes 193 ms to compile the function
- JAX takes 485 µs per evaluation on the GPU

In this case, we see that once the data is transfered and the function is
compiled, JAX on the GPU is about 30x faster for repeated evaluations.

Is this a fair comparison? Maybe. The performance that ultimately matters is for
running full applications, which inevitably include some amount of both data
transfer and compilation. Also, we were careful to pick large enough arrays
(1000x1000) and an intensive enough computation (the ``@`` operator is
performing matrix-matrix multiplication) to amortize the increased overhead of
JAX/accelerators vs NumPy/CPU. For example, if switch this example to use
10x10 input instead, JAX/GPU runs 10x slower than NumPy/CPU (100 µs vs 10 µs).

.. _To JIT or not to JIT: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#to-jit-or-not-to-jit
.. _Double (64 bit) precision: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
.. _`%time and %timeit magics`: https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-time
.. _Colab: https://colab.research.google.com/

.. comment We refer to the anchor below in JAX error messages

``Abstract tracer value encountered where concrete value is expected`` error
----------------------------------------------------------------------------
See :class:`jax.errors.ConcretizationTypeError`

.. _faq-different-kinds-of-jax-values:

Different kinds of JAX values
-----------------------------

In the process of transforming functions, JAX replaces some function
arguments with special tracer values.

You could see this if you use a ``print`` statement::

  def func(x):
    print(x)
    return jnp.cos(x)

  res = jax.jit(func)(0.)

The above code does return the correct value ``1.`` but it also prints
``Traced<ShapedArray(float32[])>`` for the value of ``x``. Normally, JAX
handles these tracer values internally in a transparent way, e.g.,
in the numeric JAX primitives that are used to implement the
``jax.numpy`` functions. This is why ``jnp.cos`` works in the example above.

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
with ``ShapedArray`` abstract value. Another example is when explicitly
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
    return jnp.where(x > 0., jnp.log(x), 0.)

  my_log(0.) ==> 0.  # Ok
  jax.grad(my_log)(0.)  ==> NaN

A short explanation is that during ``grad`` computation the adjoint corresponding
to the undefined ``jnp.log(x)`` is a ``NaN`` and when it gets accumulated to the
adjoint of the ``jnp.where``. The correct way to write such functions is to ensure
that there is a ``jnp.where`` *inside* the partially-defined function, to ensure
that the adjoint is always finite::

  def safe_for_grad_log(x):
    return jnp.log(jnp.where(x > 0., x, 1.))

  safe_for_grad_log(0.) ==> 0.  # Ok
  jax.grad(safe_for_grad_log)(0.)  ==> 0.  # Ok

The inner ``jnp.where`` may be needed in addition to the original one, e.g.::

  def my_log_or_y(x, y):
    """Return log(x) if x > 0 or y"""
    return jnp.where(x > 0., jnp.log(jnp.where(x > 0., x, 1.), y)


Additional reading:

  * `Issue: gradients through jnp.where when one of branches is nan <https://github.com/google/jax/issues/1052#issuecomment-514083352>`_.
  * `How to avoid NaN gradients when using where <https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf>`_.
