Frequently asked questions (FAQ)
================================

.. comment RST primer for Sphinx: https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html
.. comment Some links referenced here. Use `JAX - The Sharp Bits`_ (underscore at the end) to reference

.. _JAX - The Sharp Bits: https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html

We are collecting answers to frequently asked questions here.
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

.. _faq-jit-numerics:

``jit`` changes the exact numerics of outputs
---------------------------------------------
Sometimes users are surprised by the fact that wrapping a function with :func:`jit`
can change the function's outputs. For example:

>>> from jax import jit
>>> import jax.numpy as jnp
>>> def f(x):
...   return jnp.log(jnp.sqrt(x))
>>> x = jnp.pi
>>> print(f(x))
0.572365

>>> print(jit(f)(x))
0.5723649

This slight difference in output comes from optimizations within the XLA compiler:
during compilation, XLA will sometimes rearrange or elide certain operations to make
the overall computation more efficient.

In this case, XLA utilizes the properties of the logarithm to replace ``log(sqrt(x))``
with ``0.5 * log(x)``, which is a mathematically identical expression that can be
computed more efficiently than the original. The difference in output comes from
the fact that floating point arithmetic is only a close approximation of real math,
so different ways of computing the same expression may have subtly different results.

Other times, XLA's optimizations may lead to even more drastic differences.
Consider the following example:

>>> def f(x):
...   return jnp.log(jnp.exp(x))
>>> x = 100.0
>>> print(f(x))
inf

>>> print(jit(f)(x))
100.0

In non-JIT-compiled op-by-op mode, the result is ``inf`` because ``jnp.exp(x)``
overflows and returns ``inf``. Under JIT, however, XLA recognizes that ``log`` is
the inverse of ``exp``, and removes the operations from the compiled function,
simply returning the input. In this case, JIT compilation produces a more accurate
floating point approximation of the real result.

Unfortunately the full list of XLA's algebraic simplifications is not well
documented, but if you're familiar with C++ and curious about what types of
optimizations the XLA compiler makes, you can see them in the source code:
`algebraic_simplifier.cc`_.

.. _faq-slow-compile:

``jit`` decorated function is very slow to compile
--------------------------------------------------

If your ``jit`` decorated function takes tens of seconds (or more!) to run the
first time you call it, but executes quickly when called again, JAX is taking a
long time to trace or compile your code.

This is usually a sign that calling your function generates a large amount of
code in JAX's internal representation, typically because it makes heavy use of
Python control flow such as ``for`` loops. For a handful of loop iterations,
Python is OK, but if you need *many* loop iterations, you should rewrite your
code to make use of JAX's
`structured control flow primitives <https://docs.jax.dev/en/latest/control-flow.html#Structured-control-flow-primitives>`_
(such as :func:`lax.scan`) or avoid wrapping the loop with ``jit`` (you can
still use ``jit`` decorated functions *inside* the loop).

If you're not sure if this is the problem, you can try running
:func:`jax.make_jaxpr` on your function. You can expect slow compilation if the
output is many hundreds or thousands of lines long.

Sometimes it isn't obvious how to rewrite your code to avoid Python loops
because your code makes use of many arrays with different shapes. The
recommended solution in this case is to make use of functions like
:func:`jax.numpy.where` to do your computation on padded arrays with fixed
shape.

If your functions are slow to compile for another reason, please open an issue
on GitHub.

.. _faq-jit-class-methods:

How to use ``jit`` with methods?
--------------------------------

Moved to :ref:`jax-jit-class-methods`.

.. _faq-jax-vs-numpy:

Is JAX faster than NumPy?
-------------------------

One question users frequently attempt to answer with such benchmarks is whether JAX
is faster than NumPy; due to the difference in the two packages, there is not a
simple answer.

Broadly speaking:

- NumPy operations are executed eagerly, synchronously, and only on CPU.
- JAX operations may be executed eagerly or after compilation (if inside :func:`jit`);
  they are dispatched asynchronously (see :ref:`async-dispatch`); and they can
  be executed on CPU, GPU, or TPU, each of which have vastly different and continuously
  evolving performance characteristics.

These architectural differences make meaningful direct benchmark comparisons between
NumPy and JAX difficult.

Additionally, these differences have led to different engineering focus between the
packages: for example, NumPy has put significant effort into decreasing the per-call
dispatch overhead for individual array operations, because in NumPy's computational
model that overhead cannot be avoided.
JAX, on the other hand, has several ways to avoid dispatch overhead (e.g. JIT
compilation, asynchronous dispatch, batching transforms, etc.), and so reducing
per-call overhead has been less of a priority.

Keeping all that in mind, in summary: if you're doing microbenchmarks of individual
array operations on CPU, you can generally expect NumPy to outperform JAX due to its
lower per-operation dispatch overhead. If you're running your code on GPU or TPU,
or are benchmarking more complicated JIT-compiled sequences of operations on CPU, you
can generally expect JAX to outperform NumPy.

Gradients contain `NaN` where using ``where``
------------------------------------------------

If you define a function using ``where`` to avoid an undefined value, if you
are not careful you may obtain a ``NaN`` for reverse differentiation::

  def my_log(x):
    return jnp.where(x > 0., jnp.log(x), 0.)

  my_log(0.) ==> 0.  # Ok
  jax.grad(my_log)(0.)  ==> NaN

A short explanation is that during ``grad`` computation the adjoint corresponding
to the undefined ``jnp.log(x)`` is a ``NaN`` and it gets accumulated to the
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
    return jnp.where(x > 0., jnp.log(jnp.where(x > 0., x, 1.)), y)


Additional reading:

  * `Issue: gradients through jnp.where when one of branches is nan <https://github.com/jax-ml/jax/issues/1052#issuecomment-514083352>`_.
  * `How to avoid NaN gradients when using where <https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf>`_.


Why are gradients zero for functions based on sort order?
---------------------------------------------------------

If you define a function that processes the input using operations that depend on
the relative ordering of inputs (e.g. ``max``, ``greater``, ``argsort``, etc.) then
you may be surprised to find that the gradient is everywhere zero.
Here is an example, where we define `f(x)` to be a step function that returns
`0` when `x` is negative, and `1` when `x` is positive::

  import jax
  import numpy as np
  import jax.numpy as jnp

  def f(x):
    return (x > 0).astype(float)

  df = jax.vmap(jax.grad(f))

  x = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])

  print(f"f(x)  = {f(x)}")
  # f(x)  = [0. 0. 0. 1. 1.]

  print(f"df(x) = {df(x)}")
  # df(x) = [0. 0. 0. 0. 0.]

The fact that the gradient is everywhere zero may be confusing at first glance:
after all, the output does change in response to the input, so how can the gradient
be zero? However, zero turns out to be the correct result in this case.

Why is this? Remember that what differentiation is measuring the change in ``f``
given an infinitesimal change in ``x``. For ``x=1.0``, ``f`` returns ``1.0``.
If we perturb ``x`` to make it slightly larger or smaller, this does not change
the output, so by definition, :code:`grad(f)(1.0)` should be zero.
This same logic holds for all values of ``f`` greater than zero: infinitesimally
perturbing the input does not change the output, so the gradient is zero.
Similarly, for all values of ``x`` less than zero, the output is zero.
Perturbing ``x`` does not change this output, so the gradient is zero.
That leaves us with the tricky case of ``x=0``. Surely, if you perturb ``x`` upward,
it will change the output, but this is problematic: an infinitesimal change in ``x``
produces a finite change in the function value, which implies the gradient is
undefined.
Fortunately, there's another way for us to measure the gradient in this case: we
perturb the function downward, in which case the output does not change, and so the
gradient is zero.
JAX and other autodiff systems tend to handle discontinuities in this way: if the
positive gradient and negative gradient disagree, but one is defined and the other is
not, we use the one that is defined.
Under this definition of the gradient, mathematically and numerically the gradient of
this function is everywhere zero.

The problem stems from the fact that our function has a discontinuity at ``x = 0``.
Our ``f`` here is essentially a `Heaviside Step Function`_, and we can use a
`Sigmoid Function`_ as a smoothed replacement.
The sigmoid is approximately equal to the heaviside function when `x` is far from zero,
but replaces the discontinuity at ``x = 0`` with a smooth, differentiable curve.
As a result of using :func:`jax.nn.sigmoid`, we get a similar computation with
well-defined gradients::

  def g(x):
    return jax.nn.sigmoid(x)

  dg = jax.vmap(jax.grad(g))

  x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])

  with np.printoptions(suppress=True, precision=2):
    print(f"g(x)  = {g(x)}")
    # g(x)  = [0.   0.27 0.5  0.73 1.  ]

    print(f"dg(x) = {dg(x)}")
    # dg(x) = [0.   0.2  0.25 0.2  0.  ]

The :mod:`jax.nn` submodule also has smooth versions of other common rank-based
functions, for example :func:`jax.nn.softmax` can replace uses of
:func:`jax.numpy.argmax`, :func:`jax.nn.soft_sign` can replace uses of
:func:`jax.numpy.sign`, :func:`jax.nn.softplus` or :func:`jax.nn.squareplus`
can replace uses of :func:`jax.nn.relu`, etc.

How can I convert a JAX Tracer to a NumPy array?
------------------------------------------------
When inspecting a transformed JAX function at runtime, you'll find that array
values are replaced by `jax.core.Tracer` objects::

  @jax.jit
  def f(x):
    print(type(x))
    return x

  f(jnp.arange(5))

This prints the following::

  <class 'jax.interpreters.partial_eval.DynamicJaxprTracer'>

A frequent question is how such a tracer can be converted back to a normal NumPy
array. In short, **it is impossible to convert a Tracer to a NumPy array**, because
a tracer is an abstract representation of *every possible* value with a given shape
and dtype, while a numpy array is a concrete member of that abstract class.
For more discussion of how tracers work within the context of JAX transformations,
see `JIT mechanics`_.

The question of converting Tracers back to arrays usually comes up within
the context of another goal, related to accessing intermediate values in a
computation at runtime. For example:

- If you wish to print a traced value at runtime for debugging purposes, you might
  consider using :func:`jax.debug.print`.
- If you wish to call non-JAX code within a transformed JAX function, you might
  consider using :func:`jax.pure_callback`, an example of which is available at
  `Pure callback example`_.
- If you wish to input or output array buffers at runtime (for example, load data
  from file, or log the contents of the array to disk), you might consider using
  :func:`jax.experimental.io_callback`, an example of which can be found at
  `IO callback example`_.

For more information on runtime callbacks and examples of their use,
see `External callbacks in JAX`_.

Why do some CUDA libraries fail to load/initialize?
---------------------------------------------------

When resolving dynamic libraries, JAX uses the usual `dynamic linker search pattern`_.
JAX sets :code:`RPATH` to point to the JAX-relative location of the
pip-installed NVIDIA CUDA packages, preferring them if installed. If :code:`ld.so`
cannot find your CUDA runtime libraries along its usual search path, then you
must include the paths to those libraries explicitly in :code:`LD_LIBRARY_PATH`.
The easiest way to ensure your CUDA files are discoverable is to simply install
the :code:`nvidia-*-cu12` pip packages, which are included in the standard
:code:`jax[cuda_12]` install option.

Occasionally, even when you have ensured that your runtime libraries are discoverable,
there may still be some issues with loading or initializing them. A common cause of
such issues is simply having insufficient memory for CUDA library initialization at
runtime. This sometimes occurs because JAX will pre-allocate too large of a chunk of
currently available device memory for faster execution, occasionally resulting in
insufficient memory being left available for runtime CUDA library initialization. 

This is especially likely when running multiple JAX instances, running JAX in
tandem with TensorFlow which performs its own pre-allocation, or when running
JAX on a system where the GPU is being heavily utilized by other processes. When
in doubt, try running the program again with reduced pre-allocation, either by
reducing :code:`XLA_PYTHON_CLIENT_MEM_FRACTION` from the default of :code:`.75`,
or setting :code:`XLA_PYTHON_CLIENT_PREALLOCATE=false`. For more details, please
see the page on `JAX GPU memory allocation`_.

.. _faq-data-placement:

Controlling data and computation placement on devices
-----------------------------------------------------

Moved to :ref:`sharded-data-placement`.

.. _faq-benchmark:

Benchmarking JAX code
---------------------

Moved to :ref:`benchmarking-jax-code`.

.. _faq-donation:

Buffer donation
---------------

Moved to :ref:`buffer-donation`.


.. _JIT mechanics: https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables
.. _External callbacks in JAX: https://docs.jax.dev/en/latest/notebooks/external_callbacks.html
.. _Pure callback example: https://docs.jax.dev/en/latest/notebooks/external_callbacks.html#example-pure-callback-with-custom-jvp
.. _IO callback example: https://docs.jax.dev/en/latest/notebooks/external_callbacks.html#exploring-jax-experimental-io-callback
.. _Heaviside Step Function: https://en.wikipedia.org/wiki/Heaviside_step_function
.. _Sigmoid Function: https://en.wikipedia.org/wiki/Sigmoid_function
.. _algebraic_simplifier.cc: https://github.com/openxla/xla/blob/33f815e190982dac4f20d1f35adb98497a382377/xla/hlo/transforms/simplifiers/algebraic_simplifier.cc#L4851
.. _JAX GPU memory allocation: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
.. _dynamic linker search pattern: https://man7.org/linux/man-pages/man8/ld.so.8.html
