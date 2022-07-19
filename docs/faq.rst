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

.. _faq-jit-numerics:

``jit`` changes the exact numerics of outputs
---------------------------------------------
Sometimes users are surprised by the fact that wrapping a function with `jit` can
make its outputs slightly different. For example:

>>> from jax import jit
>>> def f(x, y):
...   return x + y - x
>>> x = jnp.array(1.0)
>>> y = jnp.array(0.001)
>>> print(f(x, y))
0.0010000467

>>> print(jit(f)(x, y))
0.001

This happens because of optimizations within the XLA compiler. During compilation,
XLA will often re-arrange floating point operations to simplify the expression it
computes. For example, consider the expression ``x + y - x`` above. In non-JIT
op-by-op evaluation, this addition and subtraction both accumulate standard
32-bit floating point arithmetic error, so the result is not exactly equal ``y``.
By contrast, in JIT the XLA compiler recognizes that the ``x`` and ``-x`` cancel
each other, and so it drops these terms and the return value is identically equal
to ``y``.

In general, for this and other related reasons, it is to be expected that JIT-compiled
code will produce slightly different outputs than its non-JIT compiled counterpart.

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

.. _faq-jit-class-methods:

How to use ``jit`` with methods?
--------------------------------
Most examples of :func:`jax.jit` concern decorating stand-alone Python functions,
but decorating a method within a class introduces some complication. For example,
consider the following simple class, where we've used a standard :func:`~jax.jit`
annotation on a method::

    >>> import jax.numpy as jnp
    >>> from jax import jit
     
    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   @jit  # <---- How to do this correctly?
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y

However, this approach will result in an error when you attempt to call this method::

    >>> c = CustomClass(2, True)
    >>> c.calc(3)  # doctest: +SKIP
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
      File "<stdin>", line 1, in <module
    TypeError: Argument '<CustomClass object at 0x7f7dd4125890>' of type <class 'CustomClass'> is not a valid JAX type.

The problem is that the first argument to the function is ``self``, which has type
``CustomClass``, and JAX does not know how to handle this type.
There are three basic strategies we might use in this case, and we'll discuss
them below.

Strategy 1: JIT-compiled helper function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The most straightforward approach is to create a helper function external to the class
that can be JIT-decorated in the normal way. For example::

    >>> from functools import partial
    
    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   def calc(self, y):
    ...     return _calc(self.mul, self.x, y)
    
    >>> @partial(jit, static_argnums=0)
    ... def _calc(mul, x, y):
    ...   if mul:
    ...     return x * y
    ...   return y

The result will work as expected::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6

The benefit of such an approach is that it is simple, explicit, and it avoids the need
to teach JAX how to handle objects of type ``CustomClass``. However, you may wish to
keep all the method logic in the same place.

Strategy 2: Marking ``self`` as static
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Another common pattern is to use ``static_argnums`` to mark the ``self`` argument as static.
But this must be done with care to avoid unexpected results.
You may be tempted to simply do this::

    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ...  
    ...   # WARNING: this example is broken, as we'll see below. Don't copy & paste!
    ...   @partial(jit, static_argnums=0)
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y

If you call the method, it will no longer raise an error::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6

However, there is a catch: if you mutate the object after the first method call, the
subsequent method call may return an incorrect result::

    >>> c.mul = False
    >>> print(c.calc(3))  # Should print 3
    6

What's happening here? The issue is that ``static_argnums`` relies on the hash of the object
to determine whether it has changed between calls, and the default ``__hash__`` method
for a user-defined class will not take into account the values of class attributes. That means
that on the second function call, JAX has no way of knowing that the class attribues have
changed, and uses the cached static value from the previous compilation.

For this reason, if you are marking ``self`` arguments as static, it is important that you
define an appropriate ``__hash__`` method for your class.
For example, you might proceed like this::

    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   @partial(jit, static_argnums=0)
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y
    ... 
    ...   def __hash__(self):
    ...     return hash((self.x, self.mul))
    ... 
    ...   def __eq__(self, other):
    ...     return (isinstance(other, CustomClass) and
    ...             (self.x, self.mul) == (other.x, other.mul))

Note that we've defined the ``__hash__`` method so that it depends on the hash of
relevant class attributes, and we've also defined the ``__eq__`` method because it's
good practice to do so any time you override ``__hash__`` (see
`Python Data Model: __hash__ <https://docs.python.org/3/reference/datamodel.html#object.__hash__>`_
for more information on this). With this addition, the example works correctly::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6
    >>> c.mul = False
    >>> print(c.calc(3))
    3

A downside of marking ``self`` as static is that it does not allow ``self`` to contain
array-like attributes, since arrays are not hashable. For example, this will break because
JAX arrays are not hashable::
  
    >>> c = CustomClass(jnp.array(2), True)
    >>> c.calc(3)  # doctest: +SKIP
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
      File "<stdin>", line 1, in <module
    ValueError: Non-hashable static arguments are not supported. An error occured during a call to 'calc' while trying to hash an object of type <class '__main__.CustomClass'>
  
Additionally, this also has the downside that ``calc`` will be re-compiled any time the values
within ``myfunc`` change, which could be costly depending on your program.

Strategy 3: Making ``CustomClass`` a PyTree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The most flexible approach to correctly JIT-compiling a class method is to register the
type as a custom PyTree object; see :ref:`extending-pytrees`. This lets you specify
exactly which components of the class should be treated as static and which should be
treated as dynamic. Here's how it might look::

    >>> class CustomClass:
    ...   def __init__(self, x: jnp.ndarray, mul: bool):
    ...     self.x = x
    ...     self.mul = mul
    ... 
    ...   @jit
    ...   def calc(self, y):
    ...     if self.mul:
    ...       return self.x * y
    ...     return y
    ... 
    ...   def _tree_flatten(self):
    ...     children = (self.x,)  # arrays / dynamic values
    ...     aux_data = {'mul': self.mul}  # static values
    ...     return (children, aux_data)
    ...
    ...   @classmethod
    ...   def _tree_unflatten(cls, aux_data, children):
    ...     return cls(*children, **aux_data)
    
    >>> from jax import tree_util
    >>> tree_util.register_pytree_node(CustomClass,
    ...                                CustomClass._tree_flatten,
    ...                                CustomClass._tree_unflatten)

This is certainly more involved, but it solves all the issues associated with the simpler
apporaches used above::

    >>> c = CustomClass(2, True)
    >>> print(c.calc(3))
    6

    >>> c.mul = False  # mutation is detected
    >>> print(c.calc(3))
    3

    >>> c = CustomClass(jnp.array(2), True)  # non-hashable x is supported
    >>> print(c.calc(3))
    6

So long as your ``tree_flatten`` and ``tree_unflatten`` functions correctly handle all
relevant attributes in the class, you should be able to use objects of this type directly
as arguments to JIT-compiled functions, without any special annotations.

.. _faq-data-placement:

Controlling data and computation placement on devices
-----------------------------------------------------

Let's first look at the principles of data and computation placement in JAX.

In JAX, the computation follows data placement. JAX arrays
have two placement properties: 1) the device where the data resides;
and 2) whether it is **committed** to the device or not (the data is sometimes
referred to as being *sticky* to the device).

By default, JAX arrays are placed uncommitted on the default device
(``jax.devices()[0]``), which is the first GPU or TPU by default. If no GPU or
TPU is present, ``jax.devices()[0]`` is the CPU. The default device can
temporarily overridden with the :func:`jax.default_device` context manager, or
set for the whole process by setting the environment variable ``JAX_PLATFORMS``
or the absl flag ``--jax_platforms`` to "cpu", "gpu", or "tpu"
(``JAX_PLATFORMS`` can also be a list of platforms, which determines which
platforms are available in priority order).

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

``jax.device_put(jnp.zeros(...), jax.devices()[1])`` or similar will actually create the
array of zeros on ``jax.devices()[1]``, instead of creating the array on the default
device then moving it. This is thanks to some laziness in array creation, which holds
for all the constant creation operations (``ones``, ``full``, ``eye``, etc).

(As of April 2020, :func:`jax.jit` has a `device` parameter that affects the device
placement. That parameter is experimental, is likely to be removed or changed,
and its use is not recommended.)

For a worked-out example, we recommend reading through
``test_computation_follows_data`` in
`multi_device_test.py <https://github.com/google/jax/blob/main/tests/multi_device_test.py>`_.

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
JAX/accelerators vs NumPy/CPU. For example, if we switch this example to use
10x10 input instead, JAX/GPU runs 10x slower than NumPy/CPU (100 µs vs 10 µs).

.. _To JIT or not to JIT: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#to-jit-or-not-to-jit
.. _Double (64 bit) precision: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
.. _`%time and %timeit magics`: https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-time
.. _Colab: https://colab.research.google.com/

.. _faq-jax-vs-numpy:

Is JAX faster than NumPy?
~~~~~~~~~~~~~~~~~~~~~~~~~
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

.. _faq-donation:

Buffer donation
---------------

(This feature is implemented only for TPU and GPU.)

When JAX executes a computation it uses buffers on the device for all inputs and outputs.
If you know that one of the inputs is not needed after the computation, and if it
matches the shape and element type of one of the outputs, you can specify that you
want the corresponding input buffer to be donated to hold an output. This will reduce
the memory required for the execution by the size of the donated buffer.

If you have something like the following pattern, you can use buffer donation::

   params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params, state)

You can think of this as a way to do a memory-efficient functional update
on your immutable JAX arrays. Within the boundaries of a computation XLA can
make this optimization for you, but at the jit/pmap boundary you need to
guarantee to XLA that you will not use the donated input buffer after calling
the donating function.

You achieve this by using the `donate_argnums` parameter to the functions :func:`jax.jit`,
:func:`jax.pjit`, and :func:`jax.pmap`. This parameter is a sequence of indices (0 based) into
the positional argument list::

   def add(x, y):
     return x + y

   x = jax.device_put(np.ones((2, 3)))
   y = jax.device_put(np.ones((2, 3)))
   # Execute `add` with donation of the buffer for `y`. The result has
   # the same shape and type as `y`, so it will share its buffer.
   z = jax.jit(add, donate_argnums=(1,))(x, y)

Note that this currently does not work when calling your function with key-word arguments!
The following code will not donate any buffers::

   params, state = jax.pmap(update_fn, donate_argnums=(0, 1))(params=params, state=state)

If an argument whose buffer is donated is a pytree, then all the buffers
for its components are donated::

   def add_ones(xs: List[Array]):
     return [x + 1 for x in xs]

   xs = [jax.device_put(np.ones((2, 3)), jax.device_put(np.ones((3, 4))]
   # Execute `add_ones` with donation of all the buffers for `xs`.
   # The outputs have the same shape and type as the elements of `xs`,
   # so they will share those buffers.
   z = jax.jit(add_ones, donate_argnums=0)(xs)

It is not allowed to donate a buffer that is used subsequently in the computation,
and JAX will give an error because the buffer for `y` has become invalid
after it was donated::

   # Donate the buffer for `y`
   z = jax.jit(add, donate_argnums=(1,))(x, y)
   w = y + 1  # Reuses `y` whose buffer was donated above
   # >> RuntimeError: Invalid argument: CopyToHostAsync() called on invalid buffer

You will get a warning if the donated buffer is not used, e.g., because
there are more donated buffers than can be used for the outputs::

   # Execute `add` with donation of the buffers for both `x` and `y`.
   # One of those buffers will be used for the result, but the other will
   # not be used.
   z = jax.jit(add, donate_argnums=(0, 1))(x, y)
   # >> UserWarning: Some donated buffers were not usable: f32[2,3]{1,0}

The donation may also be unused if there is no output whose shape matches
the donation::

   y = jax.device_put(np.ones((1, 3)))  # `y` has different shape than the output
   # Execute `add` with donation of the buffer for `y`.
   z = jax.jit(add, donate_argnums=(1,))(x, y)
   # >> UserWarning: Some donated buffers were not usable: f32[1,3]{1,0}

Buffer donation is implemented for GPU and TPU. You will get the above warning
anytime you try to use donation on CPU.

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
    return jnp.where(x > 0., jnp.log(jnp.where(x > 0., x, 1.), y)


Additional reading:

  * `Issue: gradients through jnp.where when one of branches is nan <https://github.com/google/jax/issues/1052#issuecomment-514083352>`_.
  * `How to avoid NaN gradients when using where <https://github.com/tensorflow/probability/blob/master/discussion/where-nan.pdf>`_.


Why are gradients zero for functions based on sort order?
---------------------------------------------------------

If you define a function that processes the input using operations that depend on
the relative ordering of inputs (e.g. ``max``, ``greater``, ``argsort``, etc.) then
you may be surprised to find that the gradient is everywhere zero.
Here is an example, where we define `f(x)` to be a step function that returns
`0` when `x` is negative, and `1` when `x` is positive:

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
This same logic holds for all values of ``f`` greater than zero: infinitessimally
perturbing the input does not change the output, so the gradient is zero.
Similarly, for all values of ``x`` less than zero, the output is zero.
Perturbing ``x`` does not change this output, so the gradient is zero.
That leaves us with the tricky case of ``x=0``. Surely, if you perturb ``x`` upward,
it will change the output, but this is problematic: an infinitesimal change in ``x``
produces a finite change inthe function value, which implies the gradient is
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
:func:`jax.numpy.sign`, :func:`jax.nn.softplus` can replace uses of
:func:`jax.nn.relu`, etc.


Additional Sections
-------------------

.. comment We refer to the anchor below in JAX error messages

``Abstract tracer value encountered where concrete value is expected`` error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
See :class:`jax.errors.ConcretizationTypeError`


.. _Heaviside Step Function: https://en.wikipedia.org/wiki/Heaviside_step_function
.. _Sigmoid Function: https://en.wikipedia.org/wiki/Sigmoid_function
