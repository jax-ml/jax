# JAX and TensorFlow interoperation (jax2tf/call_tf)

This package provides experimental support for interoperation between JAX and TensorFlow.
There are two interoperation directions:

- `jax2tf.convert`: for using JAX functions in a TensorFlow context, e.g.,
for eager or graph execution, or for saving as a TensorFlow SavedModel; and
- `jax2tf.call_tf`: for using TensorFlow  functions in a JAX context, e.g., to call a
TensorFlow library or a SavedModel inside a JAX function.

The `jax2tf.convert` mechanism can wrap a function
written in JAX, possibly including JAX transformations, and turn it into
a function that uses only TensorFlow operations. The converted function
can be called or traced from TensorFlow and will behave as if it was written in TensorFlow.
In practice this means that you can take some code written in JAX and execute it using
TensorFlow eager mode, or stage it out as a TensorFlow graph, even save it
as a SavedModel for archival, or for use with TensorFlow tools such as serving stack,
or TensorFlow Hub.

This package also contains the `jax2tf.call_tf` mechanism to call TensorFlow functions
from JAX. These functions can be called in JAX's op-by-op execution mode,
in which case the callee is executed in eager mode, or in JAX's jit (staged) context,
in which case the callee is compiled to XLA and embedded in JAX's staged XLA.

Both interoperation directions rely on the ability of
TensorFlow to use the XLA compiler (`tf.function(jit_compile=True)`). For the
`jax2tf.convert` direction the JIT compilation of the resulting TensorFlow code ensures
that the performance characteristics of the code match those of the JAX source.
For the `call_tf` direction, JIT compilation is an essential part of the implementation
mechanism. Only TensorFlow functions that can be JIT-compiled can be called from
JAX. Since the TensorFlow functions that are produced by `jax2tf.convert` can
be JIT-compiled by design, we can round-trip from JAX to TensorFlow
(e.g., a SavedModel) and back.

We describe below some general concepts and capabilities, first for
`jax2tf.convert` and [later](#calling-tensorflow-functions-from-jax)
for `jax2tf.call_tf`.

More involved examples, including using jax2tf with
Flax models and their use with TensorFlow Hub and Keras, are described in the
[examples directory](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/README.md).

For details on saving a batch-polymorphic SavedModel see [below](#shape-polymorphic-conversion).

See also some internal ongoing design discussions at `go/jax2tf-doc`.

## Usage: converting basic functions.

As a rule of thumb, if you can `jax.jit` your function then you should be able
to use `jax2tf.convert`:

```python
import jax
from jax.experimental import jax2tf
from jax import numpy as jnp

import numpy as np
import tensorflow as tf

def f_jax(x):
  return jnp.sin(jnp.cos(x))

# jax2tf.convert is a higher order function that returns a wrapped function with
# the same signature as your input function but accepting TensorFlow tensors (or
# variables) as input.
f_tf = jax2tf.convert(f_jax)

# For example you execute f_tf eagerly with valid TensorFlow inputs:
f_tf(np.random(...))

# Additionally you can use tools like `tf.function` to improve the execution
# time of your function, or to stage it out to a SavedModel:
f_tf_graph = tf.function(f_tf, autograph=False)
```

The Autograph feature of `tf.function` cannot be expected to work on
functions converted from JAX as above, so it is recommended to
set `autograph=False` in order to avoid warnings or outright errors.

It is a good idea to use XLA to compile the converted function; that is
the scenario for which we are optimizing for numerical and performance
accuracy w.r.t. the JAX execution:

```python
tf.function(jax2tf.convert(f_jax), autograph=False, jit_compile=True)(x)
```

## Usage: saved model

Since jax2tf provides a regular TensorFlow function using it with SavedModel
is trivial:

```python
# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
# Save a function that can take scalar inputs.
my_model.f = tf.function(jax2tf.convert(f_jax), autograph=False,
                         input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory',
                    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load('/some/directory')
```

An important point is that in the above code snippet **everything is standard
TensorFlow code. In particular, the saving of the model is not directly part
of the jax2tf API, and the user has full control over how to create the SavedModel**.

Just like for regular TensorFlow functions, it is possible to include in the
SavedModel multiple versions of a function for different input shapes, by
"warming up" the function on different input shapes:

```python
my_model.f = tf.function(jax2tf.convert(f_jax), autograph=False)
my_model.f(tf.ones([1, 28, 28]))  # a batch size of 1
my_model.f(tf.ones([16, 28, 28]))  # a batch size of 16
tf.saved_model.save(my_model, '/some/directory',
                    options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
```

Some special care is needed to ensure that the model parameters are not embedded
as constants in the graph and are instead saved separately as variables.
This is useful for two reasons:
the parameters could be very large and exceed the limits of the
GraphDef part of the SavedModel, or you may want to fine-tune the
model and change the value of the parameters.

```python
def model_jax(params, inputs):
  return params[0] + params[1] * inputs

# Wrap the parameter constants as tf.Variables; this will signal to the model
# saving code to save those constants as variables.
params_vars = tf.nest.map_structure(tf.Variable, params)

# Build the prediction function by closing over the `params_vars`. If you
# instead were to close over `params` your SavedModel would have no variables
# and the parameters will be included in the function graph.
prediction_tf = lambda inputs: jax2tf.convert(model_jax)(params_vars, inputs)

my_model = tf.Module()
# Tell the model saver what are the variables.
my_model.variables = tf.nest.flatten(params_vars)
my_model.f = tf.function(prediction_tf, jit_compile=True)
tf.saved_model.save(my_model)
```

For examples of how to save a Flax model as a SavedModel see the
[examples directory](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/examples/README.md).


## Differentiation

The converted code supports differentiation from TensorFlow. In order to
ensure that the result of TensorFlow differentiation is identical to the
one that JAX differentiation would produce, the jax2tf converter will
annotate the converter function with a ``tf.custom_gradient`` that,
upon TensorFlow differentiation, will lazily
call into JAX to compute the ``jax.vjp`` of the converted function, followed by
jax2tf conversion. This ensures that ultimately it is JAX that performs the
differentiation, thus respecting any custom gradients that may be present
in the original function.

The jax2tf converter has an option ``with_gradient=False`` to skip the
custom gradients and wrap instead the converted function with
``tf.raw_ops.PreventGradient`` to generated an error in case a gradient
computation is attempted.

SavedModels enables saving custom derivative rules by using the `experimental_custom_gradients` option:

```
options = tf.saved_model.SaveOptions(experimental_custom_gradients=True)
tf.saved_model.save(model, path, options=options)
```

If you use `with_gradient=True` and forget to use the `experimental_custom_gradients=True` parameter
to `tf.saved_model.save` when you later load the saved model you will see a warning:

```
WARNING:absl:Importing a function (__inference_converted_fun_25) with ops with unsaved custom gradients. Will likely fail if a gradient is requested.
```

and if you do attempt to take a gradient of the loaded model you may get an error:

```
TypeError: An op outside of the function building code is being passed
a "Graph" tensor. It is possible to have Graph tensors
leak out of the function building context by including a
tf.init_scope in your function building code.
For example, the following function will fail:
  @tf.function
  def has_init_scope():
    my_constant = tf.constant(1.)
    with tf.init_scope():
      added = my_constant * 2
The graph tensor has name: args_0:0
```

(We are working with the TF team to give a more explicit error in this case.)


## Shape-polymorphic conversion

**The shape polymorphism support is work in progress. It is meant to be sound,
but it may fail to convert some programs. Please report any bugs you encounter.**

We described above how to include in the SavedModel several specializations
of a converted function for a few specific input shapes. The converter can
also produce a shape-polymorphic TensorFlow graph that is usable with inputs
of any shape matching
certain constraints. This is useful, e.g., to allow a single SavedModel
to be used for multiple batch sizes.

The standard TensorFlow technique for producing a shape-polymorphic graph is
to warm the `tf.function` on partially-specified (shape-polymorphic) inputs, e.g.,
`tf.TensorSpec([None, 28, 28], tf.float32)` for a function that processes a
batch (of unspecified batch size) of 28x28 images.
For jax2tf it is **additionally** necessary to specify an additional `polymorphic_shapes` parameter
for the `jax2tf.convert` function:

```
f_tf = tf.function(jax2tf.convert(f_jax,
                                  polymorphic_shapes=["(b, 28, 28)"]),
                                  autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, 28, 28], tf.float32))
```

The `polymorphic_shapes` parameter, in the form of a sequence of strings corresponding
to the sequence of positional
arguments, introduces one or more dimension variables, e.g., `b`, to stand for shape
dimensions that are assumed to be unknown at JAX tracing time, even if the actual
parameter value (here `tf.TensorSpec(...)`) happens to have fully known shape.
Dimension variables are assumed to range
over all strictly positive integers.
In this particular example, we can
also abbreviate `polymorphic_shapes=["(b, _, _)"]`,
because the `_` placeholders take their value
from the corresponding dimension of the `tf.TensorSpec` (which must be known).
As a further shortcut for a series of `_` at the end of a shape specification you can
use `...`: `polymorphic_shapes=["(b, ...)"]`.

In the example above, the `polymorphic_shapes` specification does
not convey more information than the partial `tf.TensorSpec`,
except that it gives a name to the unknown dimension, which improves
error messages. The real need for named shape
variables arises when there are
multiple unknown dimensions and there is a relationship between them.
For example,
if the function to be converted is also polymorphic on the size of each
image while requiring the images to be square,
we would add a dimension variable `d` to stand for
the unknown image size:

```
f_tf = tf.function(jax2tf.convert(f_jax, polymorphic_shapes=["(b, d, d)"]), autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, None, None], tf.float32))
```

The JAX tracing mechanism performs shape checking using the same strict rules as
when the shapes are fully known. For example, given the `"(b, d, d)"`
specification for the argument `x` of a function, JAX will know that a conditional
`x.shape[-2] == x.shape[-1]` is `True`, and will also know that `x` and `jnp.sin(x)` have the
same shape of a batch of square matrices that can be passed to `jnp.matmul`.


### Correctness of shape-polymorphic tracing

We want to trust that the converted program produces the same results as the
original JAX program. More precisely:

For any function `f_jax` and any input signature `abs_sig` containing partially
known `tf.TensorSpec`, and any concrete input `x` whose shape matches `abs_sig`:

 * If the conversion to TensorFlow succeeds: `f_tf = tf.function(jax2tf.convert(f_jax, polymorphic_shapes)).get_concrete_function(abs_sig)`
 * and if the TensorFlow execution succeeds with result `y`: `f_tf(x) = y`
 * then the JAX execution would produce the same result: `f_jax(x) = y`,

It is crucial to understand that `f_jax(x)` has the freedom to re-invoke the JAX tracing machinery,
and in fact it does so for each distinct concrete input shape, while the generation of `f_tf`
uses JAX tracing only once, and invoking `f_tf(x)` does not use JAX tracing anymore. In fact,
invoking the latter invocation may happen after the `f_tf` has been serialized
to a SavedModel and reloaded in an environment where `f_jax` and the JAX
tracing machinery are not available anymore.

Correctness is very important because it would be nasty to debug a subtle discrepancy
of the code loaded from a SavedModel from the expected behavior written in JAX.
We help ensure correctness
by reusing the same JAX tracing and shape checking mechanism as when the shapes are fully known.

### Coverage of shape-polymorphic tracing

Besides correctness, a secondary goal is to be able to convert many shape-polymorphic programs,
but at the very
least batch-size-polymorphic programs, so that one SavedModel can be used for any batch sizes.
For example, we want to ensure that any function written using `jax.vmap` at the top level can be
converted with the batch dimension polymorphic and the remaining dimensions concrete.

It is reasonable to expect that there will be JAX programs for which there is a
shape-polymorphic TensorFlow graph, but which will give an error when converting with jax2tf.

### Details

In order to be able to use shape polymorphism effectively with jax2tf, it
is worth considering what happens under the hood. When the converted function
is invoked with a `TensorSpec`, the jax2tf converter will combine the
`TensorSpec` from the actual argument with the `polymorphic_shapes` parameter to
obtain a shape abstraction to be used to specialize the converted function.
Normally, the shape abstraction contains the dimension sizes, but in the
presence of shape polymorphism, some dimensions may be dimension variables.

The `polymorphic_shapes` parameter must be either `None`,
or a sequence (one per argument) of shape specifiers.
(A value `None` for `polymorphic_shapes` is equivalent to a list of `None`.
See [how optional parameters are matched to arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).)
A shape specifier is combined with a `TensorSpec` as follows:

  * A shape specifier of `None` means that the shape is given
    by the actual argument `TensorSpec`, which must be fully known.
  * Otherwise, the specifier must be a comma-separated string of dimension specifiers: `(dim_1, ..., dim_n)`, denoting
    an n-dimensional array. The `TensorSpec` must also be of rank ``n``.
    An `...` at the end of the shape specifier is expanded to a list of `_` or appropriate length.
    The corresponding dimensions from the shape specifier and the `TensorSpec` are matched:

       * the dimension specifier of `_` means that the size of the dimension is given by
         the actual `TensorSpec`, which must have a known size in the corresponding dimension.
       * a dimension specifier can also be a lowercase identifier, denoting a dimension-size
         variable ranging over strictly positive integers.
         The abstract value of the dimension is going to be set to this variable.
         The corresponding dimension in `TensorSpec` can be `None` or can be a
         constant.
       * All occurrences of a dimension variable in any dimension
         for any argument are assumed to be equal.

Note that `polymorphic_shapes` controls the shape abstraction used by JAX when tracing
the function (with `_` placeholders given by the `TensorSpec`). The `TensorSpec`
gives the shape abstraction that TensorFlow will associate with the produced
graph, and can be more specific.

A few examples of shape specifications and uses:

  * `polymorphic_shapes=["(b, _, _)", None]` can be used for a function with two arguments, the first
    having a batch leading dimension that should be polymorphic. The other dimensions for the
    first argument and the shape of the second argument are specialized based on the actual
    `TensorSpec`, which must be known. The converted function can be used, e.g.,
    with `TensorSpec`s `[None, 28, 28]` and `[28, 16]` for the first and second argument
    respectively. An alternative `TensorSpec` pair can be `[1, 28, 28]` and `[28, 16]`,
    in which case the JAX tracing is done for the same polymorphic shape given by
    `polymorphic_shapes=["(b, 28, 28)", "(28, 16)"]`.

  * `polymorphic_shapes=["(batch, _)", "(batch,)"]`: the leading dimensions of the two arguments
     must match, and are assumed to be greater than 0.
     The second dimension of the first argument is taken from the
     actual `TensorSpec`. This can be used with a `TensorSpec` pair `[None, 16]`
     and `[None]`. It can also be used with a pair of shapes `[8, 16]` and `[8]`.

### Computing with dimension variables

JAX keeps track of the shape of all intermediate results. When those shapes depend
on dimension variables JAX computes them as multi-variate polynomials
involving dimension variables, which are assumed to range over strictly positive
integers.
The dimension polynomials have the following behavior for arithmetic operations:

  * addition, subtraction, multiplication are supported without restrictions, and
    are overloaded, such that `+`, `*`, `np.sum`, `np.prod` work directly on
    dimension polynomials.
    These arise, e.g., in `jax.numpy.concatenate` or `jax.numpy.reshape`.
  * division is a special case. It is also overloaded, but it is only partially
    supported, when either (a) there is no remainder, or (b) the divisor is a constant
    in which case there may be a constant remainder. The need for division in JAX core
    arises in a couple of specific situations, e.g.,
    `jax.numpy.reshape(-1)` and operations involving striding.
  * equality and disequality are partially supported. They result in a boolean value only when
    the same result would be obtained for any valuation of the dimension variables. In
    other situations, an exception `core.InconclusiveDimensionOperation` is raised.
    The latter would happen, e.g., when comparing `a == b` or `b == 1`.
    The `==` and `!=` operations are overloaded for dimension polynomials, to prevent
    an unsafe default behavior to be used.
  * inequality is partially supported, in a similar way as equality. However, in this
    case we take into consideration that dimension variables range over strictly positive
    integers. E.g., `b >= 1`, `b >= 0`, `2 * a + b >= 3` are `True`, while `b >= 2`,
    `a >= b`, `a - b >= 0` are inconclusive and result in an exception.

For example, the following code raises the exception
`core.InconclusiveDimensionOperation` with the message
`Dimension polynomial comparison 'a + 1' == 'b' is inconclusive`.

```
jax2tf.convert(lambda x: 0 if x.shape[0] + 1 == x.shape[1] else 1,
                polymorphic_shapes=["(a, b)"])(np.ones((3, 4))
```

Note that it would be unsound for JAX to compute `x.shape[0] + 1 == x.shape[1]`
as `False` and produce a converted function that returns `1` just because the dimension polynomials
are not identical: there are some concrete input shapes for which the function
should return `0`.

### Dimension variables appearing in the numeric computation

There are some situations when dimension variables arise in the staged computation itself.
You can see in the following example how elements from the input shapes
`(1024, 28, 28)` and `(28, 28)` appear in the computation and specifically
in the `shape` parameter of the `broadcast_in_dim` JAX primitive.

```
def image_mask_jax(images, mask):
  # images: f32[B, W, W]  and mask: f32[W, W]
  return images * mask

print(jax.make_jaxpr(image_mask_jax)(np.ones((1024, 28, 28)), np.ones((28, 28))))
>> { lambda  ; a b.
>>   let c = broadcast_in_dim[ broadcast_dimensions=(1, 2)
>>                            shape=(1, 28, 28) ] b
>>      d = mul a c
>>   in (d,) }

# The following will invoke broadcast_in_dim with shape=(1, w, w)
jax2tf.convert(image_mask_jax, polymorphic_shapes=["(b, w, w)", "(w, w)"])
```

When tracing and converting with abstract shapes some primitive parameters will be dimension variables
instead of just constants, e.g., the `shape` parameter of `broadcast_in_dim` will be `(1, w, w)`.
Note that JAX primitives distinguish the inputs, which are array values,
e.g., `b` for `broadcast_in_dim` above, and the parameters, e.g., `broadcast_dimensions` and `shape`.

The conversion of `image_mask_jax` would use `tf.shape` to compute the
values of the dimension variables `b` and `w`:

```
def image_mask_tf(images, mask):
  b, w, _ = tf.shape(images) # Compute the dynamic values for the dimension variables "b" and "w"
  return tf.math.multiply(images,
                          tf.broadcast_to(tf.reshape(mask, [1, w, w]),
                                          [b, w, w]))
```

To achieve this, when we start converting a function we construct a shape environment,
mapping the dimension variables in the `polymorphic_shapes` specification to TensorFlow expressions
using `tf.shape` on the input parameters.


### Errors in presence of shape polymorphism

In addition to the `InconclusiveDimensionOperation` error discussed above,
one may encounter other kinds of errors.

When tracing with shape polymorphism we can encounter shape errors:

```
four_ones = np.ones((4,))
jax2tf.convert(lambda x, y: x + y,
               polymorphic_shapes=["(v,)", "(4,)"])(four_ones, four_ones)
```

with result in the error `'add got incompatible shapes for broadcasting: (v,), (4,)'`
because the shape abstraction is given by the `polymorphic_shapes`, even though the
actual arguments are more specific and would actually work.

Also,
```
jax2tf.convert(lambda x: jnp.matmul(x, x),
             polymorphic_shapes=["(v, 4)"])(np.ones((4, 4)))
```

will result in the error `dot_general requires contracting dimensions to have the same shape, got [4] and [v]`. What is
happening here is that in the process of type checking the `matmul` operation, JAX
will want to ensure the size of the two axes is the same (`v == 4`).
Note that `v` can stand for any integer greater than 0, so the value of the
equality expression can be true or false. Since it is not always true
that `v == 4`, the shape checking rules fail with the above error.
Since the converted function works only for square matrices, the correct
`polymorphic_shapes` is `["(v, v)"]`.

You would also encounter shape errors if the code attempts to use the
dimension variables in unsupported arithmetic operations, such as in the code
below that fails to compute the inferred dimension for a `reshape` operations:

```
jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
               polymorphic_shapes=["(b, ...)"])(np.ones((4, 5, 7)))
```

In this case you will see the error `Cannot divide evenly the sizes of shapes (b, 5, 7) and (2, -1)`.
This is because the shape of `x` is `(b, 5, 7)`, with a total size represented as the
dimension polynomial `35 b`, which is not divisible by `2`.
Note that the following will succeed:

```
## The resulting symbolic shape is (2, 15 b).
jax2tf.convert(lambda x: jnp.reshape(x, (2, -1)),
               polymorphic_shapes=["(b, ...)"])(np.ones((4, 5, 6)))

## The resulting symbolic shape is (6 b2, b1).
jax2tf.convert(lambda x: jnp.reshape(x, (-1, x.shape[0])),
               polymorphic_shapes=["(b1, b2, ...)"])(np.ones((4, 5, 6)))
```

Finally, certain codes that use shapes in the actual computation may not yet work
if those shapes are polymorphic. In the code below, the expression `x.shape[0]`
will have the value of the dimension variable `v`. This case is not yet implemented:

```
jax2tf.convert(lambda x: jnp.sum(x, axis=0) / x.shape[0],
               polymorphic_shapes=["(v, _)"])(np.ones((4, 4)))
```

## Known issues

### Incomplete TensorFlow data type coverage

There are a number of cases when the TensorFlow ops that are used by the
jax2tf converter are not supported by TensorFlow for the same data types as in JAX.
There is an
[up-to-date list of unimplemented cases](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).

If you try to convert and run in TensorFlow a program with partially supported primitives, you may see TensorFlow errors that
a TensorFlow op is used with an supported data type, or that
there is no supported TensorFlow kernel for the op for the given
data type. The former case can happen even if you `jit_compile`
the TensorFlow program, and it is a priority to fit. The latter
case only appears in TensorFlow non-compiled mode and you can
avoid the problem if you use XLA to `jit_compile` (always recommended).

Our priority is to ensure numerical and performance accuracy for
the converted program **when using XLA to compile the converted program**.
It is always a good idea to use XLA on the JAX-converted function.

Sometimes you cannot compile the entire TensorFlow function for your
model, because in addition to the function that is converted from JAX,
it may include some pre-processing TensorFlow code that
is not compileable with XLA, e.g., string parsing. Even in those situations
you can instruct TensorFlow to compile only the portion that originates
from JAX:

```python
def entire_tf_fun(x):
  y = preprocess_tf_fun_not_compileable(x)
  # Compile the code that is converted from JAX
  z = tf.function(jax2tf.convert(compute_jax_fn),
                  autograph=False, jit_compile=True)(y)
  return postprocess_tf_fun_not_compileable(z)
```

You won't be able to compile the `entire_tf_fun`, but you can still execute
it knowing that the JAX-converted code is compiled. You can even save
the function to a SavedModel, knowing that upon restore the
JAX-converted code will be compiled.

For a more elaborate example, see the test `test_tf_mix_jax_with_uncompileable`
in [savedmodel_test.py](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/tests/savedmodel_test.py).

### Missing converter features

There is currently no support for `pmap` or`xmap`, nor for the collective
operations. There is support for `sharded_jit` and `pjit`.

### SavedModel may be large

If you suspect that the SavedModel is larger than it should be, check first
that you are not including the parameters as constants in the graph (see [above](#usage-saved-model)).

Additionally, the SavedModel obtained from a `jax2tf.convert`-ed function may include source
location information. This ensures that the debugging experience is similar
for JAX with XLA vs. `jax2tf.convert` with XLA. However, this debugging information
increases the size of the SavedModel, even possibly doubling it. You can
disable the generation of this metadata with the parameter
`include_xla_op_metadata`.

### SavedModel supports only first-order gradients

The `jax2tf`-converted function supports higher-order gradients, but when the
function is saved in a SavedModel, only the first-order gradient is saved.

### Converting gradients for functions with integer arguments or unused arguments

When JAX differentiates functions with integer or boolean arguments, the gradients will
be zero-vectors with a special `float0` type (see PR 4039](https://github.com/google/jax/pull/4039)).
This type is translated to `int32` when converting to TF.
For example,

```python
x = np.int16(2)
def f_jax(x):  # x: int16
  return x * 2.

jax.grad(f_jax, allow_int=True)(x)
# returns a special `float0`: array((b'',), dtype=[('float0', 'V')])

jax2tf.convert(jax.grad(f_jax, allow_int=True))(x))
# returns a tf.Tensor(0, shape=(), dtype=int32)
```

Note that this is different from how TensorFlow handles gradients
for integer or boolean arguments: sometimes the gradient is `None`,
sometimes it is a zero with the same dtype as the argument, and
sometimes it is a one with the same dtype as the argument (e.g.,
for the identity function).

```python
def f_tf(x):  # x: int16
  return tf.cast(x, tf.float32) * 2.

xv = tf.Variable(x)
with tf.GradientTape(persistent=True) as tape:
  print(tape.gradient(f_tf(xv), xv))
  # returns None
  print(tape.gradient(f_tf(xv), xv,
                      unconnected_gradients=tf.UnconnectedGradients.ZERO))
  # returns 0 with the same shape and dtype as x
```

When differentiating functions with unused arguments, TF by default
returns the value `None` for the corresponding gradients. The
`tape.gradient` function takes the option `tf.UnconnectedGradients.ZERO`
to ask that gradients for unused arguments be zero.

Functions converted with `jax2tf.convert` behave the same way under
`tf.UnconnectedGradients.ZERO`, but by default, they will return
`None` only for gradients corresponding to integer arguments.

```
# x1 and x3 are not used. x3 has integer type.
def fn(x0, x1, x2, x3):
  return x0 * 0. + x2 * 2.

xs = [tf.Variable(x) for x in [10., 11., 12., 13]]
with tf.GradientTape(persistent=True) as tape:
 res = fn(*xs)

g_tf_native = tape.gradient(res, xs)
# Returns: 0., None, 2., None

g_tf_native_0 = tape.gradient(res, xs,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
# Returns: 0., 0., 2., 0

# Now with jax2tf.convert
with tf.GradientTape() as tape:
  res = jax2tf.convert(fn, with_gradient=True)(*xs0

g_jax2tf = tape.gradient(res, xs)
# Returns: 0., 0., 2., None
# Note that the gradient for x1 is 0.

g_jaxx2tf_0 = tape.gradient(res, xs,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO)
# Returns: 0., 0., 2., 0
# In this case we get the same result as for TF native.
```

### Different 64-bit precision in JAX and TensorFlow

JAX behaves somewhat differently than TensorFlow in the handling
of 32-bit vs. 64-bit values. However, the `jax2tf.convert` function
always behaves like the JAX function.

JAX interprets the type of Python scalars differently based on
`JAX_ENABLE_X64` flag. (See
[JAX - The Sharp Bits: Double (64bit) precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).)
In the default configuration, the
flag is unset, and JAX interprets Python constants as 32-bit,
e.g., the type of `3.14` is `float32`. This is also what
TensorFlow always does. JAX goes further, it forces
all explicitly-specified 64-bit values to be interpreted as
32-bit:

```
# with JAX_ENABLE_X64=0
jnp.sin(3.14)  # Has type float32
tf.math.sin(3.14)  # Has type float32

jnp.sin(np.float64(3.14))  # Also has type float32
tf.math.sin(np.float64(3.14))  # Has type float64

# The jax2tf.convert function behaves like the JAX function.
jax2tf.convert(jnp.sin)(3.14)  # Has type float32
jax2tf.convert(jnp.sin)(np.float64(3.14))  # Has type float32

# The following will still compute `sin` in float32 (with a tf.cast on the argument).
tf.function(jax2tf.convert(jnp.sin))(tf.Variable(3.14, tf.float64))
```

When the `JAX_ENABLE_X64` flas is set, JAX uses 64-bit types
for Python scalars and respects the explicit 64-bit types:

```
# with JAX_ENABLE_X64=1
jnp.sin(3.14)  # Has type float64
tf.math.sin(3.14)  # Has type float32

# The jax2tf.convert function behaves like the JAX function.
jax2tf.convert(jnp.sin)(3.14)  # Has type float64

# The following will compute `sin` in float64.
tf.function(jax2tf.convert(jnp.sin))(tf.Variable(3.14, tf.float64))

# The following will compute `sin` in float32.
tf.function(jax2tf.convert(jnp.sin))(tf.Variable(3.14))
```

This is achieved by inserting `tf.cast` operations
on the input arguments inside the converted function,
if necessary.

If you want to create a `tf.Variable` or `tf.TensorSpec` with the
same dtype, you should use `jax2tf.dtype_of_val`:

```
# The following two calls will convert jax_fun at the same dtypes
# independently of the value of JAX_ENABLE_X64.
jax2tf.convert(jax_fun)(3.14)
jax2tf.convert(jax_fun)(tf.Variable(3.14, dtype=jax2tf.dtype_of_val(3.14))
```

### Unchecked assumption that the dimension variables take strictly positive values

The shape polymorphic conversion is sound with the assumption that the dimension
variables take non-zero values. In the following example, the function to be converted
has different behavior for empty shapes. The broken assumption is caught by jax2tf if
the converted function is executed eagerly, but not if it is first traced to a
TensorFlow graph:

```
def f_jax(x):
  return 0 if x.shape[0] == 0 else 1

x0 = np.array([], np.float32)
self.assertEqual(0, f_jax(x0))  # JAX sees that the x.shape[0] == 0

# jax2tf catches the broken assumption b >= 1 if the converted function is executed
# eagerly.
# Raises: ValueError: PolyShape ('b',) has dimension variable 'b' corresponding to 0, for argument shapes (TensorShape([0]),)
jax2tf.convert(f_jax, polymorphic_shapes=["b"])(x0))

# However, if we first trace to a TensorFlow graph, we may miss the broken assumption:
f_tf = tf.function(
        jax2tf.convert(f_jax, polymorphic_shapes=["b"])).get_concrete_function(tf.TensorSpec([None], dtype=np.float32))
self.assertEqual(1, f_tf(x0))
```

Another possible source of unsoundness is that JAX assumes that all unknown
dimensions represented by the same dimension variable have equal size. As before,
this assumption is checked if the converted function is executed eagerly, but
it may be missed if it is first traced to a TensorFlow graph:

```
def f_jax(x):
  return 0 if x.shape[0] != x.shape[1] else 1

x45 = np.ones((4, 5), dtype=np.float32)
self.assertEqual(0, f_jax(x45))  # JAX seems that x.shape[0] != x.shape[1]

# jax2tf catches the broken assumption x.shape[0] == x.shape[1] if the converted
# function is executed eagerly.
# Raises: ValueError: PolyShape ('b, b',) has dimension variable 'b' corresponding to multiple values {4, 5}, for argument shapes (TensorShape([4, 5]),)
jax2tf.convert(f_jax, polymorphic_shapes=["b, b"])(x45)

# However, if we first trace to a TensorFlow graph, we may miss the broken assumption.
f_tf = tf.function(
    jax2tf.convert(f_jax, polymorphic_shapes=["b, b"])).get_concrete_function(tf.TensorSpec([None, None], dtype=np.float32))
self.assertEqual(1, f_tf(x45))
```

### TensorFlow XLA ops

For most JAX primitives there is a natural TF op that fits the needed semantics.
There are a few (listed below) JAX primitives for which there is no
single TF op with matching semantics.
This is not so surprising, because JAX primitives have been designed
to be compiled to [HLO ops](https://www.tensorflow.org/xla/operation_semantics),
while the corresponding TF ops are sometimes higher-level.
For the cases when there is no matching canonical TF op,
we use a set of special TF ops that are thin wrappers over HLO ops
(a subset of those registered in
[tf2xla/ops/xla_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/ops/xla_ops.cc)
and implemented in,
e.g.,
[tf2xla/kernels/xla_pad_op.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/kernels/xla_pad_op.cc).)
We refer to these ops here as the XLA TF ops. Note that these are
still regular TF ops, e.g., they can be saved in a SavedModel.

There are several drawbacks of using XLA TF ops:

   * These ops will only be executable by a consumer that has XLA linked in.
   This should not be a problem for TPU execution, since that requires XLA anyway.
   But for other platforms (CPU, GPU, embedded) this can be a drawback in certain settings.
   * These ops are not yet recognized by tools that process
   tf.Graph, e.g., TensorFlow.js converter.

We use the following XLA TF ops:

   * `XlaPad` (wraps XLA Pad operator). We use this instead of `tf.pad` in order to
     support `lax.pad` interior padding (dilation) or negative edge padding.
   * `XlaConv2` (wraps XLA ConvGeneralDilated operator).
   * `XlaDotV2` (wraps XLA DotGeneral operator).
   * `XlaGather` (wraps XLA Gather operator). We could use `tf.gather` in some
     cases but not always. Also, `tf.gather` has a different semantics than `lax.gather`
     for index out of bounds.
   * `XlaScatter` (wraps XLA Scatter operator).
   * `XlaSelectAndScatter` (wraps XLA SelectAndScatter operator).
   * `XlaDynamicSlice` (wraps XLA DynamicSlice operator).
     We use this instead of `tf.slice` for reasons explained above for `XlaGather`.
   * `XlaDynamicUpdateSlice` (wraps XLA DynamicUpdateSlice operator).
   * `XlaReduceWindow` (wraps XLA ReduceWindow operator). These are used
     for `lax.reduce_window_sum_p`, `lax.reduce_window_min_p`,
     `lax.reduce_window_max_p`, and `lax.reduce_window_p`.
   * `XlaVariadicReduceV2` (for `lax.reduce`, `lax.argmin`, `lax.argmax`).
   * `XlaVariadicSort` (wraps XLA Sort operator).

### Different performance characteristics

The converted code may have slightly different performance characteristics than
the original JAX code.
We do expect that the performance characteristics of converted code
should approximate those of JAX when used with the XLA compiler (`tf.function(jit_compile=True)`).
This is because
during conversion we try to generate one TensorFlow op for one JAX primitive.
We expect that the lowering that XLA does is similar to that done by JAX
before conversion. (This is a hypothesis, we have not yet verified it extensively.)

There is one know case when the performance of the converted code will be different.
JAX programs use a [stateless
deterministic PRNG](https://github.com/google/jax/blob/main/design_notes/prng.md)
and it has an internal JAX primitive for it.
This primitive is at the moment converted to a soup of tf.bitwise operations,
which has a clear performance penalty. We plan to look into using the
HLO [RNGBitGenerator](https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator)
(exposed as a TFXLA op), which does implement
the same basic Threefry algorithm as JAX’s PRNG, although that would
result in different results than JAX’s PRNG.

In absence of TensorFlow XLA compilation,
if one were to write the same functionality in JAX idiomatic code vs.
native TensorFlow idiomatic code we could end up with very different compilation paths.
Take for example, the case of batch normalization.
In TensorFlow if one uses [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization),
a “high-level” TensorFlow op for batch
normalization is generated, and in the absence of XLA, on CPU or GPU,
a custom C++ “high-level” kernel implementing batch normalization is executed.
In JAX, there is no primitive for batch normalization, and instead the
operation is decomposed into low-level primitives (e.g., [flax.nn.BatchNorm](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.BatchNorm.html#flax.nn.BatchNorm),
or haiku.BatchNorm).
Once those primitives are converted to TensorFlow, and the resulting code is
run without XLA, the ensemble of the kernels executed will quite
possibly behave differently, performance-wise or even numerically,
than either the TensorFlow native or JAX native batch normalization.
A similar example is that of an LSTM cell.

# Calling TensorFlow functions from JAX

The function ```call_tf``` allows JAX functions to call
TensorFlow functions. These functions can be called anywhere in a JAX
computation, including in staging contexts ``jax.jit``, ``jax.pmap``, ``jax.xmap``,
or inside JAX's control-flow primitives. In non-staging contexts,
the TensorFlow function is called in eager mode.
For now, only reverse-mode autodiff is supported for these functions
(no forward-mode autodiff, nor ``vmap``).

As a trivial example, consider computing ``sin(cos(1.))`` with ``sin`` done in JAX and ``cos`` in TF:

```python
  from jax.experimental import jax2tf

  # This is a TF function. It will be called with TensorFlow-compatible arguments,
  # such as `numpy.ndarray`, `tf.Tensor` or `tf.Variable`, or a pytree thereof.
  # It should return a similar result. This function will be called using
  # TensorFlow eager mode if called from outside JAX staged contexts (`jit`,
  # `pmap`, or control-flow primitives), and will be called using TensorFlow
  # compiled mode otherwise. In the latter case, the function must be compileable
  # with XLA (`tf.function(func, jit_compile=True)`)
  def cos_tf(x):
    return tf.math.cos(x)

  # Compute cos with TF and sin with JAX
  def cos_tf_sin_jax(x):
    return jax.numpy.sin(jax2tf.call_tf(cos_tf)(x))

  # Calls `cos_tf` in TF eager mode
  x = np.float32(1.)
  cos_tf_sin_jax(x)

  # Compiles `cos_tf` using TF and embeds the XLA computation into the JAX
  # XLA computation (containing `sin`). The XLA compiler may even be able to
  # fuse through JAX-TF computations.
  jax.jit(cos_tf_sin_jax)(x)

  # Uses TF gradient for `cos_tf` and JAX gradient for `sin`
  jax.grad(cos_tf_sin_jax)(x)
```

If you inspect the generated HLO for ``cos_tf_sin_jax`` you will see that the
main JAX computation (``ENTRY xla_computation_cos_tf_sin_jax``) makes a call to
the ``a_inference_cos_tf_68__``HLO function that was compiled by TF from ``cos_tf``:

```
    HloModule xla_computation_cos_tf_sin_jax.18

    a_inference_cos_tf_68__.4 {
      arg0.5 = f32[] parameter(0), parameter_replication={false}
      reshape.6 = f32[] reshape(arg0.5)
      cosine.7 = f32[] cosine(reshape.6)
      reshape.8 = f32[] reshape(cosine.7)
      tuple.9 = (f32[]) tuple(reshape.8)
      ROOT get-tuple-element.10 = f32[] get-tuple-element(tuple.9), index=0
    }

    ENTRY xla_computation_cos_tf_sin_jax.18 {
      constant.2 = pred[] constant(false)
      constant.3 = pred[] constant(false)
      parameter.1 = f32[] parameter(0)
      call.11 = f32[] call(parameter.1), to_apply=a_inference_cos_tf_68__.4
      tuple.12 = (f32[]) tuple(call.11)
      get-tuple-element.13 = f32[] get-tuple-element(tuple.12), index=0
      tuple.14 = (f32[]) tuple(get-tuple-element.13)
      get-tuple-element.15 = f32[] get-tuple-element(tuple.14), index=0
      sine.16 = f32[] sine(get-tuple-element.15)
      ROOT tuple.17 = (f32[]) tuple(sine.16)
    }
```

For a more elaborate example, including round-tripping from JAX
to TensorFlow and back through a SavedModel, with support for
custom gradients,
see the test `test_round_trip_custom_grad_saved_model`
in [call_tf_test.py](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/tests/call_tf_test.py).

All the metadata inserted by TF during tracing and compilation, e.g.,
source location information and op names, is carried through to the
JAX XLA computation.

The TF custom gradients are respected, since it is TF that generates the
gradient computation.

In op-by-op mode, when we call TensorFlow in eager mode, we use
DLPack to try to avoid copying the data. This works for CPU (for
DeviceArray data or for np.ndarray that are aligned on 16-byte
boundaries) and on GPU (for DeviceArray).
The zero-copy does not yet work on TPU.


### Limitations of call_tf

The TF function must be compileable (`tf.function(func, jit_compile=True`)
when used in a JAX staging context, e.g., `jax.jit`, `lax.scan`, `lax.cond`,
but not when used in a JAX op-by-op mode. For example, the following
function uses strings operations that are not supported by XLA:

```python
def f_tf_non_compileable(x):
   return tf.strings.length(tf.strings.format("Hello {}!", [x]))

f_jax = jax2tf.call_tf(f_tf_non_compileable)
# Works in op-by-op mode
f_jax(np.float32(42.))

# Fails in jit mode
jax.jit(f_jax)(np.float(42.))
```

Another similar situation is when a function uses input values in
place of shapes. In this case TF actually does compile the function
but re-compiles it for each distinct value of the input. This is
not allowed when used from JAX:

```python
def f_tf_dynamic_shape(x):
  return x[x[0]:5]
x = np.array([1, 2], dtype=np.int32)

f_jax = jax2tf.call_tf(f_tf_dynamic_shape)
# Works in op-by-op mode
f_jax(x)

# Fails in jit mode
jax.jit(f_jax)(x)
```

``call_tf`` works best with pure TF functions that do not capture
``tf.Variable``s or tensors from the environment, and all such
context is passed in explicitly through arguments, and if variables
are modified, the resulting values are passed out through results.
There is a best-effort mechanism that can handle variable capture
and variable updates,
except in the case of a function that modifies ``tf.Variable``s
and is used in a JAX jitted context. Calling the ``inpure_func_tf``
will give an error:

```python
       var1 = tf.Variable(1.)
       def impure_func_tf(x):
         var1.write(11.)  # BAD: should not write to variables
         return x + var1
       jax2tf.call_tf(impure_func_tf)(tf.constant(2.))  # Works in eager mode
       jax.jit(jax2tf.call_tf(impure_func_tf))(tf.constant(2.))  # Fails in jit mode
```

The error can be avoided by passing the variable explicitly:

```python
       def pure_func_tf(x, var1)
          new_var1 = 11.
          return x + new_var1, new_var1
```

This use case is likely to be revisited.

A TF function wrapped with `call_tf` cannot be applied to inputs whose
shapes are not constants. The may arise when you try to apply
`jax2tf.convert` with polymorphic shapes on the result of
`call_tf`:

```python
def fun_jax(x):
  return jax2tf.call_tf(tf.math.sin)(x)

# The following will throw an error.
jax2tf.convert(fun_jax, polymorphic_shapes=["b, ..."])(x)
```

This is unsatisfying, because the result of the above conversion
could be simply `tf.math.sin`, which is batch polymorphic. But
JAX cannot keep track of shapes through a `call_tf` call, and it
cannot be sure that the shape-polymorphic conversion is safe.

# Misc notes

## TensorFlow versions supported

The ``jax2tf.convert`` and `call_tf` require very recent versions of TensorFlow.
As of today, the tests are run using `tf_nightly==2.7.0.dev20210715`.

## Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](https://github.com/google/jax/blob/main/README.md#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).

## Updating the limitations documentation

The jax2tf tests are parameterized by a set of limitations
(see `tests/primitive_harness.py` and `tests/jax2tf_limitations.py`).
The limitations specify test harnesses that are known to fail, by
JAX primitive, data type, device type, and TensorFlow execution mode (`eager`,
`graph`, or `compiled`). These limitations are also used
to generate tables of limitations, e.g.,

   * [List of primitives not supported in JAX](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/jax_primitives_coverage.md),
     e.g., due to unimplemented cases in the XLA compiler, and
   * [List of primitives not supported in jax2tf](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md),
     e.g., due to unimplemented cases in TensorFlow. This list is incremental
     on top of the unsupported JAX primitives.

There are instructions for updating those documents at the end of each
document.

The set of limitations is an over-approximation, in the sense that if XLA
or TensorFlow improves and support more cases, no test will fail. Instead,
periodically, we check for unnecessary limitations. We do this by uncommenting
two assertions (in `tests/jax_primitives_coverage_test.py` and in
`tests/tf_test_util.py`) and runing all the tests. With these assertions enabled
the tests will fail and point out unnecessary limitations. We remove limitations
until the tests pass. Then we re-generate the documentation.
