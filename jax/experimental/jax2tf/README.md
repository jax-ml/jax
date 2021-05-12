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
[examples directory](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/examples/README.md).

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

## Usage: saved model

Since jax2tf provides a regular TensorFlow function using it with SavedModel
is trivial:

```python
# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
# Save a function that can take scalar inputs.
my_model.f = tf.function(jax2tf.convert(f_jax), input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory')

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
tf.saved_model.save(my_model, '/some/directory')
```

For examples of how to save a Flax model as a SavedModel see the
[examples directory](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/examples/README.md).


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

Currently, there is a bug that prevents using custom gradients with SavedModel
(see [Caveats](#caveats) below).

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
to warm the function on partially-specified (shape-polymorphic) inputs, e.g.,
`tf.TensorSpec([None, 28, 28], tf.float32)` for a function that processes a
batch (of unspecified batch size) of 28x28 images.
For jax2tf it is also necessary to specify an additional `polymorphic_shapes` parameter
for the `jax2tf.convert` function:

```
f_tf = tf.function(jax2tf.convert(f_jax,
                                  polymorphic_shapes=["(b, 28, 28)"]),
                                  autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, 28, 28], tf.float32))
```

The `polymorphic_shapes` parameter, in the form of a list of strings corresponding
to the list of
arguments, introduces one or more shape variables, e.g., `b`, to stand for shape
dimensions that are unknown at JAX tracing time.
In this particular example, we can
also use the `polymorphic_shapes=["(b, _, _)"]`,
because the `_` placeholders take their value
from the corresponding dimension of the `tf.TensorSpec` (which must be known).
As a shortcut for a series of `_` at the end of a shape specification you can
use `...`: `polymorphic_shapes=["(b, ...)"]`

In the example above, the `polymorphic_shapes` specification does
not convey more information than the partial `tf.TensorSpec`,
except that it gives a name to the unknown dimension so that it
can be recognized in the error messages. The need for named shape
variables arises when there are
multiple unknown dimensions and there is a relationship between them.
For example,
if the function to be converted is also polymorphic on the size of each
image while requiring the images to be square,
we would add a shape variable `d` to stand for
the unknown image size:

```
f_tf = tf.function(jax2tf.convert(f_jax, polymorphic_shapes=["(b, d, d)"]), autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, None, None], tf.float32))
```

The JAX tracing mechanism performs shape checking using the same strict rules as
when the shapes are fully known. For example, given the `"(b, d, d)"`
specification for the argument `x` of a function, JAX will know that a conditional
`x.shape[-2] == x.shape[-1]` is `True`, will know that `x` and `jnp.sin(x)` have the
same shape of a batch of square matrices that can be passed to `jnp.matmul`.


### Correctness of shape-polymorphic tracing

We want to trust that the converted program produces the same results as the
original JAX program:

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
of the code running in production from the expected behavior written in JAX.
We help ensure correctness
by reusing the same JAX tracing and shape checking mechanism as when the shapes are fully known.

### Coverage of shape-polymorphic tracing

A complementary goal is to be able to convert many shape-polymorphic programs, but at the very
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
    The
    corresponding dimensions from the shape specifier and the `TensorSpec` are matched:

       * the dimension specifier of `_` means that the size of the dimension is given by
         the actual `TensorSpec`, which must have a known size in the corresponding dimension.
       * a dimension specifier can also be a lowercase identifier, denoting a dimension-size
         variable ranging over strictly positive integers.
         The abstract value of the dimension is going to be set to this variable.
         The corresponding dimension in `TensorSpec` can be `None` or can be a
         constant.
       * All occurrences of a shape variable in any dimension
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
    `polymorphic_shapes=["(b, 28, 28)", "(28, 16)"]` but the TensorFlow graph is monomorphic
    for the shapes given by `TensorSpec`.

  * `polymorphic_shapes=["(batch, _)", "(batch,)"]`: the leading dimensions of the two arguments
     must match, and are assumed to be greater than 0.
     The second dimension of the first argument is taken from the
     actual `TensorSpec`. This can be used with a `TensorSpec` pair `[None, 16]`
     and `[None]`. It can also be used with a pair `[8, 16]` and `[5]`.

### Shape variables used in the computation

There are some situations when shape variables arise in the computation itself.
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

# will invoke broadcast_in_dim with shape=(1, w, w)
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
  b, w, _ = tf.shape(images) # Compute the dynamic values for the shape variables "b" and "w"
  return tf.math.multiply(images,
                          tf.broadcast_to(tf.reshape(mask, [1, w, w]),
                                          [b, w, w]))
```

To achieve this, when we start converting a function we construct a shape environment,
mapping the shape variables in the `polymorphic_shapes` specification to TensorFlow expressions
using `tf.shape` on the input parameters.


### Errors in presence of shape polymorphism

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

will result in the error `Shape variable comparison v == 4 is inconclusive`. What is
happening here is that in the process of type checking the `matmul` operation, JAX
will want to ensure the size of the two axes is the same (`v == 4`).
Note that `v` can stand for any integer greater than 0, so the value of the
equality expression can be true or false. In this case you will see
the `core.InconclusiveDimensionOperation` exception with the above message.
Since the converted function work only for square matrices, the correct
`polymorphic_shapes` is `["(v, v)"]`.

You would also encounter shape errors if the code attempts to use the
dimension variables in arithmetic operations, such as in the code
below that attempts to flatten an array with a polymorphic batch
dimension:

```
jax2tf.convert(lambda x: jnp.reshape(x, np.prod(x.shape)),
             polymorphic_shapes=["(b, ...)"])(np.ones((3, 4, 5)))
```

In this case you will see the error `TypeError: unsupported operand type(s) for *: 'DimVar' and 'int'`.
The most flattening you can do is on the known dimensions, keeping the variable
dimension intact:

```
jax2tf.convert(lambda x: jnp.reshape(x, (x.shape[0], np.prod(x.shape[1:]))),
               polymorphic_shapes=["(b, _, _)"])(np.ones((3, 4, 5)))
```


Finally, certain codes that use shapes in the actual computation may not yet work
if those shapes are polymorphic. In the code below, the expression `x.shape[0]`
will have the value of the shape variable `v`. This case is not yet implemented:

```
jax2tf.convert(lambda x: jnp.sum(x, axis=0) / x.shape[0],
               polymorphic_shapes=["(v, _)"])(np.ones((4, 4)))
```

## Caveats

### Incomplete TensorFlow data type coverage

There are a number of cases when the TensorFlow ops that are used by the
jax2tf converter are not supported by TensorFlow for fewer data types than JAX.
There is an
[up-to-date list of unimplemented cases](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/g3doc/primitives_with_limited_support.md).

### Missing features

There is currently no support for replicated (e.g. `pmap`) or multi-device
(e.g. `sharded_jit`) functions. The collective operations are not yet handled.

### No SavedModel fine-tuning

Currently, TensorFlow SavedModel does not properly save the `tf.custom_gradient`.
It does save however some attributes that on model restore result in a warning
that the model might not be differentiable, and trigger an error if differentiation
is attempted. The plan is to fix this. Note that if no gradients are requested,
the PreventGradient ops will be saved along with the converted code and will
give a nice error if differentiation of the converted code is attempted.

### Converting gradients for integer-argument functions

When JAX differentiates over functions with integer arguments, the gradients will
be zero-vectors with a special `float0` type (see PR 4039](https://github.com/google/jax/pull/4039)).
This type is translated to `bfloat16` when converting to TF. For example,

```python
def f_jax(x):  # x: int32
  return x * 2.

jax.grad(f_jax, allow_int=True)(2)
# returns a special `float0`: array((b'',), dtype=[('float0', 'V')])

jax2tf.convert(jax.grad(f_jax, allow_int=True))(2))
# returns a `bfloat16` zero: tf.Tensor(0, shape=(), dtype=bfloat16)
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
We refer to these ops here as the TFXLA ops.

There are several drawbacks of using TFXLA ops:

   * These ops will only be executable by a consumer that has XLA linked in.
   This should not be a problem for TPU execution, since that requires XLA anyway.
   But for other platforms (CPU, GPU, embedded) this can be a drawback in certain settings.
   * These ops are not yet recognized by tools that process
   tf.Graph, e.g., TensorFlow.js converter.

We use the following TFXLA ops:

   * `XlaPad` (wraps XLA Pad operator). We use this instead of `tf.pad` in order to
     support `lax.pad` interior padding (dilation) or negative edge padding.
   * `XlaConv` (wraps XLA ConvGeneralDilated operator).
   * `XlaDot` and `XlaDotV2` (wraps XLA DotGeneral operator).
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
deterministic PRNG](https://github.com/google/jax/blob/master/design_notes/prng.md)
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
  # graph mode otherwise. In the latter case, the function must be compileable
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

## Notes:

  * The TF function must be compileable (`tf.function(func, jit_compile=True`)
    when used in a JAX staging context. 
  * All the metadata inserted by TF during tracing and compilation, e.g.,
    source location information and op names, is carried through to the
    JAX XLA computation.
  * The TF custom gradients are respected, since it is TF that generates the
    gradient computation.
  * In op-by-op mode, when we call TensorFlow in eager mode, we use 
    DLPack to try to avoid copying the data. This works for CPU (for
    DeviceArray data or for np.ndarray that are aligned on 16-byte
    boundaries) and on GPU (for DeviceArray). 
    The zero-copy does not yet work on TPU.   
  * ``call_tf`` works best with pure TF functions that do not capture
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

## TODO

  * Ensure that there is no array copy through the host when running in eager
    mode (JAX op-by-op).
  * Show how use ``call_tf`` to load a SavedModel into JAX.


# Additional notes

## TensorFlow versions supported

The ``jax2tf.convert`` and `call_tf` require very recent versions of TensorFlow.
As of today, the tests are run using `tf_nightly==2.5.0-dev20210315`.

## Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](https://github.com/google/jax/blob/master/README.md#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).
