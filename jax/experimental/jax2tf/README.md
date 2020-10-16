# JAX to TensorFlow converter

This package provides an experimental JAX converter that can take a function
written in JAX, possibly including JAX transformations, and turn it into
a function that uses only TensorFlow operations. The converted function
can be used in a TensorFlow context and will behave as if it was written in TensorFlow.
In practice this means that you can take some code written in JAX and execute it using
TensorFlow eager mode, or stage it out as a TensorFlow graph, even save it
as a SavedModel for use with TensorFlow tools such as serving tools,
or TensorFlow Hub.

### Usage: converting basic functions.

We describe below some simple usage scenarios. More involved examples, including
Flax models and their use with TensorFlow Hub and Keras, are described in the
[getting started Colab](JAX2TF_getting_started.ipynb).

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

### Usage: saved model

Since jax2tf provides a regular TensorFlow function using it with SavedModel
is trivial.

```python
# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
# Save a function that can take scalar inputs.
my_model.f = tf.function(f_tf, input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory')

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load('/some/directory')
```

Just like for regular TensorFlow functions, it is possible to include in the
SavedModel multiple versions of a function for different input shapes, by
"warming up" the function on different input shapes:

```
my_model.f = tf.function(jax2tf.convert(f_jax), autograph=False)
my_model.f(tf.ones([1, 28, 28, 1]))  # a batch size of 1
my_model.f(tf.ones([16, 28, 28, 1]))  # a batch size of 16
```

More involved examples of using SavedModel are described in the
[getting started Colab](JAX2TF_getting_started.ipynb).

## Differentiation

The converted code supports differentiation from TF.
The main challenge with TF-differentiation of the converted code is that some
of the JAX primitives or functions that were used in the original JAX code might
have had JAX custom gradients. One example is the ``jax.nn.relu``, which
at 0 has a JAX custom gradient of 0, but the primitive-by-primitive conversion
to TF is not mathematically differentiable at 0 and may generate another value. In this
particular case the TF differentiation of the raw conversion returns 1.
If we were to use ``tf.nn.relu``, then we would get a correct custom TF gradient of 0,
because ``tf.nn.relu`` has a TF custom gradient.

Other challenges for differentiation are that some of the TF ops used in translation
do not yet have differentiation rules defined.
For other ops, we would have to ensure that they match the JAX differentiation rules.

All of these problems are solved by having the converter annotate the converted
function with a ``tf.custom_gradient`` that, upon TF differentiation, will lazily
call into JAX to compute the ``jax.vjp`` of the converted function, followed by
jax2tf conversion.
This ensures that JAX’s differentiation uses the JAX rules and custom gradients.
In particular, TF’s internal op differentiation rules will not be used at all,
and we need not worry about ops not having differentiation rules that match JAX's.
The jax2tf converter has an option to skip the custom gradients and wrap
instead the converted function with ``tf.raw_ops.PreventGradient`` to generated an
error in case a gradient computation is attempted.

Currently there is a bug that prevents using custom gradients with SavedModel
(see Caveats below).

## Caveats

1.  Some JAX primitives are converted into
[special TensorFlow ops](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/tf2xla/ops/xla_ops.cc)
  that are thin wrapper over XLA ops. For this reason, the graphs produced by jax2tf
  require XLA in order to run. Also, these ops may not be recognized by
  SavedModel converters, such as the TensorFlow.js converter.
  We use the following such operations:

     * ``XlaPad`` (wraps XLA Pad operator). We use this instead of ``tf.pad`` in order to
     support ``lax.pad`` interior padding (dilation) or negative edge padding.
     * ``XlaConv`` (wraps XLA ConvGeneralDilated operator).
     * ``XlaGather`` (wraps XLA Gather operator). We could use ``tf.gather`` in some
     cases but not always. Also, ``tf.gather`` has a different semantics than ``lax.gather``
     for index out of bounds.
     * ``XlaScatter`` (wraps XLA Scatter operator).
     * ``XlaSelectAndScatter`` (wraps XLA SelectAndScatter operator).
     * ``XlaDynamicSlice`` (wraps XLA DynamicSlice operator).
     We use this instead of ``tf.slice`` for reasons explained above for ``XlaGather``.
     * ``XlaDynamicUpdateSlice`` (wraps XLA DynamicUpdateSlice operator).
     * ``XlaReduceWindow`` (wraps XLA ReduceWindow operator). These are used
     for ``lax.reduce_window_sum_p``, ``lax.reduce_window_min_p``,
     ``lax.reduce_window_max_p``, and ``lax.reduce_window_p``.
     * ``XlaSort`` (wraps XLA Sort operator).

2.  A small number of JAX primitives are not yet converted, or are converted only
    for certain data types. The main
    reason is that the required TensorFlow ops are not implemented for certain
    data types on certain devices. There is an
    [up-to-date list of unimplemented cases](primitives_with_limited_support.md).

3.  Currently, TF SavedModel does not properly save the ``tf.custom_gradient``.
    It does save however some attributes that on model restore result in a warning
    that the model might not be differentiable, and trigger an error if differentiation
    is attempted. The plan is to fix this. Note that if no gradients are requested,
    the PreventGradient ops will be saved along with the converted code and will
    give a nice error if differentiation of the converted code is attempted.

4.  There is currently no support for replicated (e.g. `pmap`) or multi-device
    (e.g. `sharded_jit`) functions.

5.  The converted code may have slightly different performance characteristics than
    the original JAX code.
    If one were to write the same functionality in JAX idiomatic code vs.
    native TF idiomatic code we could end up with very different compilation paths,
    when TF is used without XLA. Take for example, the case of batch normalization.
    In TF if one uses [tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization),
    a “high-level” TF op for batch
    normalization is generated, and in the absence of XLA, on CPU or GPU,
    a custom C++ “high-level” kernel implementing batch normalization is executed.
    In JAX, there is no primitive for batch normalization, and instead the
    operation is decomposed into low-level primitives (e.g., [flax.nn.BatchNorm](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.BatchNorm.html#flax.nn.BatchNorm),
    or haiku.BatchNorm).
    Once those primitives are converted to TF, and the resulting code is
    run without XLA, the ensemble of the kernels executed will quite
    possibly behave differently, performance-wise or even numerically,
    than either the TF native or JAX native batch normalization.
    A similar example is that of an LSTM cell.

    Yet another example are the PRNG primitives. JAX programs use a [stateless
    deterministic PRNG](https://github.com/google/jax/blob/master/design_notes/prng.md)
    and it has an internal JAX primitive for it.
    This primitive is at the moment converted to a soup of tf.bitwise operations,
    which has a clear performance penalty. We plan to look into using the
    HLO [RNGBitGenerator](https://www.tensorflow.org/xla/operation_semantics#rngbitgenerator)

    (exposed as a TFXLA op), which does implement
    the same basic Threefry algorithm as JAX’s PRNG, although that would
    result in different results than JAX’s PRNG.

    We do expect that the performance characteristics of converted code
    should approximate those of JAX or TF native with XLA. This is because
    during conversion we try to generate one TF op for one JAX primitive.
    We expect that the lowering that XLA does is similar to that done by JAX
    before conversion. (This is a hypothesis, we have not verified it extensively.)

### Shape-polymorphic conversion

**The shape polymorphism support is work in progress. Please report any bugs you encounter.**

We described above how to include in the SavedModel several specializations
of a converted function for a few specific input shapes. The converter can also produce
a shape-polymorphic TensorFlow graph that is usable with inputs of any shape matching
certain constraints. This is useful, e.g., to allow a SavedModel to be used for multiple
batch sizes.

The standard TensorFlow technique for producing a shape-polymorphic graph is
to warm the function on partially-specified (shape polymorphic) inputs.
For example, if `f_tf` is a TensorFlow function taking a batch of images,
represented as 4D arrays, then:

```
tf.function(f_tf).get_concrete_function(tf.TensorSpec([None, 28, 28, 1], tf.float32))
```

will generate and cache a TensorFlow graph that is expected to work for any
value of the leading dimension. We use below the term `TensorSpec` to refer to
a *TensorFlow shape specification*, which can be written directly as above,
possibly including `None` for some dimensions, or can be derived implicitly
from actual NumPy arrays or `tf.Tensor` passed as actual arguments.

One challenge for the jax2tf converter is that TensorFlow's shape checking mechanism
is too permissive in presence of polymorphic shapes. For example, the
function `lambda x: x + tf.transpose(x)` only makes sense for square matrices,
but this constraint is not expressible in TensorFlow; we must use the more
permissive `[None, 16]` shape. The mental model here is that graph generation
will succeed if the polymorphic `TensorSpec` includes *at least one shape* for
which the graph is well-defined.
If the graph is used for other shapes there will be a shape error at runtime.

JAX includes *experimental* support for shape polymorphism but the shape checking
rules are stricter than in TensorFlow: we must give the jax2tf converter a specification
of what shapes to specialize the function to, such that the specialization is guaranteed
to work for *any shape* that matches the specification. For the above example, one can
obtain a graph that works for all 2D *square* matrices as follows:

```
f_jax = lambda x: x + x.T
f_tf = tf.function(jax2tf.convert(f_jax, in_shapes=["(b, b)"]), autograph=False)
f_tf.get_concrete_function(tf.TensorSpec([None, None], tf.float32))
```

The novel element here is ```in_shapes=["(b, b)"]```, which introduces
a dimension variable ```b``` and specifies that the first input is 2D with
both dimension sizes equal to ```b```. Note that the most precise `TensorSpec`
available to capture all 2D square matrices is `[None, None]`, which is
strictly more permissive (TensorFlow does not currently have a notation to
specify shapes whose dimensions have constraints).

In order to be able to use shape polymorphism effectively with jax2tf, it
is worth considering what happens under the hood. When the converted function
is invoked with a `TensorSpec`, the jax2tf converter will combine the
`TensorSpec` from the actual argument with the `in_shapes` parameter to
obtain a shape abstraction to be used to specialize the converted function.
Normally, the shape abstraction contains the dimension sizes, but in the
presence of shape polymorphism, some dimensions may be polynomials
of dimension variables.

The `in_shapes` parameter must be either `None`,
or a sequence (one per argument) of shape specifiers.
(A value `None` for `in_shapes` is equivalent to a list of `None`.
See [how optional parameters are matched to arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).)
A shape specifier is combined with a `TensorSpec` as follows:

  * A shape specifier of `None` means that the shape is given
    by the actual `TensorSpec`, which must be fully known.
  * Otherwise, the specifier must be a comma-separated string of the form `(dim_1, ..., dim_n)`, denoting
    an n-dimensional array. The `TensorSpec` must also be of rank ``n``. The
    corresponding dimensions from the shape specifier and the `TensorSpec` are matched:

       * the dimension specifier of `_` means that the value of the dimension is given by
         the actual `TensorSpec`, which have a constant in the corresponding dimension.
       * a dimension specifier can also be a lowercase identifier, denoting a dimension-size
         variable. The abstract value of the dimension is going to be set to this variable.
         The corresponding dimension in `TensorSpec` should be `None` or can be a
         constant.
       * a dimension specifier can also be a polynomial expression involving dimension
         variables. The expression can involve `+`, `*`, integer constants with optional negation
         sign, and dimension variables. All occurrences of a dimension variable in any dimension
         for any argument are assumed to be equal. The corresponding dimension in `TensorSpec`
         should be `None` or can be a known constant.

Note that `in_shapes` controls the shape abstraction used by JAX when tracing
the function (with `_` placeholders given by the `TensorSpec`). The `TensorSpec`
gives the shape abstraction that TensorFlow will associate with the produced
graph, and can be more specific.

A few examples of shape specifications and uses:

  * `in_shapes=["(b, _, _)", None]` can be used for a function with two arguments, the first
    having a batch leading dimension that should be polymorphic. The other dimensions for the
    first argument and the shape of the second argument are specialized based on the actual
    `TensorSpec`, which must be known. The converted function can be used, e.g.,
    with `TensorSpec`s `[None, 28, 28]` and `[28, 16]` for the first and second argument
    respectively. An alternative `TensorSpec` pair can be `[1, 28, 28]` and `[28, 16]`,
    in which case the JAX tracing is done for the same polymorphic shape given by
    `in_shapes=["(b, 28, 28)", "(28, 16)"]` but the TensorFlow graph is monomorphic
    for the shapes given by `TensorSpec`.

  * `in_shapes=["(batch, _)", "(batch,)"]`: the leading dimensions of the two arguments
     must match. The second dimension of the first argument is taken from the
     actual `TensorSpec`. This can be used with a `TensorSpec` pair `[None, 16]`
     and `[None]`. It can also be used with a pair `[8, 16]` and `[5]`. (TODO:
     add some checking that the `TensorSpec` has at least a satisfying solution
     for the dimension variables.)

#### Errors in presence of shape polymorphism

When tracing with shape polymorphism we can encounter shape errors:

```
four_ones = np.ones((4,))
jax2tf.convert(lambda x, y: x + y,
               in_shapes=["(v,)", "(4,)"])(four_ones, four_ones)
```

with result in the error 'add got incompatible shapes for broadcasting: (v,), (4,)'
because the shape abstraction is given by the `in_shapes`, even though the
actual arguments are more specific and would actually work.

Also,
```
jax2tf.convert(lambda x: jnp.matmul(x, x),
             in_shapes=["(v, 4)"])(np.ones((4, 4)))
```

will result in the error 'dot_general requires contracting dimensions to have the same shape, got [4] and [v].'.
Since the converted function work only for square matrices, the correct
`in_shapes` is `["(v, v)"]`.

You may also encounter shape errors due to not-yet-implemented shape-polymorphism
rules for JAX primitives:

```
jax2tf.convert(lambda x: jnp.split(x, 2),
             in_shapes=["(2*v,)"])(four_ones)
```

will give the error 'Only integers, .* tensors are valid indices, got 0' (TO FIX).

Finally, certain codes that use shapes in the actual computation may not yet work
if those shapes are polymorphic. In the code below, the expression `x.shape[0]`
will have the value of the shape variable `v`. This case is not yet implemented:

```
jax2tf.convert(lambda x: jnp.sum(x, axis=0) / x.shape[0],
               in_shapes=["(v, _)"])(np.ones((4, 4)))
```

### Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](../../../../../#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).

As of today, the tests are run using `tf_nightly==2.4.0.dev20200916`.
