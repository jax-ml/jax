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
