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
f_tf_graph = tf.function(f_tf)
```

### Usage: saved model

Since jax2tf provides a regular TensorFlow function using it with SavedModel
is trivial.

```python
# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
my_model.f = tf.function(f_tf, input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory')

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load('/some/directory')
```

More involved examples of using SavedModel are described in the 
[getting started Colab](JAX2TF_getting_started.ipynb).

## Differentiaion

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
   
5. In the current version of the converter, every distinct input signature for the 
   converted function will trigger its own distinct conversion. This ensures accurate
   conversion, even if the original JAX code takes different control-flow paths 
   for different input signatures. When using SavedModel, one would have to 
   `warm up` the saved function on all the necessary input signatures. We are working
   on a mechanism to allow a single jax2tf conversion to produce a shape-polymorphic
   TF graph.  

6.  The converted code may have slightly different performance characteristics than
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


### Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](../../../../../#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).

As of today, the tests are run using `tf_nightly==2.4.0.dev20200916`.
