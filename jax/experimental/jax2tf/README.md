# JAX to TensorFlow converter

WARNING: This is beta quality and not ready for production use. Please expect
API incompleteness and changes!

This package provides an experimental JAX interpreter that implements most JAX
primitives using TensorFlow operations. In practice this means that you can take
some code written in JAX and execute it using TensorFlow eager mode, or stage it 
out as a TensorFlow graph.

Most commonly people want to use this tool in order to:

1.  Serialize JAX programs (and state) to disk as `tf.Graph`s.
2.  Use JAX programs as part of an otherwise TensorFlow code base.

## Caveats

1.  The graphs produced require you to link XLA in order to run.
2.  Some of the operations used by this converter are not covered by TensorFlow API
    guarantee. Please DO NOT YET COUNT ON LONG-TERM COMPATIBILITY of the SavedModel
    produced with this converter, until we vet all the ops used for API guarantee.
3.  You can take first-order gradients of most of the converted functions, and those will
    be JAX-accurate because they use the JAX logic and custom gradients. But such
    custom gradients are currently lost when saving a SavedModel.
    Please DO NOT TRUST GRADIENTS of converted functions once they have been
    saved in SavedModel.
4.  Taking gradients of converted functions that use certain primitives will fail.
5.  Not all jax primitives are supported so lowering may fail in some cases.
6.  There is currently no support for replicated (e.g. `pmap`) or multi-device
    (e.g. `sharded_jit`) functions.
7.  We have not yet tested enough corner-cases of the conversion, it is possible
    the some inaccuracy between the values computed by the JAX function and the
    converted function.


### Converting basic functions.

As a rule of thumb, if you can `jax.jit` your function then you should be able
to use `jax2tf.convert`:

```python
import jax
from jax.experimental import jax2tf
def some_jax_function(x, y, z):
  return jax.something(x, y, z)

# tf_ops.from_jax is a higher order function that returns a wrapped function with
# the same signature as your input function but accepting TensorFlow tensors (or
# variables) as input.
tf_version = jax2tf.convert(some_jax_function)

# For example you can call tf_version with some TensorFlow tensors:
tf_version(tf_x, tf_y, tf_z)

# Additionally you can use tools like `tf.function` to improve the execution
# time of your function, or to stage it out to a saved model:
tf_version = tf.function(tf_version)
```

### Saved model

WARNING: Please see the caveats section above for more information on potential
issues with saved models.

Since jax2tf provides a regular TensorFlow function using it with saved model
is trivial.

```python
f_jax = jax.jit(lambda x: jnp.sin(jnp.cos(x)))
f_tf = jax2tf.convert(f_jax)

# You can save the model just like you would with any other TensorFlow function:
my_model = tf.Module()
my_model.f = tf.function(f_tf, input_signature=[tf.TensorSpec([], tf.float32)])
tf.saved_model.save(my_model, '/some/directory')

# Restoring (note: the restored model does *not* require JAX to run, just XLA).
restored_model = tf.saved_model.load('/some/directory')
```

To explain (TODO):

  * saved model and custom gradients
  * saved model and XLA

### Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](../../../../../#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).

## Known limitations

#### Errors due to nested invocations of XLA

In some rare cases, running the converted function will result in compilation
errors of the form
`InvalidArgumentError: Function invoked by the following node is not compilable`.
This can be fixed by ensuring that the conversion happens in a compilation context, e.g.:

```
tf.function(jax2tf.convert(func), experimental_compile=True)(*args)
```

Explanation: some of the TF operations that are used in the conversion have the
correct (i.e., matching JAX) semantics only if they are compiled with XLA. An example
is `tf.gather` that for out-of-bounds index has different semantics when XLA
is used vs. when XLA is not used. Only the XLA semantics matches the JAX semantics. 
To work around this problem, the conversion will wrap the use of these TF ops with a
`tf.xla.experimental.compile(tf.gather)`. However, there seems to be a problem where the 
`tf.xla.experimental.compile` cannot be used if we are already in a compilation
context (results in the error mentioned above). (See b/162814494). 
The converter watches for an
enclosing compilation context and will not use the compiler if we are already
in a compilation context. That is why the solution mentioned above works.

One instance when you can still get this error is on TPU, when you use the
converted function in graph mode but without invoking the compiler:
`tf.function(jax2tf.convert(fun), experimental_compile=False)`. Since there
is no compilation context, the converter will apply the compiler internally, 
but since we are on TPU, when executing the graph the compiler will be called 
again, resulting in the error.

#### Converted functions cannot be differentiated when loaded from SavedModel

TODO


