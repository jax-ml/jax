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

### Running on GPU

To run jax2tf on GPU, both jaxlib and TensorFlow must be installed with support
for CUDA. One must be mindful to install a version of CUDA that is compatible
with both [jaxlib](../../../../../#pip-installation) and
[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations).
