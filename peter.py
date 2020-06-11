from jax import jit, grad, make_jaxpr
import jax.numpy as jnp

# @jit
# def f(x):
#   newshape = (jnp.prod(x.shape),)
#   return x.reshape(newshape)


# f(jnp.zeros((2, 2)))

import jax.numpy as jnp
from jax import jit, grad, vmap

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return jnp.sum((preds - targets) ** 2)




from jax import random

def init_layer(key, n_in, n_out):
  k1, k2 = random.split(key)
  W = random.normal(k1, (n_in, n_out))
  b = random.normal(k2, (n_out,))
  return W, b

layer_sizes = [5, 2, 3]

key = random.PRNGKey(0)
key, *keys = random.split(key, len(layer_sizes))
params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

key, *keys = random.split(key, 3)
inputs = random.normal(keys[0], (8, 5))
targets = random.normal(keys[1], (8, 3))
batch = (inputs, targets)


jaxpr = make_jaxpr(grad(loss))(params, batch)
