from functools import partial
import operator as op

import numpy as onp

import jax.numpy as np
from jax import pmap, grad
from jax import lax
from jax.tree_util import tree_map
from jax.lib.xla_bridge import device_count


step_size = 0.01
rng = onp.random.RandomState(0)

def predict(params, inputs):
  for W, b in params:
    outputs = np.dot(inputs, W) + b
    inputs = np.tanh(outputs)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return np.sum((predictions - targets)**2)

def update(params, batch):
  grads = grad(loss)(params, batch)
  new_params = [(W - step_size * dW, b - step_size * db)
                for (W, b), (dW, db) in zip(params, grads)]
  return new_params

# initialize parameters
layer_sizes = [2, 4, 3]  # input size 2, output size 3
scale = 0.01
params = [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# set up fake data
inputs = rng.randn(10, 2)  # batch size 10, feature size 2
targets = rng.randn(10, 3)  # batch size 10, output size 3
batch = (inputs, targets)


# standard functions
print(loss(params, batch))
print(update(params, batch)[0][0])


# reshape / replicate data
num_devices = device_count()
spmd_params = pmap(lambda _: params)(onp.arange(num_devices))
spmd_inputs = inputs.reshape((num_devices, -1, 2))
spmd_targets = targets.reshape((num_devices, -1, 3))
spmd_batch = (spmd_inputs, spmd_targets)

@partial(pmap, axis_name='i')
def spmd_loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  batch_loss = np.sum((predictions - targets)**2)
  return lax.psum(batch_loss, 'i')
print(spmd_loss(spmd_params, spmd_batch))

@partial(pmap, axis_name='i')
def spmd_update(params, batch):
  grads = grad(loss)(params, batch)  # loss, not spmd_loss
  grads = tree_map(lambda x: lax.psum(x, 'i'), grads)
  new_params = [(W - step_size * dW, b - step_size * db)
                for (W, b), (dW, db) in zip(params, grads)]
  return new_params
print(spmd_update(spmd_params, spmd_batch)[0][0])
