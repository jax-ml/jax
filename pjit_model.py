from functools import partial

import numpy.random as npr

import jax.numpy as np
from jax import lax
from jax import grad, pjit, papply


### set up some synthetic data

rng = npr.RandomState(0)
R = lambda *shape: rng.randn(*shape).astype("float32")
layer_sizes = [3, 2]
params = [(R(m, n), R(n)) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

input_batch = R(5, 3)
target_batch = R(5, 2)
batch = (input_batch, target_batch)


### standard definition

def predict(params, inputs):
  for W, b in params:
    outputs = np.dot(inputs, W) + b
    inputs = np.tanh(outputs)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  perex_loss = -np.mean(preds * targets, axis=1)
  return np.sum(perex_loss)

print 'single-machine'
print loss(params, batch)
print grad(loss)(params, batch)
print


### writing an spmd program manually

def spmd_loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  perex_loss = -np.mean(preds * targets)
  return lax.psum(perex_loss, axis_name='i')  # 'allreduce-sum' on hidden axis

# compiling the grad function for parallel execution
gradfun = pjit(grad(spmd_loss), axis_name='i', in_axes=(None, 0), out_axes=None)
print 'manual spmd program, compile-of-grad version'
print gradfun(params, batch)  # parallel execution, fwd and bwd fused together
print

# or, grad-of-compile version
spmd_loss = pjit(spmd_loss, axis_name='i', in_axes=(None, 0), out_axes=None)
print 'manual spmd program, grad-of-compile version'
print spmd_loss(params, batch)        # parallel execution
print grad(spmd_loss)(params, batch)  # parallel execution, fwd and bwd separate
print

# or get both with compile-of-grad-of-compile
gradfun = pjit(grad(spmd_loss), axis_name='i', in_axes=(None, 0), out_axes=None)
print 'manual spmd program, compile-of-grad-of-compile version'
print spmd_loss(params, batch)        # parallel execution
print grad(spmd_loss)(params, batch)  # parallel execution, fwd and bwd fused
print


### getting an spmd program from the standard definition with papply

# TODO papply!
# spmd_loss, axis_name = papply(loss, axis_size=5, in_axes=(None, 0))
# spmd_loss = pjit(spmd_loss, axis_name=axis_name, in_axes=(None, 0), out_axes=None)

# print spmd_loss(params, batch)        # parallel execution
# print
