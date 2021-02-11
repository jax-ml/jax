---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "oDP4nK_Zgyg-", "colab_type": "text"}

# MAML Tutorial with JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google/jax/blob/master/docs/notebooks/maml.ipynb)

Eric Jang

Blog post: https://blog.evjang.com/2019/02/maml-jax.html


21 Feb 2019

Pedagogical tutorial for implementing Model-Agnostic Meta-Learning with JAX's awesome `grad` and `vmap` and `jit` operators.

## Overview

In this notebook we'll go through:

- how to take gradients, gradients of gradients.
- how to fit a sinusoid function with a neural network (and do auto-batching with vmap)
- how to implement MAML and check its numerics
- how to implement MAML for sinusoid task (single-task objective, batching task instances).
- extending MAML to handle batching at the task-level

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: zKVdo3FtgyhE

### import jax.numpy (almost-drop-in for numpy) and gradient operators.
import jax.numpy as jnp
from jax import grad
```

+++ {"id": "gMgclHhxgyhI", "colab_type": "text"}

## Gradients of Gradients

JAX makes it easy to compute gradients of python functions. Here, we thrice-differentiate $e^x$ and $x^2$

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 123
colab_type: code
id: Mt-uRwBGgyhJ
outputId: db7f718c-c2fb-4f7e-f31c-39a0d36c7051
---
f = lambda x : jnp.exp(x)
g = lambda x : jnp.square(x)
print(grad(f)(1.)) # = e^{1}
print(grad(grad(f))(1.))
print(grad(grad(grad(f)))(1.))

print(grad(g)(2.)) # 2x = 4
print(grad(grad(g))(2.)) # x = 2
print(grad(grad(grad(g)))(2.)) # x = 0
```

+++ {"id": "7mAd3We_gyhP", "colab_type": "text"}

## Sinusoid Regression and vmap

To get you familiar with JAX syntax first, we'll optimize neural network params with fixed inputs on a mean-squared error loss to $f_\theta(x) = sin(x)$.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: JN9KA1PvgyhQ

from jax import vmap # for auto-vectorizing functions
from functools import partial # for use with vmap
from jax import jit # for compiling functions for speedup
from jax import random # stax initialization uses jax.random
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax # neural network layers
import matplotlib.pyplot as plt # visualization
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: DeEALFIHgyhU

# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Dense(40), Relu,
    Dense(40), Relu,
    Dense(1)
)

rng = random.PRNGKey(0)
in_shape = (-1, 1,)
out_shape, net_params = net_init(rng, in_shape)
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: izIi-P1agyhY

def loss(params, inputs, targets):
    # Computes average loss for the batch
    predictions = net_apply(params, inputs)
    return jnp.mean((targets - predictions)**2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 287
colab_type: code
id: sROmpDEmgyhb
outputId: d1bf00d7-99e7-445e-b439-ea2fabd7a646
---
# batch the inference across K=100
xrange_inputs = jnp.linspace(-5,5,100).reshape((100, 1)) # (k, 1)
targets = jnp.sin(xrange_inputs)
predictions = vmap(partial(net_apply, net_params))(xrange_inputs)
losses = vmap(partial(loss, net_params))(xrange_inputs, targets) # per-input loss
plt.plot(xrange_inputs, predictions, label='prediction')
plt.plot(xrange_inputs, losses, label='loss')
plt.plot(xrange_inputs, targets, label='target')
plt.legend()
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: PxAEhrPGgyhh

import numpy as np
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays 
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: iZtAZfEZgyhk

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)
opt_state = opt_init(net_params)

# Define a compiled update step
@jit
def step(i, opt_state, x1, y1):
    p = get_params(opt_state)
    g = grad(loss)(p, x1, y1)
    return opt_update(i, g, opt_state)

for i in range(100):
    opt_state = step(i, opt_state, xrange_inputs, targets)
net_params = get_params(opt_state)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 287
colab_type: code
id: Rm9WIz2egyho
outputId: 183de82d-fdf0-4b81-9b14-01a85e6b8839
---
# batch the inference across K=100
targets = jnp.sin(xrange_inputs)
predictions = vmap(partial(net_apply, net_params))(xrange_inputs)
losses = vmap(partial(loss, net_params))(xrange_inputs, targets) # per-input loss
plt.plot(xrange_inputs, predictions, label='prediction')
plt.plot(xrange_inputs, losses, label='loss')
plt.plot(xrange_inputs, targets, label='target')
plt.legend()
```

+++ {"id": "7E8gAJBzgyhs", "colab_type": "text"}

## MAML: Optimizing for Generalization

Suppose task loss function $\mathcal{L}$ is defined with respect to model parameters $\theta$, input features $X$, input labels $Y$. MAML optimizes the following:

$\mathcal{L}(\theta - \nabla \mathcal{L}(\theta, x_1, y_1), x_2, y_2)$

$x_1, y_2$ and $x_2, y_2$ are identically distributed from $X, Y$. Therefore, MAML objective can be thought of as a differentiable cross-validation error (w.r.t. $x_2, y_2$) for a model that learns (via a single gradient descent step) from $x_1, y_1$. Minimizing cross-validation error provides an inductive bias on generalization.

The following toy example checks MAML numerics via parameter $x$ and input $y$.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 88
colab_type: code
id: 2YBFsM2dgyht
outputId: 46160194-04b7-46c9-897d-ecb11e9738be
---
# gradients of gradients test for MAML
# check numerics
g = lambda x, y : jnp.square(x) + y
x0 = 2.
y0 = 1.
print('grad(g)(x0) = {}'.format(grad(g)(x0, y0))) # 2x = 4
print('x0 - grad(g)(x0) = {}'.format(x0 - grad(g)(x0, y0))) # x - 2x = -2
def maml_objective(x, y):
    return g(x - grad(g)(x, y), y)
print('maml_objective(x,y)={}'.format(maml_objective(x0, y0))) # x**2 + 1 = 5
print('x0 - maml_objective(x,y) = {}'.format(x0 - grad(maml_objective)(x0, y0))) # x - (2x)
```

+++ {"id": "V9G-PMxygyhx", "colab_type": "text"}

## Sinusoid Task + MAML


Now let's re-implement the Sinusoidal regression task from Chelsea Finn's [MAML paper](https://arxiv.org/abs/1703.03400).

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: s1v5VABkgyhy

alpha = .1
def inner_update(p, x1, y1):
    grads = grad(loss)(p, x1, y1)
    inner_sgd_fn = lambda g, state: (state - alpha*g)
    return tree_multimap(inner_sgd_fn, grads, p)

def maml_loss(p, x1, y1, x2, y2):
    p2 = inner_update(p, x1, y1)
    return loss(p2, x2, y2)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
id: bQvg749Xgyh2
outputId: 5043f859-c537-41b8-c390-23670795d57b
---
x1 = xrange_inputs
y1 = targets
x2 = jnp.array([0.])
y2 = jnp.array([0.])
maml_loss(net_params, x1, y1, x2, y2)
```

+++ {"id": "zMB6BwPogyh6", "colab_type": "text"}

Let's try minimizing the MAML loss (without batching across multiple tasks, which we will do in the next section)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 371
colab_type: code
id: pB5ldBO-gyh7
outputId: b2365aa4-d7b8-40a0-d759-8257d3e4d768
---
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)  # this LR seems to be better than 1e-2 and 1e-4
out_shape, net_params = net_init(rng, in_shape)
opt_state = opt_init(net_params)

@jit
def step(i, opt_state, x1, y1, x2, y2):
    p = get_params(opt_state)
    g = grad(maml_loss)(p, x1, y1, x2, y2)
    l = maml_loss(p, x1, y1, x2, y2)
    return opt_update(i, g, opt_state), l
K=20

np_maml_loss = []

# Adam optimization
for i in range(20000):
    # define the task
    A = np.random.uniform(low=0.1, high=.5)
    phase = np.random.uniform(low=0., high=jnp.pi)
    # meta-training inner split (K examples)
    x1 = np.random.uniform(low=-5., high=5., size=(K,1))
    y1 = A * np.sin(x1 + phase)
    # meta-training outer split (1 example). Like cross-validating with respect to one example.
    x2 = np.random.uniform(low=-5., high=5.)
    y2 = A * np.sin(x2 + phase)
    opt_state, l = step(i, opt_state, x1, y1, x2, y2)
    np_maml_loss.append(l)
    if i % 1000 == 0:
        print(i)
net_params = get_params(opt_state)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 287
colab_type: code
id: ogcpFdJ9gyh_
outputId: 856924a3-ede5-44ba-ba3c-381673713fad
---
# batch the inference across K=100
targets = jnp.sin(xrange_inputs)
predictions = vmap(partial(net_apply, net_params))(xrange_inputs)
plt.plot(xrange_inputs, predictions, label='pre-update predictions')
plt.plot(xrange_inputs, targets, label='target')

x1 = np.random.uniform(low=-5., high=5., size=(K,1))
y1 = 1. * np.sin(x1 + 0.)

for i in range(1,5):
    net_params = inner_update(net_params, x1, y1)
    predictions = vmap(partial(net_apply, net_params))(xrange_inputs)
    plt.plot(xrange_inputs, predictions, label='{}-shot predictions'.format(i))
plt.legend()
```

+++ {"id": "7TMYcZKVgyiD", "colab_type": "text"}

## Batching Meta-Gradient Across Tasks

Kind of does the job but not that great. Let's reduce the variance of gradients in outer loop by averaging across a batch of tasks (not just one task at a time). 

vmap is awesome it enables nice handling of batching at two levels: inner-level "intra-task" batching, and outer level batching across tasks.

From a software engineering perspective, it is nice because the "task-batched" MAML implementation simply re-uses code from the non-task batched MAML algorithm, without losing any vectorization benefits.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: 9Pj04Z7MgyiF

def sample_tasks(outer_batch_size, inner_batch_size):
    # Select amplitude and phase for the task
    As = []
    phases = []
    for _ in range(outer_batch_size):        
        As.append(np.random.uniform(low=0.1, high=.5))
        phases.append(np.random.uniform(low=0., high=jnp.pi))
    def get_batch():
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = np.random.uniform(low=-5., high=5., size=(inner_batch_size, 1))
            y = A * np.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return jnp.stack(xs), jnp.stack(ys)
    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 287
colab_type: code
id: 7dCIGObKgyiJ
outputId: c169b529-0f16-4f20-d20e-d802765e4068
---
outer_batch_size = 2
x1, y1, x2, y2 = sample_tasks(outer_batch_size, 50)
for i in range(outer_batch_size):
    plt.scatter(x1[i], y1[i], label='task{}-train'.format(i))
for i in range(outer_batch_size):
    plt.scatter(x2[i], y2[i], label='task{}-val'.format(i))
plt.legend()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 35
colab_type: code
id: BrSX--wpgyiP
outputId: 6d81e7ff-7cd9-4aef-c665-952d442369d5
---
x2.shape
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 371
colab_type: code
id: P3WQ8_k2gyiU
outputId: fed1b78b-7910-4e44-a80b-18f447379022
---
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
out_shape, net_params = net_init(rng, in_shape)
opt_state = opt_init(net_params)

# vmapped version of maml loss.
# returns scalar for all tasks.
def batch_maml_loss(p, x1_b, y1_b, x2_b, y2_b):
    task_losses = vmap(partial(maml_loss, p))(x1_b, y1_b, x2_b, y2_b)
    return jnp.mean(task_losses)

@jit
def step(i, opt_state, x1, y1, x2, y2):
    p = get_params(opt_state)
    g = grad(batch_maml_loss)(p, x1, y1, x2, y2)
    l = batch_maml_loss(p, x1, y1, x2, y2)
    return opt_update(i, g, opt_state), l

np_batched_maml_loss = []
K=20
for i in range(20000):
    x1_b, y1_b, x2_b, y2_b = sample_tasks(4, K)
    opt_state, l = step(i, opt_state, x1_b, y1_b, x2_b, y2_b)
    np_batched_maml_loss.append(l)
    if i % 1000 == 0:
        print(i)
net_params = get_params(opt_state)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 287
colab_type: code
id: PmxHLrhYgyiX
outputId: 33ac699e-c66d-46e2-affa-98ae948d52e8
---
# batch the inference across K=100
targets = jnp.sin(xrange_inputs)
predictions = vmap(partial(net_apply, net_params))(xrange_inputs)
plt.plot(xrange_inputs, predictions, label='pre-update predictions')
plt.plot(xrange_inputs, targets, label='target')

x1 = np.random.uniform(low=-5., high=5., size=(10,1))
y1 = 1. * np.sin(x1 + 0.)

for i in range(1,3):
    net_params = inner_update(net_params, x1, y1)
    predictions = vmap(partial(net_apply, net_params))(xrange_inputs)
    plt.plot(xrange_inputs, predictions, label='{}-shot predictions'.format(i))
plt.legend()
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 287
colab_type: code
id: cQf2BeDjgyib
outputId: fc52caf6-1379-4d60-fe44-99f4e4518698
---
# Comparison of maml_loss for task batch size = 1 vs. task batch size = 8
plt.plot(np.convolve(np_maml_loss, [.05]*20), label='task_batch=1')
plt.plot(np.convolve(np_batched_maml_loss, [.05]*20), label='task_batch=4')
plt.ylim(0., 1e-1)
plt.legend()
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: vCHCvXh-mm1v


```
