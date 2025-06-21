---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "PxHrg4Cjuapm"}

# Distributed arrays and automatic parallelization

<!--* freshness: { reviewed: '2024-04-16' } *-->

+++ {"id": "pFtQjv4SzHRj"}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/Distributed_arrays_and_automatic_parallelization.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/Distributed_arrays_and_automatic_parallelization.ipynb)

This tutorial discusses parallelism via `jax.Array`, the unified array object model available in JAX v0.4.1 and newer.

```{code-cell}
:id: FNxScTfq3vGF

from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
```

+++ {"id": "eyHMwyEfQJcz"}

⚠️ WARNING: The notebook requires 8 devices to run.

```{code-cell}
:id: IZMLqOUV3vGG

if len(jax.local_devices()) < 8:
  raise Exception("Notebook requires 8 devices to run")
```

+++ {"id": "3f37ca93"}

## Intro and a quick example

By reading this tutorial notebook, you'll learn about `jax.Array`, a unified 
datatype for representing arrays, even with physical storage spanning multiple
devices. You'll also learn about how using `jax.Array`s together with `jax.jit`
can provide automatic compiler-based parallelization.

Before we think step by step, here's a quick example.
First, we'll create a `jax.Array` sharded across multiple devices:

```{code-cell}
:id: Gf2lO4ii3vGG

from jax.sharding import PartitionSpec as P, NamedSharding
```

```{code-cell}
:id: q-XBTEoy3vGG

# Create a Sharding object to distribute a value across devices:
mesh = jax.make_mesh((4, 2), ('x', 'y'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 166
id: vI39znW93vGH
outputId: 4f702753-8add-4b65-a4af-0f18f098cc46
---
# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (8192, 8192))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "jZ0ZY9Um9Jg4"}

Next, we'll apply a computation to it and visualize how the result values are
stored across multiple devices too:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 166
id: -qCnHZl83vGI
outputId: 0e131c23-5765-43ae-f232-6417ae1acbb2
---
z = jnp.sin(y)
jax.debug.visualize_array_sharding(z)
```

+++ {"id": "5qccVQoE9tEi"}

The evaluation of the `jnp.sin` application was automatically parallelized
across the devices on which the input values (and output values) are stored:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: _VTzN0r03vGI
outputId: c03eecab-4c86-4dac-d776-5fc72cbb5273
---
# `x` is present on a single device
%timeit -n 5 -r 5 jnp.sin(x).block_until_ready()
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: QuzhU1g63vGI
outputId: 8135cca0-871b-4b6a-a7e5-02e78c2028c7
---
# `y` is sharded across 8 devices.
%timeit -n 5 -r 5 jnp.sin(y).block_until_ready()
```

+++ {"id": "xWknFQbQ-bzV"}

Now let's look at each of these pieces in more detail!


## `Sharding` describes how array values are laid out in memory across devices

+++ {"id": "W6HsXauGxL6w"}

### Sharding basics, and the `NamedSharding` subclass

+++ {"id": "NWDyp_EjVHkg"}

To parallelize computation across multiple devices, we first must lay out input data across multiple devices.

In JAX, `Sharding` objects describe distributed memory layouts. They can be used with `jax.device_put` to produce a value with distributed layout.

For example, here's a value with a single-device `Sharding`:

```{code-cell}
:id: VmoX4SUp3vGJ

import jax
x = jax.random.normal(jax.random.key(0), (8192, 8192))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: vNRabO2J3vGJ
outputId: 40fd7172-a16c-4dd8-e2e1-17bb3afe5409
---
jax.debug.visualize_array_sharding(x)
```

+++ {"id": "HhCjhK0zXIqX"}

Here, we're using the `jax.debug.visualize_array_sharding` function to show where the value `x` is stored in memory. All of `x` is stored on a single device, so the visualization is pretty boring!

But we can shard `x` across multiple devices by using `jax.device_put` and a `Sharding` object. First, we make a `numpy.ndarray` of `Devices` using `jax.make_mesh`, which takes hardware topology into account for the `Device` order:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 166
id: zpB1JxyK3vGN
outputId: 8e385462-1c2c-4256-c38a-84299d3bd02c
---
from jax.sharding import Mesh, PartitionSpec, NamedSharding

P = PartitionSpec

mesh = jax.make_mesh((4, 2), ('a', 'b'))
y = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "OW_Cc92G1-nr"}

We can define a helper function to make things simpler:

```{code-cell}
:id: 8g0Md2Gd3vGO

default_mesh = jax.make_mesh((4, 2), ('a', 'b'))

def mesh_sharding(
    pspec: PartitionSpec, mesh: Optional[Mesh] = None,
  ) -> NamedSharding:
  if mesh is None:
    mesh = default_mesh
  return NamedSharding(mesh, pspec)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 166
id: zp3MfS4Y3vGO
outputId: 032fdd7e-19a1-45da-e1ad-b3227fa43ee6
---
y = jax.device_put(x, mesh_sharding(P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "xZ88riVm1mv5"}

Here, we use `P('a', 'b')` to express that the first and second axes of `x` should be sharded over the device mesh axes `'a'` and `'b'`, respectively. We can easily switch to `P('b', 'a')` to shard the axes of `x` over different devices:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: FigK5Zsa3vGO
outputId: e488d073-9d02-4376-a6af-19d6d5509c7d
---
y = jax.device_put(x, mesh_sharding(P('b', 'a')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 166
id: hI-HD0xN3vGO
outputId: b0c2e863-3aee-4417-b45f-21b2187f6ef7
---
# This `None` means that `x` is not sharded on its second dimension,
# and since the Mesh axis name 'b' is not mentioned, shards are
# replicated across it.
y = jax.device_put(x, mesh_sharding(P('a', None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "AqcAsNUgXCZz"}

Here, because `P('a', None)` doesn't mention the `Mesh` axis name `'b'`, we get replication over the axis `'b'`. The `None` here is just acting as a placeholder to line up against the second axis of the value `x`, without expressing sharding over any mesh axis. (As a shorthand, trailing `None`s can be omitted, so that `P('a', None)` means the same thing as `P('a')`. But it doesn't hurt to be explicit!)

To shard only over the second axis of `x`, we can use a `None` placeholder in the `PartitionSpec`:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: EXBExMQC3vGP
outputId: c80e6177-12a6-40ef-b4e4-934dad22da3d
---
y = jax.device_put(x, mesh_sharding(P(None, 'b')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: PjUpG8uz3vGP
outputId: a0f59dc5-b509-4b8b-bd22-bcd69f696763
---
y = jax.device_put(x, mesh_sharding(P(None, 'a')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "--AZgW1P3HFT"}

For a fixed mesh, we can even partition one logical axis of `x` over multiple device mesh axes:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 298
id: fVcPbDUA3vGP
outputId: da3f435d-dfc1-4a41-ec90-691cd7c748a0
---
y = jax.device_put(x, mesh_sharding(P(('a', 'b'), None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "c1tTFudr3Ae7"}

Using `NamedSharding` makes it easy to define a device mesh once and give its axes names, then just refer to those names in `PartitionSpec`s for each `device_put` as needed.

+++ {"id": "rhWzHgGf4mkg"}

## Computation follows data sharding and is automatically parallelized

+++ {"id": "JukoaRhl4tXJ"}

With sharded input data, the compiler can give us parallel computation. In particular, functions decorated with `jax.jit` can operate over sharded arrays without copying data onto a single device. Instead, computation follows sharding: based on the sharding of the input data, the compiler decides shardings for intermediates and output values, and parallelizes their evaluation, even inserting communication operations as necessary.

For example, the simplest computation is an elementwise one:

```{code-cell}
:id: _EmQwggc3vGQ

mesh = jax.make_mesh((4, 2), ('a', 'b'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 349
id: LnT0vWjc3vGQ
outputId: 8e642049-61eb-458d-af79-ac449b58d11b
---
x = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
print('input sharding:')
jax.debug.visualize_array_sharding(x)

y = jnp.sin(x)
print('output sharding:')
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "7tY2gVRfazaT"}

Here for the elementwise operation `jnp.sin` the compiler chose the output sharding to be the same as the input. Moreover, the compiler automatically parallelized the computation, so that each device computed its output shard from its input shard in parallel.

In other words, even though we wrote the `jnp.sin` computation as if a single machine were to execute it, the compiler splits up the computation for us and executes it on multiple devices.

We can do the same for more than just elementwise operations too. Consider a matrix multiplication with sharded inputs:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 548
id: Dq043GkP3vGQ
outputId: 3eff7b67-d7f0-4212-c9d3-2cc271ac1f98
---
y = jax.device_put(x, NamedSharding(mesh, P('a', None)))
z = jax.device_put(x, NamedSharding(mesh, P(None, 'b')))
print('lhs sharding:')
jax.debug.visualize_array_sharding(y)
print('rhs sharding:')
jax.debug.visualize_array_sharding(z)

w = jnp.dot(y, z)
print('out sharding:')
jax.debug.visualize_array_sharding(w)
```

+++ {"id": "_EPNaWzgazft"}

Here the compiler chose the output sharding so that it could maximally parallelize the computation: without needing communication, each device already has the input shards it needs to compute its output shard.

How can we be sure it's actually running in parallel? We can do a simple timing experiment:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: QjQ5u8qh3vGQ
outputId: 0aefc170-833c-4a6a-e003-5990d3db31d9
---
x_single = jax.device_put(x, jax.devices()[0])
jax.debug.visualize_array_sharding(x_single)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 8tn8lOj73vGR
outputId: d9898c93-7afc-416b-8c40-4d9551613cd0
---
np.allclose(jnp.dot(x_single, x_single),
            jnp.dot(y, z))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: D7PpZwhR3vGR
outputId: 4901a11b-2354-4d26-a897-b88def07a716
---
%timeit -n 5 -r 5 jnp.dot(x_single, x_single).block_until_ready()
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: rgo_yVHF3vGR
outputId: e51216cf-b073-4250-d422-67f9fd72f6aa
---
%timeit -n 5 -r 5 jnp.dot(y, z).block_until_ready()
```

+++ {"id": "gglQIMXJnnJw"}

Even copying a sharded `Array` produces a result with the sharding of the input:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 166
id: f1Zw-2lH3vGR
outputId: 43d7a642-fde4-47a6-901f-dfdc64d6a613
---
w_copy = jnp.copy(w)
jax.debug.visualize_array_sharding(w_copy)
```

+++ {"id": "3qfPjJdhgerc"}

So computation follows data placement: when we explicitly shard data with `jax.device_put`, and apply functions to that data, the compiler attempts to parallelize the computation and decide the output sharding. This policy for sharded data is a generalization of [JAX's policy of following explicit device placement](https://docs.jax.dev/en/latest/faq.html#controlling-data-and-computation-placement-on-devices).

+++ {"id": "QRB95LaWuT80"}

### When explicit shardings disagree, JAX errors

But what if two arguments to a computation are explicitly placed on different sets of devices, or with incompatible device orders?
In these ambiguous cases, an error is raised:

```{code-cell}
:id: 1vAkZAOY3vGR

import textwrap
from termcolor import colored

def print_exception(e):
  name = colored(f'{type(e).__name__}', 'red', force_color=True)
  print(textwrap.fill(f'{name}: {str(e)}'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: DHh0N3vn3vGS
outputId: 8c4652f7-c484-423b-ad78-182134280187
---
sharding1 = NamedSharding(Mesh(jax.devices()[:4], 'x'), P('x'))
sharding2 = NamedSharding(Mesh(jax.devices()[4:], 'x'), P('x'))

y = jax.device_put(x, sharding1)
z = jax.device_put(x, sharding2)
try: y + z
except ValueError as e: print_exception(e)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Im7DkoOl3vGS
outputId: 1b6fcd7a-762b-4366-a96d-aea63bad7fe0
---
devices = jax.devices()
permuted_devices = [devices[i] for i in [0, 1, 2, 3, 6, 7, 4, 5]]

sharding1 = NamedSharding(Mesh(devices, 'x'), P('x'))
sharding2 = NamedSharding(Mesh(permuted_devices, 'x'), P('x'))

y = jax.device_put(x, sharding1)
z = jax.device_put(x, sharding2)
try: y + z
except ValueError as e: print_exception(e)
```

+++ {"id": "6ZYcK8eXrn0p"}

We say arrays that have been explicitly placed or sharded with `jax.device_put` are _committed_ to their device(s), and so won't be automatically moved. See the [device placement FAQ](https://docs.jax.dev/en/latest/faq.html#controlling-data-and-computation-placement-on-devices) for more information.

When arrays are _not_ explicitly placed or sharded with `jax.device_put`, they are placed _uncommitted_ on the default device.
Unlike committed arrays, uncommitted arrays can be moved and resharded automatically: that is, uncommitted arrays can be arguments to a computation even if other arguments are explicitly placed on different devices.

For example, the output of `jnp.zeros`, `jnp.arange`, and `jnp.array` are uncommitted:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: _QvtKL8r3vGS
outputId: 761b1208-fe4b-4c09-a7d2-f62152183ef0
---
y = jax.device_put(x, sharding1)
y + jnp.ones_like(y)
y + jnp.arange(y.size).reshape(y.shape)
print('no error!')
```

+++ {"id": "dqMKl79NaIWF"}

## Constraining shardings of intermediates in `jit`ted code

+++ {"id": "g4LrDDcJwkHc"}

While the compiler will attempt to decide how a function's intermediate values and outputs should be sharded, we can also give it hints using `jax.lax.with_sharding_constraint`. Using `jax.lax.with_sharding_constraint` is much like `jax.device_put`, except we use it inside staged-out (i.e. `jit`-decorated) functions:

```{code-cell}
:id: jniSFm5V3vGT

mesh = jax.make_mesh((4, 2), ('x', 'y'))
```

```{code-cell}
:id: Q1wuDp-L3vGT

x = jax.random.normal(jax.random.key(0), (8192, 8192))
x = jax.device_put(x, NamedSharding(mesh, P('x', 'y')))
```

```{code-cell}
:id: rqEDj0wB3vGT

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('y', 'x')))
  return y
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 347
id: zYFS-n4r3vGT
outputId: 0ac96b8f-ed23-4413-aed9-edd00a841c37
---
jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: 8g_2Y8wp3vGT

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
  return y
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 347
id: AiRFtVsR3vGT
outputId: 2edacc2c-ac80-4519-c9d1-bee364a22b31
---
jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "_Y1P5wLTzJSz"}

By adding `with_sharding_constraint`, we've constrained the sharding of the output. In addition to respecting the annotation on a particular intermediate, the compiler will use annotations to decide shardings for other values.

It's often a good practice to annotate the outputs of computations, for example based on how the values are ultimately consumed.

+++ {"id": "QUkXWG-baMUs"}

## Examples: neural networks

+++ {"id": "g7y0OJBSGoSW"}

**⚠️ WARNING: The following is meant to be a simple demonstration of automatic sharding propagation with `jax.Array`, but it may not reflect best practices for real examples.** For instance, real examples may require more use of `with_sharding_constraint`.

+++ {"id": "3ii_UPkG3gzP"}

We can use `jax.device_put` and `jax.jit`'s computation-follows-sharding features to parallelize computation in neural networks. Here are some simple examples, based on this basic neural network:

```{code-cell}
:id: mEKF3zIF3vGU

import jax
import jax.numpy as jnp
```

```{code-cell}
:id: Mocs3oGe3vGU

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.maximum(outputs, 0)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))
```

```{code-cell}
:id: glBB8tzW3vGU

loss_jit = jax.jit(loss)
gradfun = jax.jit(jax.grad(loss))
```

```{code-cell}
:id: R0x62AIa3vGU

def init_layer(key, n_in, n_out):
  k1, k2 = jax.random.split(key)
  W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
  b = jax.random.normal(k2, (n_out,))
  return W, b

def init_model(key, layer_sizes, batch_size):
  key, *keys = jax.random.split(key, len(layer_sizes))
  params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

  key, *keys = jax.random.split(key, 3)
  inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
  targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

  return params, (inputs, targets)

layer_sizes = [784, 8192, 8192, 8192, 10]
batch_size = 8192

params, batch = init_model(jax.random.key(0), layer_sizes, batch_size)
```

+++ {"id": "sJv_h0AS2drh"}

### 8-way batch data parallelism

```{code-cell}
:id: mJLqRPpSDX0i

mesh = jax.make_mesh((8,), ('batch',))
```

```{code-cell}
:id: _Q5NbdOn3vGV

sharding = NamedSharding(mesh, P('batch'))
replicated_sharding = NamedSharding(mesh, P())
```

```{code-cell}
:id: 3KC6ieEe3vGV

batch = jax.device_put(batch, sharding)
params = jax.device_put(params, replicated_sharding)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: MUb-QE2b3vGV
outputId: 5a27f007-c572-44f8-9f49-6e745ee739e8
---
loss_jit(params, batch)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: HUkw0u413vGV
outputId: 07e481a1-97fb-4bd0-d754-cb6d8317bff6
---
step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: paCw6Zaj3vGV
outputId: ad4cce34-3a6a-4d44-9a86-477a7fee4841
---
%timeit -n 5 -r 5 gradfun(params, batch)[0][0].block_until_ready()
```

```{code-cell}
:id: BF86UWpg3vGV

batch_single = jax.device_put(batch, jax.devices()[0])
params_single = jax.device_put(params, jax.devices()[0])
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Z1wgUKXk3vGV
outputId: d66767b7-3f17-482f-b811-919bb1793277
---
%timeit -n 5 -r 5 gradfun(params_single, batch_single)[0][0].block_until_ready()
```

+++ {"id": "3AjeeB7B4NP6"}

### 4-way batch data parallelism and 2-way model tensor parallelism

```{code-cell}
:id: k1hxOfgRDwo0

mesh = jax.make_mesh((4, 2), ('batch', 'model'))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 314
id: sgIWCjJK3vGW
outputId: 8cb0f19f-3942-415c-c57a-31bb81784f46
---
batch = jax.device_put(batch, NamedSharding(mesh, P('batch', None)))
jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(batch[1])
```

```{code-cell}
:id: q9PQP-0eEAO6

replicated_sharding = NamedSharding(mesh, P())
```

```{code-cell}
:id: BqCjYCgg3vGW

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params

W1 = jax.device_put(W1, replicated_sharding)
b1 = jax.device_put(b1, replicated_sharding)

W2 = jax.device_put(W2, NamedSharding(mesh, P(None, 'model')))
b2 = jax.device_put(b2, NamedSharding(mesh, P('model')))

W3 = jax.device_put(W3, NamedSharding(mesh, P('model', None)))
b3 = jax.device_put(b3, replicated_sharding)

W4 = jax.device_put(W4, replicated_sharding)
b4 = jax.device_put(b4, replicated_sharding)

params = (W1, b1), (W2, b2), (W3, b3), (W4, b4)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: _lSJ63sh3vGW
outputId: bcd3e33e-36b5-4787-9cd2-60623fd6e5fa
---
jax.debug.visualize_array_sharding(W2)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 199
id: fxkfWYkk3vGW
outputId: 59e60b16-fe37-47d4-8214-96096ffbd79c
---
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: uPCVs-_k3vGW
outputId: 618516e9-9736-4ca0-dd22-09d094ce57a2
---
print(loss_jit(params, batch))
```

```{code-cell}
:id: L9JebLK_3vGW

step_size = 1e-5

for _ in range(30):
    grads = gradfun(params, batch)
    params = [(W - step_size * dW, b - step_size * db)
              for (W, b), (dW, db) in zip(params, grads)]
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: c9Sbl69e3vGX
outputId: 2ee3d432-7172-46ca-e01a-614e83345808
---
print(loss_jit(params, batch))
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 380
id: lkAF0dAb3vGX
outputId: 6c1e317e-cded-4af4-8080-0de835fa4c71
---
(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params
jax.debug.visualize_array_sharding(W2)
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: I1Npor3i3vGX
outputId: 479c4d81-cb0b-40a5-89ba-394c10dc3297
---
%timeit -n 10 -r 10 gradfun(params, batch)[0][0].block_until_ready()
```

+++ {"id": "3diqi5VRBy6S"}

## Sharp bits

+++ {"id": "OTfoXNnxFYDJ"}

### Generating random numbers

JAX comes with a functional, deterministic [random number generator](https://docs.jax.dev/en/latest/jep/263-prng.html). It underlies the various sampling functions in the [`jax.random` module](https://docs.jax.dev/en/latest/jax.random.html), such as `jax.random.uniform`.

JAX's random numbers are produced by a counter-based PRNG, so in principle, random number generation should be a pure map over counter values. A pure map is a trivially partitionable operation in principle. It should require no cross-device communication, nor any redundant computation across devices.

However, the existing stable RNG implementation is not automatically partitionable, for historical reasons.

+++ {"id": "ht_zYFVXNrjN"}

Consider the following example, where a function draws random uniform numbers and adds them to the input, elementwise:

```{code-cell}
:id: kwS-aQE_3vGX

@jax.jit
def f(key, x):
  numbers = jax.random.uniform(key, x.shape)
  return x + numbers

key = jax.random.key(42)
mesh = Mesh(jax.devices(), 'x')
x_sharding = NamedSharding(mesh, P('x'))
x = jax.device_put(jnp.arange(24), x_sharding)
```

+++ {"id": "ZgSA9x9NLMaP"}

On a partitioned input, the function `f` produces output that is also partitioned:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 67
id: Oi97rpLz3vGY
outputId: 9dd63254-a483-4847-c0f5-5a4367bf08e9
---
jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "WnjlWDUYLkp6"}

But if we inspect the compiled computation for `f` on this partitioned input, we see that it does involve some communication:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 64wIZuSJ3vGY
outputId: fa166d45-ca9c-457a-be84-bcc9236d0730
---
f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "AXp9i8fbL8DD"}

One way to work around this is to configure JAX with the experimental upgrade flag `jax_threefry_partitionable`. With the flag on, the "collective permute" operation is now gone from the compiled computation:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 1I7bqxA63vGY
outputId: 756e0a36-ff14-438f-bbd4-3ef03f97a47b
---
jax.config.update('jax_threefry_partitionable', True)
f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "WV8ZccM5SXOU"}

The output is still partitioned:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 67
id: zHPJzdn23vGY
outputId: 3332de0f-4827-4f0b-b9ef-69249b7c6bc6
---
jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "kaK--hPmSPpV"}

One caveat to the `jax_threefry_partitionable` option, however, is that _the random values produced may be different than without the flag set_, even though they were generated by the same random key:

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: nBUHBBal3vGY
outputId: 4b9be948-ccab-4a31-a06f-37ec9c7b5235
---
jax.config.update('jax_threefry_partitionable', False)
print('Stable:')
print(f(key, x))
print()

jax.config.update('jax_threefry_partitionable', True)
print('Partitionable:')
print(f(key, x))
```

+++ {"id": "8BDPqgOrTMfK"}

In `jax_threefry_partitionable` mode, the JAX PRNG remains deterministic, but its implementation is new (and under development). The random values generated for a given key will be the same at a given JAX version (or a given commit on the `main` branch), but may vary across releases.
