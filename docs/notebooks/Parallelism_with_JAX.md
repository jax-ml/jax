---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "PxHrg4Cjuapm"}

# Parallelism with JAX

+++ {"id": "pFtQjv4SzHRj"}

**This tutorial discusses parallelism via `jax.Array`, the unified array object model available in JAX v0.4.0 and newer.**

See {ref}`jax-array-migration` guide for migrating existing pre-v0.4.0 codebases to `jax.Array`.

**The features required by `jax.Array` are not supported by the Colab TPU runtime at this time.**

```{code-cell}
:id: 41dde63b

import os

import functools
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update('jax_array', True)
```

+++ {"id": "eyHMwyEfQJcz"}

⚠️ WARNING: The notebook requires multiple devices in order to run correctly.

+++ {"id": "3f37ca93"}

## `Sharding` describes how array values are laid out in memory across devices

+++ {"id": "W6HsXauGxL6w"}

### Sharding basics, and the `PositionalSharding` subclass

+++ {"id": "NWDyp_EjVHkg"}

To parallelize computation across multiple devices, we first have to describe how the data that computation acts on can be distributed across multiple devices. That means describing distributed layouts conveniently.

In JAX, `Sharding` objects describe distributed memory layouts. They can be used with `jax.device_put` to produce a value with distributed layout.

For example, here's a value with a single-device `Sharding`:

```{code-cell}
:id: s5jXIod7VcWW

import jax
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
```

+++ {"id": "HhCjhK0zXIqX"}

Here, we're using the `jax.debug.visualize_array_sharding` function to show where the value `x` is stored in memory. All of `x` is stored on a single device, so the visualization is pretty boring!

But we can shard `x` across multiple devices by using `jax.device_put` and a `Sharding` object. First, we make a `numpy.ndarray` of `Devices` using `mesh_utils.create_device_mesh`, which takes hardware topology into account for the `Device` order:

```{code-cell}
:id: 8fc925d2
:outputId: 40f044de-0a39-46cb-9e61-2277648a9c0e

jax.debug.visualize_array_sharding(x)
```

+++ {"id": "xKC-WWgc8QGo"}

A quick example of what `jax.Array` can do before we dive into more details:

```{code-cell}
:id: kiZ59Mho5lzk

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Let's create a Sharding object which we will use to distribute
# a value across devices.
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

```{code-cell}
:id: t_Mw8cxU7YSK
:outputId: b44f6bea-c195-4b9a-d872-815112e8ef2e

y = jax.device_put(x, sharding.reshape(4, 2))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "jZ0ZY9Um9Jg4"}

We distributed `x` which was on a single device before to 8 devices.

```{code-cell}
:id: WUzDakoG7a2l
:outputId: b575be88-cbb1-4081-fa98-1ba5c6421921

z = jnp.sin(y)
jax.debug.visualize_array_sharding(z)
```

+++ {"id": "5qccVQoE9tEi"}

After doing an `jnp.sin` operation on a distributed `jax.Array`, the sharding on the `output` was preserved. Also the operation itself happened on multiple devices. To test that, we can do a timing experiment:

```{code-cell}
:id: RdfmbYxr-2I_
:outputId: 50e659d4-9ca0-499e-bd49-66fe59384675

# `x` is present on a single device
%timeit -n 5 -r 5 jnp.sin(x).block_until_ready()
```

```{code-cell}
:id: cNLJQY_r-W75
:outputId: da362eae-810a-438a-81e3-66c16698a456

# `y` is sharded across 8 devices.
%timeit -n 5 -r 5 jnp.sin(y).block_until_ready()
```

+++ {"id": "xWknFQbQ-bzV"}

We can now continue with the rest of the details of `Sharding`s:

```{code-cell}
:id: f4d7c00a

from jax.experimental import mesh_utils
devices = mesh_utils.create_device_mesh((8,))
```

+++ {"id": "lbOKFWmBX1iv"}

Then, we create a `PositionalSharding` and use it with `device_put`:

```{code-cell}
:id: K2PL4LwBX0JE
:outputId: 1c3bbe5e-3377-49a4-a8f0-57e5f224a535

from jax.sharding import PositionalSharding

sharding = PositionalSharding(devices)

x = jax.device_put(x, sharding.reshape(8, 1))
jax.debug.visualize_array_sharding(x)
```

+++ {"id": "TUu69IWXZdTm"}

Here `sharding` is a `PositionalSharding` which acts like an array with sets of devices as elements:

```{code-cell}
:id: d6fd0d23
:outputId: 2eeea24d-553d-4049-82ba-a431a08f5ac8

sharding
```

+++ {"id": "uRLpOcmNj_Vt"}

By writing `PositionalSharding(ndarray_of_devices)`, we fix the device order and the initial shape. Then we can reshape it:

```{code-cell}
:id: b5445d3b
:outputId: 16630351-cb09-42c9-92e5-fe53bc5399d5

sharding.reshape(8, 1)
```

```{code-cell}
:id: pS7xTZeBm6Dt
:outputId: 1d799888-7aed-4415-9812-234059321797

sharding.reshape(4, 2)
```

+++ {"id": "KBu6WLfhm7ra"}

To use `device_put` with a data array `x`, we can reshape the `sharding` into a shape that is _congruent_ with `x.shape`, meaning a shape with the same length as `x.shape` and where each element evenly divides the corresponding element of `x.shape`:
```python
def is_congruent(x_shape: Sequence[int], sharding_shape: Sequence[int]) -> bool:
  return (len(x_shape) == len(sharding_shape) and
          all(d1 % d2 == 0 for d1, d2 in zip(x_shape, sharding_shape)))
```

For example, we can reshape `sharding` to have shape `(4, 2)`, then use it in a `device_put`:

```{code-cell}
:id: 6JhLL3i_sPth
:outputId: 9cc7ee3e-b15c-435a-8f7d-cb127bf0ce07

sharding = sharding.reshape(4, 2)
print(sharding)
```

```{code-cell}
:id: 5FCqZfhWt88c
:outputId: e9abbf02-2a9c-4d34-94e0-8d73cfdc96fa

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "tyg9F-UIsU__"}

Here `y` represents the same _value_ as `x`, but its shards (i.e. slices) are stored in different devices' memories.

Different `sharding` shapes result in different distributed layouts (i.e. shardings) of the result:

```{code-cell}
:id: nt4IbVMkswlO
:outputId: ce106d8a-8ddf-4129-bf5d-1982a28f9160

sharding = sharding.reshape(1, 8)
print(sharding)
```

```{code-cell}
:id: AyZzDpnFuIpz
:outputId: 7cf42e32-4da6-4693-f44e-eae9ec37f203

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "0PuamOvXubcf"}

In some cases, we don't just want to store each slice of `x` in a single device's memory; we might want to _replicate_ some slices, meaning storing copies of a slice's values in multiple devices' memories.

With `PositionalSharding`, we can express replication by calling the reducer method `replicate`:

```{code-cell}
:id: l5t_Mg_Rux6j
:outputId: 68ae315b-7040-4753-f803-514d88a20333

sharding = sharding.reshape(4, 2)
print(sharding.replicate(axis=0, keepdims=True))
```

```{code-cell}
:id: Gi3sDdqAu_8W
:outputId: 3d5b3d72-d68c-481e-854a-b3c2bbea85b6

y = jax.device_put(x, sharding.replicate(axis=0, keepdims=True))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "FzeP0kpTvJv-"}

Here the visualization shows that `x` is sharded two ways along its second dimension (and not sharded along the first dimension), and each of those shards is replicated four ways (i.e. stored in four device memories).

The `replicate` method acts similar to the familiar NumPy array reduction methods like `.sum()` and `.prod()`. It operates along an axis performing a set union. So if `sharding` has shape `(4, 2)`, then `sharding.replicate(0, keepdims=True)` has shape `(1, 2)`, and `sharding.replicate(1, keepdims=True)` has shape `(4, 1)`. Unlike analogous NumPy methods, `keepdims=True` is actually the default, so reduced-over axes aren't squeezed:

```{code-cell}
:id: vDlU8hgJvson
:outputId: 472ddce5-cdec-4434-9a59-995aa1d3cf17

print(sharding.replicate(0).shape)
print(sharding.replicate(1).shape)
```

```{code-cell}
:id: vHWC4womxCdf
:outputId: e5f9a04a-42c1-4c0d-fafa-66c57cd6f774

y = jax.device_put(x, sharding.replicate(1))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "D31t5POXxHHJ"}

### `NamedSharding` gives a way to express shardings with names

+++ {"id": "ayMKWeTmxl-X"}

So far we've worked with `PositionalSharding`, but there are alternative ways to express shardings. In fact, `Sharding` is an interface, and any class that implements that interface can be used with functions like `device_put`.

Another convenient way to express sharding is with the `NamedSharding`:

```{code-cell}
:id: bQCdEAHQ1q8J
:outputId: 41dcae82-22cd-47cc-8f7c-cf0031c1e667

from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding
P = PartitionSpec

devices = mesh_utils.create_device_mesh((4, 2))
mesh = maps.Mesh(devices, axis_names=('a', 'b'))
y = jax.device_put(x, NamedSharding(mesh, P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "OW_Cc92G1-nr"}

We can define a helper function to make things simpler:

```{code-cell}
:id: 2sDUx-VbzvVz

devices = mesh_utils.create_device_mesh((4, 2))
default_mesh = maps.Mesh(devices, axis_names=('a', 'b'))

def mesh_sharding(
    pspec: PartitionSpec, mesh: Optional[maps.Mesh] = None,
) -> NamedSharding:
  if mesh is None:
    mesh = default_mesh
  return NamedSharding(mesh, pspec)
```

```{code-cell}
:id: KirNGYXLzvK6
:outputId: 777c9b33-a19c-414d-8e52-df3338bc0411

y = jax.device_put(x, mesh_sharding(P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "xZ88riVm1mv5"}

Here, we use `P('a', 'b')` to express that the first and second axes of `x` should be sharded over the device mesh axes `'a'` and `'b'`, respectively. We can easily switch to `P('b', 'a')` to shard the axes of `x` over different devices:

```{code-cell}
:id: JJaKU2pJ2eAC
:outputId: ceb3f087-3dbe-4b51-a024-b3af97f12a20

y = jax.device_put(x, mesh_sharding(P('b', 'a')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: nLlRM7DZ25-K
:outputId: c844464e-fec5-4161-ea6d-731fa0f43740

# `None` means that `x` is replicated on the 1st dimension (counting from 0).
y = jax.device_put(x, mesh_sharding(P('a', None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "AqcAsNUgXCZz"}

Here, because `P('a', None)` doesn't mention the `Mesh` axis name `'b'`, we get replication over the axis `'b'`. The `None` here is just acting as a placeholder to line up against the second axis of the value `x`, without expressing sharding over any mesh axis. (As a shorthand, trailing `None`s can be omitted, so that `P('a', None)` means the same thing as `P('a')`. But it doesn't hurt to be explicit!)

To shard only over the second axis of `x`, we can use a `None` placeholder in the `PartitionSpec`:

```{code-cell}
:id: svq_HGHU29HV
:outputId: 318302c1-c865-4e7c-db05-26fc3a435e78

y = jax.device_put(x, mesh_sharding(P(None, 'b')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: oRhRKETlX0RD
:outputId: 966287ce-de6f-4273-fe70-3f4ab5a74bfd

y = jax.device_put(x, mesh_sharding(P(None, 'a')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "--AZgW1P3HFT"}

For a fixed mesh, we can even partition one logical axis of `x` over multiple device mesh axes:

```{code-cell}
:id: ldq5Ws2A3Bbl
:outputId: bd8d18e6-81c0-40b3-b9ae-e1315877d819

y = jax.device_put(x, mesh_sharding(P(('a', 'b'), None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "c1tTFudr3Ae7"}

Using `NamedSharding` makes it easy to define a device mesh once and give its axes names, then just refer to those names in `PartitionSpec`s for each `device_put` as needed.

+++ {"id": "rhWzHgGf4mkg"}

## Computation follows data sharding and is automatically parallelized

+++ {"id": "JukoaRhl4tXJ"}

With sharded data, the compiler can give us parallel computation. In particular, functions decorated with `jax.jit` can operate over sharded arrays without copying data onto a single device. Instead, computation follows sharding: based on the sharding of the input data, the compiler decides shardings for intermediates and output values, and parallelizes their evaluation, even inserting communication operations as necessary.

For example, the simplest computation is an elementwise one:

```{code-cell}
:id: _NqZnEUHgZQv

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

```{code-cell}
:id: x89raigTazVJ
:outputId: 7d59f9fe-6509-416e-ebf3-9bad49146435

x = jax.device_put(x, sharding.reshape(4, 2))
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
:id: D52tW3y-cx32
:outputId: dd895028-1a70-485b-ef46-7accb43b261b

y = jax.device_put(x, sharding.reshape(4, 2).replicate(1))
z = jax.device_put(x, sharding.reshape(4, 2).replicate(0))
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
:id: BUcN-RqtfRml
:outputId: 67e49f89-f76a-4e38-a231-00c76886a90b

x_single = jax.device_put(x, jax.devices()[0])
jax.debug.visualize_array_sharding(x_single)
```

```{code-cell}
:id: iKrmBxJ-fhM9
:outputId: 743ebbfe-ef93-4cd5-e030-694e7018f781

np.allclose(jnp.dot(x_single, x_single),
            jnp.dot(y, z))
```

```{code-cell}
:id: gpcGJ1PSfSAV
:outputId: 463083fd-2e19-4bf8-fb4b-3e5f42934b91

%timeit -n 5 -r 5 jnp.dot(x_single, x_single).block_until_ready()
```

```{code-cell}
:id: 1LMWZuYRfSGT
:outputId: 5c98dc6e-9ddf-4176-8b82-fd2d16517ad2

%timeit -n 5 -r 5 jnp.dot(y, z).block_until_ready()
```

+++ {"id": "gglQIMXJnnJw"}

Even copying a sharded `Array` produces a result with the sharding of the input:

```{code-cell}
:id: sdhFK3VGntbc
:outputId: 662e6811-354b-4d26-be10-eb65ffdf21fa

w_copy = jnp.copy(w)
jax.debug.visualize_array_sharding(w_copy)
```

+++ {"id": "3qfPjJdhgerc"}

So computation follows data placement: when we explicitly shard data with `jax.device_put`, and apply functions to that data, the compiler attempts to parallelize the computation and decide the output sharding. This policy for sharded data is a generalization of [JAX's policy of following explicit device placement](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices).

+++ {"id": "QRB95LaWuT80"}

### When explicit shardings disagree, JAX errors

But what if two arguments to a computation are explicitly placed on different sets of devices, or with incompatible device orders?
In these ambiguous cases, an error is raised:

```{code-cell}
:id: 9xmq1Jbatxwz

import textwrap
from termcolor import colored

def print_exception(e):
  name = colored(f'{type(e).__name__}', 'red')
  print(textwrap.fill(f'{name}: {str(e)}'))
```

```{code-cell}
:id: Yah71IjBqyKD
:outputId: fd28bfaa-bf1c-4236-9c9b-fbf32ca3a8b9

sharding1 = PositionalSharding(jax.devices()[:4])
sharding2 = PositionalSharding(jax.devices()[4:])

y = jax.device_put(x, sharding1.reshape(2, 2))
z = jax.device_put(x, sharding2.reshape(2, 2))
try: y + z
except ValueError as e: print_exception(e)
```

```{code-cell}
:id: HSHDAuJDqyO3
:outputId: a2b76cea-6944-4323-873b-1ba4ced9baa8

devices = jax.devices()
permuted_devices = [devices[i] for i in [0, 1, 2, 3, 6, 7, 4, 5]]

sharding1 = PositionalSharding(devices)
sharding2 = PositionalSharding(permuted_devices)

y = jax.device_put(x, sharding1.reshape(4, 2))
z = jax.device_put(x, sharding2.reshape(4, 2))
try: y + z
except ValueError as e: print_exception(e)
```

+++ {"id": "6ZYcK8eXrn0p"}

We say arrays that have been explicitly placed or sharded with `jax.device_put` are _committed_ to their device(s), and so won't be automatically moved. See the [device placement FAQ](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices) for more information.

When arrays are _not_ explicitly placed or sharded with `jax.device_put`, they are placed _uncommitted_ on the default device.
Unlike committed arrays, uncommitted arrays can be moved and resharded automatically: that is, uncommitted arrays can be arguments to a computation even if other arguments are explicitly placed on different devices.

For example, the output of `jnp.zeros`, `jnp.arange`, and `jnp.array` are uncommitted:

```{code-cell}
:id: hDa3ogiwrvZx
:outputId: f2c186ee-4626-4352-f9af-794c042e2b89

y = jax.device_put(x, sharding1.reshape(4, 2))
y + jnp.ones_like(y)
y + jnp.arange(y.size).reshape(y.shape)
print('no error!')
```

+++ {"id": "dqMKl79NaIWF"}

## Constraining shardings of intermediates in `jit`ted code

+++ {"id": "g4LrDDcJwkHc"}

While the compiler will attempt to decide how a function's intermediate values and outputs should be sharded, we can also give it hints using `jax.lax.with_sharding_constraint`. Using `jax.lax.with_sharding_constraint` is much like `jax.device_put`, except we use it inside staged-out (i.e. `jit`-decorated) functions:

```{code-cell}
:id: _acYTsBsxpyP

jax.lax.with_sharding_constraint = jax.experimental.pjit.with_sharding_constraint
```

```{code-cell}
:id: KIi13NFHxz77

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
```

```{code-cell}
:id: UTRs-Zf2x8oJ

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
x = jax.device_put(x, sharding.reshape(4, 2))
```

```{code-cell}
:id: fkm3YKfZwkWt

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, sharding.reshape(2, 4))
  return y
```

```{code-cell}
:id: tIglE_fayQqw
:outputId: ebcf2cb5-e67f-4d53-cc29-10212a88e127

jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: DzuzKQhG2QLE

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, sharding.replicate())
  return y
```

```{code-cell}
:id: lmDxCQ1W2TlD
:outputId: 7664dabd-9ba2-4d53-a7da-55ead835c3f5

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

**⚠️ WARNING: The following is meant to be a demonstration of automatic sharding propagation with `jax.Array`, but it is _not a recommended practice_.**

For neural network training, it is a good idea to constraint parameter and input sharding to be the same at every training step which can be achieved e.g. using `pjit`'s `out_axis_resources` parameter and `with_sharding_constraint`.

+++ {"id": "3ii_UPkG3gzP"}

We can use `jax.device_put` and `jax.jit`'s computation-follows-sharding features to parallelize computation in neural networks. Here are some simple examples, based on this basic neural network:

```{code-cell}
:id: sDAeJoNp_VyP

import jax
import jax.numpy as jnp
```

```{code-cell}
:id: t-J6YtpA2db0

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
:id: 4USnNl6w4Y1K

loss_jit = jax.jit(loss)
gradfun = jax.jit(jax.grad(loss))
```

```{code-cell}
:id: nfqG0N1g2dhk

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

params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)
```

+++ {"id": "sJv_h0AS2drh"}

### 8-way batch data parallelism

```{code-cell}
:id: uxZ4Czqyzrc5

sharding = PositionalSharding(jax.devices()).reshape(8, 1)
```

```{code-cell}
:id: q9maIR6K4T9r

batch = jax.device_put(batch, sharding)
params = jax.device_put(params, sharding.replicate())
```

```{code-cell}
:id: CtKIMM6ry7Ov
:outputId: 336c1892-a3e1-4a2a-a0ee-a16dd7a24103

loss_jit(params, batch)
```

```{code-cell}
:id: tAM6NQkly8lw
:outputId: 8e5caf04-3c38-426b-8e8a-823760d6688c

step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))
```

```{code-cell}
:id: Eix05eVQy-LZ
:outputId: 8f0d79b4-9ff3-47a9-d25f-ecc770e72195

%timeit -n 5 -r 5 gradfun(params, batch)[0][0].block_until_ready()
```

```{code-cell}
:id: W-19ajlSy_gF

batch_single = jax.device_put(batch, jax.devices()[0])
params_single = jax.device_put(params, jax.devices()[0])
```

```{code-cell}
:id: DBHfeKyUzBD9
:outputId: a96ee2c8-179d-4dfa-aa72-5c0076ff6881

%timeit -n 5 -r 5 gradfun(params_single, batch_single)[0][0].block_until_ready()
```

+++ {"id": "3AjeeB7B4NP6"}

### 4-way batch data parallelism and 2-way model tensor parallelism

```{code-cell}
:id: gw1WZyXu4owx

sharding = sharding.reshape(4, 2)
```

```{code-cell}
:id: P0s_ibu8z0hW
:outputId: c9bafdc7-a811-4a38-db6b-89d55355a241

batch = jax.device_put(batch, sharding.replicate(1))
jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(batch[1])
```

```{code-cell}
:id: 7kNJVPBjz5nq

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params

W1 = jax.device_put(W1, sharding.replicate())
b1 = jax.device_put(b1, sharding.replicate())

W2 = jax.device_put(W2, sharding.replicate(0))
b2 = jax.device_put(b2, sharding.replicate(0))

W3 = jax.device_put(W3, sharding.replicate(0).T)
b3 = jax.device_put(b3, sharding.replicate())

W4 = jax.device_put(W4, sharding.replicate())
b4 = jax.device_put(b4, sharding.replicate())

params = (W1, b1), (W2, b2), (W3, b3), (W4, b4)
```

```{code-cell}
:id: I8ZJiiGb0HJk
:outputId: 5c406158-6b71-45ec-bdad-2d6b39b237ca

jax.debug.visualize_array_sharding(W2)
```

```{code-cell}
:id: t2fsJ_Ow0LgK
:outputId: 50ad8513-cb50-44d3-e0df-551ddd3c9fcb

jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
:id: xnNgGB7-0Nh4
:outputId: 338d8bd4-0e0d-4a4a-99f7-e408b95ab274

print(loss_jit(params, batch))
```

```{code-cell}
:id: ygV3-IBV0Qx3

step_size = 1e-5

for _ in range(30):
    grads = gradfun(params, batch)
    params = [(W - step_size * dW, b - step_size * db)
              for (W, b), (dW, db) in zip(params, grads)]
```

```{code-cell}
:id: VWXN24Xh0Tkc
:outputId: 13379078-3863-42f3-e5f4-368095c3a35c

print(loss_jit(params, batch))
```

```{code-cell}
:id: Cq3TzYU70Vfd
:outputId: 9aae2051-3e4f-4918-c221-81d21edb563a

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params
jax.debug.visualize_array_sharding(W2)
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
:id: hAeLBs9D0Z8T
:outputId: 3eeacdd1-b44b-46e7-9868-dbdbca4a78e2

%timeit -n 10 -r 10 gradfun(params, batch)[0][0].block_until_ready()
```

+++ {"id": "3diqi5VRBy6S"}

## Sharp bits

+++ {"id": "OTfoXNnxFYDJ"}

### Generating random numbers

JAX comes with a functional, deterministic [random number generator](https://jax.readthedocs.io/en/latest/jep/263-prng.html). It underlies the various sampling functions in the [`jax.random` module](https://jax.readthedocs.io/en/latest/jax.random.html), such as `jax.random.uniform`.

JAX's random numbers are produced by a counter-based PRNG, so in principle, random number generation should be a pure map over counter values. A pure map is a trivially partitionable operation in principle. It should require no cross-device communication, nor any redundant computation across devices.

However, the existing stable RNG implementation is not automatically partitionable, for historical reasons.

+++ {"id": "ht_zYFVXNrjN"}

Consider the following example, where a function draws random uniform numbers and adds them to the input, elementwise:

```{code-cell}
:id: quUwyGoiHub2

@jax.jit
def f(key, x):
  numbers = jax.random.uniform(key, x.shape)
  return x + numbers

key = jax.random.PRNGKey(42)
x_sharding = jax.sharding.PositionalSharding(jax.devices())
x = jax.device_put(jnp.arange(24), x_sharding)
```

+++ {"id": "ZgSA9x9NLMaP"}

On a partitioned input, the function `f` produces output that is also partitioned:

```{code-cell}
:id: nxPHw0loLgFd
:outputId: 59514362-61c5-4181-c7e9-ad7777218779

jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "WnjlWDUYLkp6"}

But if we inspect the compiled computation for `f` on this partitioned input, we see that it does involve some communication:

```{code-cell}
:id: uf6hTkjvKifH
:outputId: 3af4c656-0753-4dd5-e4be-0097bba43256

f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "AXp9i8fbL8DD"}

One way to work around this is to configure JAX with the experimental upgrade flag `jax_threefry_partitionable`. With the flag on, the "collective permute" operation is now gone from the compiled computation:

```{code-cell}
:id: G87r_Aq6Ts_F
:outputId: 5c8cd729-5f2a-450d-ee40-fb788f91707a

jax.config.update('jax_threefry_partitionable', True)
f_exe = f.lower(key, x).compile()
print('Communicating?', 'collective-permute' in f_exe.as_text())
```

+++ {"id": "WV8ZccM5SXOU"}

The output is still partitioned:

```{code-cell}
:id: 8RplTPyRSTbW
:outputId: 74e2ef51-c5c0-4c25-cc90-2f465961bdae

jax.debug.visualize_array_sharding(f(key, x))
```

+++ {"id": "kaK--hPmSPpV"}

One caveat, however, is that _the random values produced may be different than before_, even though they were generated by the same random key:

```{code-cell}
:id: f_EjYjOpSO18
:outputId: b50278f2-927d-4aea-fb04-cdd5c2fc1a66

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
