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

+++ {"id": "pFtQjv4SzHRj"}

**The features required by `jax.Array` are not supported by the Colab TPU runtime at this time.**

```{code-cell}
:id: 41dde63b

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

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

### `Sharding` describes how array values are laid out in memory across devices

+++ {"id": "W6HsXauGxL6w"}

#### Sharding basics, and the `PositionalSharding` subclass

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
:outputId: 97d6e6b7-7c4f-460a-ed69-0622f385cbbf

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
:outputId: 99a0f04c-6e5a-42b9-e6d0-f022ffdfa09c

y = jax.device_put(x, sharding.reshape(4, 2))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "jZ0ZY9Um9Jg4"}

We distributed `x` which was on a single device before to 8 devices.

```{code-cell}
:id: WUzDakoG7a2l
:outputId: ece03df1-ef85-49c6-f884-1f1be03e765f

z = jnp.sin(y)
jax.debug.visualize_array_sharding(z)
```

+++ {"id": "5qccVQoE9tEi"}

After doing an `jnp.sin` operation on a distributed `jax.Array`, the sharding on the `output` was preserved. Also the operation itself happened on multiple devices. To test that, we can do a timing experiment:

```{code-cell}
:id: RdfmbYxr-2I_
:outputId: 8d858a67-b8e8-4193-fdaa-0c57431821ab

# `x` is present on a single device
%timeit -n 5 -r 5 jnp.sin(x).block_until_ready()
```

```{code-cell}
:id: cNLJQY_r-W75
:outputId: b38b5f98-67de-46cf-8892-8f7b5337fc6c

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
:outputId: a38aa107-33c4-4a85-b1c0-6a223dc22df1

from jax.sharding import PositionalSharding

sharding = PositionalSharding(devices)

x = jax.device_put(x, sharding.reshape(8, 1))
jax.debug.visualize_array_sharding(x)
```

+++ {"id": "TUu69IWXZdTm"}

Here `sharding` is a `PositionalSharding` which acts like an array with sets of devices as elements:

```{code-cell}
:id: d6fd0d23
:outputId: ee8e5ac2-06b2-4aab-fa89-040c40495f0b

sharding
```

+++ {"id": "uRLpOcmNj_Vt"}

By writing `PositionalSharding(ndarray_of_devices)`, we fix the device order and the initial shape. Then we can reshape it:

```{code-cell}
:id: b5445d3b
:outputId: f6e9af0f-fee7-47e6-e6f0-5ceb205dbaf1

sharding.reshape(8, 1)
```

```{code-cell}
:id: pS7xTZeBm6Dt
:outputId: 89acd8eb-2889-418f-94ab-d9cbfecd456b

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
:outputId: 89928f9a-9b33-4b0f-860b-68c39fd542f7

sharding = sharding.reshape(4, 2)
print(sharding)
```

```{code-cell}
:id: 5FCqZfhWt88c
:outputId: dffe1f80-09f8-4266-d9d9-26b63cd11495

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "tyg9F-UIsU__"}

Here `y` represents the same _value_ as `x`, but its shards (i.e. slices) are stored in different devices' memories.

Different `sharding` shapes result in different distributed layouts (i.e. shardings) of the result:

```{code-cell}
:id: nt4IbVMkswlO
:outputId: 51f39abc-451b-4543-c5f5-63588d6f357c

sharding = sharding.reshape(1, 8)
print(sharding)
```

```{code-cell}
:id: AyZzDpnFuIpz
:outputId: 6469d5f7-0489-4435-e690-c946195f96f3

y = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "0PuamOvXubcf"}

In some cases, we don't just want to store each slice of `x` in a single device's memory; we might want to _replicate_ some slices, meaning storing copies of a slice's values in multiple devices' memories.

With `PositionalSharding`, we can express replication by calling the reducer method `replicate`:

```{code-cell}
:id: l5t_Mg_Rux6j
:outputId: 51768d97-50c1-4bc0-fe36-1869c2e9f2c1

sharding = sharding.reshape(4, 2)
print(sharding.replicate(axis=0, keepdims=True))
```

```{code-cell}
:id: Gi3sDdqAu_8W
:outputId: 523fc513-9a30-49b7-9152-051033615a2d

y = jax.device_put(x, sharding.replicate(axis=0, keepdims=True))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "FzeP0kpTvJv-"}

Here the visualization shows that `x` is sharded two ways along its second dimension (and not sharded along the first dimension), and each of those shards is replicated four ways (i.e. stored in four device memories).

The `replicate` method acts similar to the familiar NumPy array reduction methods like `.sum()` and `.prod()`. It operates along an axis performing a set union. So if `sharding` has shape `(4, 2)`, then `sharding.replicate(0, keepdims=True)` has shape `(1, 2)`, and `sharding.replicate(1, keepdims=True)` has shape `(4, 1)`. Unlike analogous NumPy methods, `keepdims=True` is actually the default, so reduced-over axes aren't squeezed:

```{code-cell}
:id: vDlU8hgJvson
:outputId: 970d6942-0cb0-4e4e-e471-01cbfb343179

print(sharding.replicate(0).shape)
print(sharding.replicate(1).shape)
```

```{code-cell}
:id: vHWC4womxCdf
:outputId: 9095b282-d87e-4df7-c99f-248acaf3d412

y = jax.device_put(x, sharding.replicate(1))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "D31t5POXxHHJ"}

#### `NamedSharding` gives a way to express shardings with names

+++ {"id": "ayMKWeTmxl-X"}

So far we've worked with `PositionalSharding`, but there are alternative ways to express shardings. In fact, `Sharding` is an interface, and any class that implements that interface can be used with functions like `device_put`.

Another convenient way to express sharding is with the `NamedSharding`:

```{code-cell}
:id: bQCdEAHQ1q8J
:outputId: 2c423db8-4dcf-49f1-aed1-6ac1ba6cfbf8

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
:outputId: 274ae2e6-37a3-462f-b437-5c8d40955ec0

y = jax.device_put(x, mesh_sharding(P('a', 'b')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "xZ88riVm1mv5"}

Here, we use `P('a', 'b')` to express that the first and second axes of `x` should be sharded over the device mesh axes `'a'` and `'b'`, respectively. We can easily switch to `P('b', 'a')` to shard the axes of `x` over different devices:

```{code-cell}
:id: JJaKU2pJ2eAC
:outputId: 6fcd07f7-a7df-4805-f6dd-d66c709f8d90

y = jax.device_put(x, mesh_sharding(P('b', 'a')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: nLlRM7DZ25-K
:outputId: 6e2f211b-99b7-4192-bf58-ae47fb0e0c81

# `None` means that `x` is replicated on the 1st dimension (counting from 0).
y = jax.device_put(x, mesh_sharding(P('a', None)))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "AqcAsNUgXCZz"}

Here, because `P('a', None)` doesn't mention the `Mesh` axis name `'b'`, we get replication over the axis `'b'`. The `None` here is just acting as a placeholder to line up against the second axis of the value `x`, without expressing sharding over any mesh axis. (As a shorthand, trailing `None`s can be omitted, so that `P('a', None)` means the same thing as `P('a')`. But it doesn't hurt to be explicit!)

To shard only over the second axis of `x`, we can use a `None` placeholder in the `PartitionSpec`:

```{code-cell}
:id: svq_HGHU29HV
:outputId: 6e6a3111-faa9-42e4-d16b-75eb08ca315a

y = jax.device_put(x, mesh_sharding(P(None, 'b')))
jax.debug.visualize_array_sharding(y)
```

```{code-cell}
:id: oRhRKETlX0RD
:outputId: fd305fbf-5060-4d61-8f67-cce3dcb32736

y = jax.device_put(x, mesh_sharding(P(None, 'a')))
jax.debug.visualize_array_sharding(y)
```

+++ {"id": "--AZgW1P3HFT"}

For a fixed mesh, we can even partition one logical axis of `x` over multiple device mesh axes:

```{code-cell}
:id: ldq5Ws2A3Bbl
:outputId: 5f6b28fd-2747-4854-d272-de8fee2fe0c7

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
:outputId: e2fb6c30-3ad7-4400-ce61-51d38041824a

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
:outputId: 3319f678-133e-43c8-f3d2-624dc9cfd8f7

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
:outputId: c7fe34fc-741c-46c3-da24-e095ba39573c

x_single = jax.device_put(x, jax.devices()[0])
jax.debug.visualize_array_sharding(x_single)
```

```{code-cell}
:id: iKrmBxJ-fhM9
:outputId: 4f36ae48-46e7-47ba-8318-8f824b78f6e5

np.allclose(jnp.dot(x_single, x_single),
            jnp.dot(y, z))
```

```{code-cell}
:id: gpcGJ1PSfSAV
:outputId: 3f7d522f-fd8a-4752-917e-502bfbe64750

%timeit -n 5 -r 5 jnp.dot(x_single, x_single).block_until_ready()
```

```{code-cell}
:id: 1LMWZuYRfSGT
:outputId: ff08bcb3-4f2f-4397-dec5-8a029eb3259a

%timeit -n 5 -r 5 jnp.dot(y, z).block_until_ready()
```

+++ {"id": "gglQIMXJnnJw"}

Even copying a sharded `Array` produces a result with the sharding of the input:

```{code-cell}
:id: sdhFK3VGntbc
:outputId: e2309311-6b58-4d5b-d449-802cadfa35e4

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
:outputId: 544a4ca2-86d1-4df9-a4fd-7fc252471000

sharding1 = PositionalSharding(jax.devices()[:4])
sharding2 = PositionalSharding(jax.devices()[4:])

y = jax.device_put(x, sharding1.reshape(2, 2))
z = jax.device_put(x, sharding2.reshape(2, 2))
try: y + z
except ValueError as e: print_exception(e)
```

```{code-cell}
:id: HSHDAuJDqyO3
:outputId: 79b629ab-6715-44e0-c3f1-c531ad513150

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
:outputId: 009bd5f2-742d-415c-d0c7-a87cb3ba8e5d

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
:outputId: db786eab-94e7-47ed-9dae-a82d1ccc979f

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
:outputId: b8c23c00-bc35-46bc-b0a9-f8f381aa01fe

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
:outputId: d7e41395-9709-4ead-c3cd-03dd5096eafa

loss_jit(params, batch)
```

```{code-cell}
:id: tAM6NQkly8lw
:outputId: cf59f694-ccae-42d5-871e-5cc10e2547ba

step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))
```

```{code-cell}
:id: Eix05eVQy-LZ
:outputId: dfe51c4b-edbe-4a14-81ca-7a787973f30a

%timeit -n 5 -r 5 gradfun(params, batch)[0][0].block_until_ready()
```

```{code-cell}
:id: W-19ajlSy_gF

batch_single = jax.device_put(batch, jax.devices()[0])
params_single = jax.device_put(params, jax.devices()[0])
```

```{code-cell}
:id: DBHfeKyUzBD9
:outputId: f0713c06-b3ac-4692-e30e-f33eeed34da5

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
:outputId: 24588445-88e3-4563-c9bd-d6847f5453a9

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
:outputId: 49789525-1e02-4e69-ab2a-987387e051c3

jax.debug.visualize_array_sharding(W2)
```

```{code-cell}
:id: t2fsJ_Ow0LgK
:outputId: 6f8fbff9-74ee-4191-f5af-1e1882628230

jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
:id: xnNgGB7-0Nh4
:outputId: e3470e88-d0c9-4f4a-e861-5a8dc2a6c164

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
:outputId: 0696952f-efa8-4e4c-ee57-6546b75eafdb

print(loss_jit(params, batch))
```

```{code-cell}
:id: Cq3TzYU70Vfd
:outputId: 3a956c68-498d-439f-c621-bf0f9339dc0b

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params
jax.debug.visualize_array_sharding(W2)
jax.debug.visualize_array_sharding(W3)
```

```{code-cell}
:id: hAeLBs9D0Z8T
:outputId: 4c2cb486-1ede-4b64-d2f6-2236c99bc882

%timeit -n 10 -r 10 gradfun(params, batch)[0][0].block_until_ready()
```
