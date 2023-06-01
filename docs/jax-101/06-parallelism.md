---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "tCOWitsAS1EE"}

# Parallel Evaluation in JAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/06-parallelism.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/google/jax/blob/main/docs/jax-101/06-parallelism.ipynb)

*Authors: Vladimir Mikulik & Roman Ring*

In this section we will discuss the facilities built into JAX for single-program, multiple-data (SPMD) code.

SPMD refers to a parallelism technique where the same computation (e.g., the forward pass of a neural net) is run on different input data (e.g., different inputs in a batch) in parallel on different devices (e.g., several TPUs).

Conceptually, this is not very different from vectorisation, where the same operations occur in parallel in different parts of memory on the same device. We have already seen that vectorisation is supported in JAX as a program transformation, `jax.vmap`. JAX supports device parallelism analogously, using `jax.pmap` to transform a function written for one device into a function that runs in parallel on multiple devices. This colab will teach you all about it.

+++ {"id": "7mCgBzix2fd3"}

## TPU Setup

This notebook requires multiple accelerators and we recommend running it using Kaggle TPU VMs.

+++ {"id": "gN6VbcdRTcdE"}

Next run the following to see the TPU devices you have available:

```{code-cell} ipython3
:id: tqbpCcqY3Cn7
:outputId: 1fb88cf7-35f7-4565-f370-51586213b988

import jax
jax.devices()
```

+++ {"id": "4_EDa0Dlgtf8"}

## The basics

The most basic use of `jax.pmap` is completely analogous to `jax.vmap`, so let's return to the convolution example from the [Vectorisation notebook](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/03-vectorization.ipynb).

```{code-cell} ipython3
:id: IIQKBr-CgtD2
:outputId: 6e7f8755-fdfd-4cf9-e2b5-a10c5a870dd4

import numpy as np
import jax.numpy as jnp

x = np.arange(5)
w = np.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```

+++ {"id": "lqxz9NNJOQ9Z"}

Now, let's convert our `convolve` function into one that runs on entire batches of data. In anticipation of spreading the batch across several devices, we'll make the batch size equal to the number of devices:

```{code-cell} ipython3
:id: ll-hEa0jihzx
:outputId: 788be05a-10d4-4a05-8d9d-49d0083541ab

n_devices = jax.local_device_count() 
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)

xs
```

```{code-cell} ipython3
:id: mi-nysDWYbn4
:outputId: 2d115fc3-52f5-4a68-c3a7-115111a83657

ws
```

+++ {"id": "8kseIB09YWJw"}

As before, we can vectorise using `jax.vmap`:

```{code-cell} ipython3
:id: TNb9HsFXYVOI
:outputId: 2e60e07a-6687-49ab-a455-60d2ec484363

jax.vmap(convolve)(xs, ws)
```

+++ {"id": "TDF1vzt_5GMC"}

To spread out the computation across multiple devices, just replace `jax.vmap` with `jax.pmap`:

```{code-cell} ipython3
:id: KWoextrails4
:outputId: bad1fbb7-226a-4538-e442-20ce0c1c8fad

jax.pmap(convolve)(xs, ws)
```

+++ {"id": "E69cVxQPksxe"}

Note that the parallelized `convolve` returns a `jax.Array`. That is because the elements of this array are sharded across all of the devices used in the parallelism. If we were to run another parallel computation, the elements would stay on their respective devices, without incurring cross-device communication costs.

```{code-cell} ipython3
:id: P9dUyk-ciquy
:outputId: 99ea4c6e-cff7-4611-e9e5-bf016fa9716c

jax.pmap(convolve)(xs, jax.pmap(convolve)(xs, ws))
```

+++ {"id": "iuHqht-OYqca"}

The outputs of the inner `jax.pmap(convolve)` never left their devices when being fed into the outer `jax.pmap(convolve)`.

+++ {"id": "vEFAJXN2q3dV"}

## Specifying `in_axes`

Like with `vmap`, we can use `in_axes` to specify whether an argument to the parallelized function should be broadcast (`None`), or whether it should be split along a given axis. Note, however, that unlike `vmap`, only the leading axis (`0`) is supported by `pmap` at the time of writing this guide.

```{code-cell} ipython3
:id: 6Es5WVuRlXnB
:outputId: 7e9612ae-d6e0-4d79-a228-f0403fcf8237

jax.pmap(convolve, in_axes=(0, None))(xs, w)
```

+++ {"id": "EoN6drHDOlk4"}

Notice how we get equivalent output to what we observe above with `jax.pmap(convolve)(xs, ws)`, where we manually replicated `w` when creating `ws`. Here, it is replicated via broadcasting, by specifying it as `None` in `in_axes`.

+++ {"id": "rRE8STSU5cjx"}

Keep in mind that when calling the transformed function, the size of the specified axis in arguments must not exceed the number of devices available to the host.

+++ {"id": "0lZnqImd7G6U"}

## `pmap` and `jit`

`jax.pmap` JIT-compiles the function given to it as part of its operation, so there is no need to additionally `jax.jit` it.

+++ {"id": "1jZqk_2AwO4y"}

## Communication between devices

The above is enough to perform simple parallel operations, e.g. batching a simple MLP forward pass across several devices. However, sometimes we need to pass information between the devices. For example, perhaps we are interested in normalizing the output of each device so they sum to 1.
For that, we can use special [collective ops](https://jax.readthedocs.io/en/latest/jax.lax.html#parallel-operators) (such as the `jax.lax.p*` ops `psum`, `pmean`, `pmax`, ...). In order to use the collective ops we must specify the name of the `pmap`-ed axis through `axis_name` argument, and then refer to it when calling the op. Here's how to do that:

```{code-cell} ipython3
:id: 0nCxGwqmtd3w
:outputId: 6f9c93b0-51ed-40c5-ca5a-eacbaf40e686

def normalized_convolution(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  output = jnp.array(output)
  return output / jax.lax.psum(output, axis_name='p')

jax.pmap(normalized_convolution, axis_name='p')(xs, ws)
```

+++ {"id": "9ENYsJS42YVK"}

The `axis_name` is just a string label that allows collective operations like `jax.lax.psum` to refer to the axis bound by `jax.pmap`. It can be named anything you want -- in this case, `p`. This name is essentially invisible to anything but those functions, and those functions use it to know which axis to communicate across.

`jax.vmap` also supports `axis_name`, which allows `jax.lax.p*` operations to be used in the vectorisation context in the same way they would be used in a `jax.pmap`:

```{code-cell} ipython3
:id: nT61xAYJUqCW
:outputId: e8831025-78a6-4a2b-a60a-3c77b35214ef

jax.vmap(normalized_convolution, axis_name='p')(xs, ws)
```

+++ {"id": "JSK-9dbWWV2O"}

Note that `normalized_convolution` will no longer work without being transformed by `jax.pmap` or `jax.vmap`, because `jax.lax.psum` expects there to be a named axis (`'p'`, in this case), and those two transformations are the only way to bind one.

## Nesting `jax.pmap` and `jax.vmap`

The reason we specify `axis_name` as a string is so we can use collective operations when nesting `jax.pmap` and `jax.vmap`. For example:

```python
jax.vmap(jax.pmap(f, axis_name='i'), axis_name='j')
```

A `jax.lax.psum(..., axis_name='i')` in `f` would refer only to the pmapped axis, since they share the `axis_name`. 

In general, `jax.pmap` and `jax.vmap` can be nested in any order, and with themselves (so you can have a `pmap` within another `pmap`, for instance).

+++ {"id": "WzQHxnHkCxej"}

## Example

Here's an example of a regression training loop with data parallelism, where each batch is split into sub-batches which are evaluated on separate devices.

There are two places to pay attention to:
* the `update()` function
* the replication of parameters and splitting of data across devices.

If this example is too confusing, you can find the same example, but without parallelism, in the next notebook, [State in JAX](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/07-state.ipynb). Once that example makes sense, you can compare the differences to understand how parallelism changes the picture.

```{code-cell} ipython3
:id: cI8xQqzRrc-4

from typing import NamedTuple, Tuple
import functools

class Params(NamedTuple):
  weight: jnp.ndarray
  bias: jnp.ndarray


def init(rng) -> Params:
  """Returns the initial model params."""
  weights_key, bias_key = jax.random.split(rng)
  weight = jax.random.normal(weights_key, ())
  bias = jax.random.normal(bias_key, ())
  return Params(weight, bias)


def loss_fn(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
  """Computes the least squares error of the model's predictions on x against y."""
  pred = params.weight * xs + params.bias
  return jnp.mean((pred - ys) ** 2)

LEARNING_RATE = 0.005

# So far, the code is identical to the single-device case. Here's what's new:


# Remember that the `axis_name` is just an arbitrary string label used
# to later tell `jax.lax.pmean` which axis to reduce over. Here, we call it
# 'num_devices', but could have used anything, so long as `pmean` used the same.
@functools.partial(jax.pmap, axis_name='num_devices')
def update(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> Tuple[Params, jnp.ndarray]:
  """Performs one SGD update step on params using the given data."""

  # Compute the gradients on the given minibatch (individually on each device).
  loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

  # Combine the gradient across all devices (by taking their mean).
  grads = jax.lax.pmean(grads, axis_name='num_devices')

  # Also combine the loss. Unnecessary for the update, but useful for logging.
  loss = jax.lax.pmean(loss, axis_name='num_devices')

  # Each device performs its own update, but since we start with the same params
  # and synchronise gradients, the params stay in sync.
  new_params = jax.tree_map(
      lambda param, g: param - g * LEARNING_RATE, params, grads)

  return new_params, loss
```

+++ {"id": "RWce8YZ4Pcmf"}

Here's how `update()` works:

Undecorated and without the `pmean`s, `update()` takes data tensors of shape `[batch, ...]`, computes the loss function on that batch and evaluates its gradients.

We want to spread the `batch` dimension across all available devices. To do that, we add a new axis using `pmap`. The arguments to the decorated `update()` thus need to have shape `[num_devices, batch_per_device, ...]`. So, to call the new `update()`, we'll need to reshape data batches so that what used to be `batch` is reshaped to `[num_devices, batch_per_device]`. That's what `split()` does below. Additionally, we'll need to replicate our model parameters, adding the `num_devices` axis. This reshaping is how a pmapped function knows which devices to send which data.

At some point during the update step, we need to combine the gradients computed by each device -- otherwise, the updates performed by each device would be different. That's why we use `jax.lax.pmean` to compute the mean across the `num_devices` axis, giving us the average gradient of the batch. That average gradient is what we use to compute the update.

Aside on naming: here, we use `num_devices` for the `axis_name` for didactic clarity while introducing `jax.pmap`. However, in some sense that is tautologous: any axis introduced by a pmap will represent a number of devices. Therefore, it's common to see the axis be named something semantically meaningful, like `batch`, `data` (signifying data parallelism) or `model` (signifying model parallelism).

```{code-cell} ipython3
:id: _CTtLrsQ-0kK

# Generate true data from y = w*x + b + noise
true_w, true_b = 2, -1
xs = np.random.normal(size=(128, 1))
noise = 0.5 * np.random.normal(size=(128, 1))
ys = xs * true_w + true_b + noise

# Initialise parameters and replicate across devices.
params = init(jax.random.PRNGKey(123))
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
```

+++ {"id": "dmCMyLP9SV99"}

So far, we've just constructed arrays with an additional leading dimension. The params are all still all on the host (CPU). `pmap` will communicate them to the devices when `update()` is first called, and each copy will stay on its own device subsequently.

```{code-cell} ipython3
:id: YSCgHguTSdGW
:outputId: a8bf28df-3747-4d49-e340-b7696cf0c27d

type(replicated_params.weight)
```

+++ {"id": "90VtjPbeY-hD"}

The params will become a jax.Array when they are returned by our pmapped `update()` (see further down).

+++ {"id": "eGVKxk1CV-m1"}

We do the same to the data:

```{code-cell} ipython3
:id: vY61QJoFWCII
:outputId: f436a15f-db97-44cc-df33-bbb4ff222987

def split(arr):
  """Splits the first axis of `arr` evenly across the number of devices."""
  return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

# Reshape xs and ys for the pmapped `update()`.
x_split = split(xs)
y_split = split(ys)

type(x_split)
```

+++ {"id": "RzfJ-oK5WERq"}

The data is just a reshaped vanilla NumPy array. Hence, it cannot be anywhere but on the host, as NumPy runs on CPU only. Since we never modify it, it will get sent to the device at each `update` call, like in a real pipeline where data is typically streamed from CPU to the device at each step.

```{code-cell} ipython3
:id: atOTi7EeSQw-
:outputId: c8daf141-63c4-481f-afa5-684c5f7b698d

def type_after_update(name, obj):
  print(f"after first `update()`, `{name}` is a", type(obj))

# Actual training loop.
for i in range(1000):

  # This is where the params and data gets communicated to devices:
  replicated_params, loss = update(replicated_params, x_split, y_split)

  # The returned `replicated_params` and `loss` are now both jax.Arrays,
  # indicating that they're on the devices.
  # `x_split`, of course, remains a NumPy array on the host.
  if i == 0:
    type_after_update('replicated_params.weight', replicated_params.weight)
    type_after_update('loss', loss)
    type_after_update('x_split', x_split)

  if i % 100 == 0:
    # Note that loss is actually an array of shape [num_devices], with identical
    # entries, because each device returns its copy of the loss.
    # So, we take the first element to print it.
    print(f"Step {i:3d}, loss: {loss[0]:.3f}")


# Plot results.

# Like the loss, the leaves of params have an extra leading dimension,
# so we take the params from the first device.
params = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))
```

```{code-cell} ipython3
:id: rvVCACv9UZcF
:outputId: 5c472d0f-1236-401b-be55-86e3dc43875d

import matplotlib.pyplot as plt
plt.scatter(xs, ys)
plt.plot(xs, params.weight * xs + params.bias, c='red', label='Model Prediction')
plt.legend()
plt.show()
```

+++ {"id": "4wFJcqbhbn81"}

## Aside: hosts and devices in JAX

When running on TPU, the idea of a 'host' becomes important. A host is the CPU that manages several devices. A single host can only manage so many devices (usually 8), so when running very large parallel programs, multiple hosts are needed, and some finesse is required to manage them.

```{code-cell} ipython3
:id: 3DO8NwW5hurX
:outputId: 6df0bdd7-fee2-4805-9bfe-38e41bdaeb50

jax.devices()
```

+++ {"id": "sJwayfCoy15a"}

When running on CPU you can always emulate an arbitrary number of devices with a nifty `--xla_force_host_platform_device_count` XLA flag, e.g. by executing the following before importing JAX:
```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.devices()
```
```
[CpuDevice(id=0),
 CpuDevice(id=1),
 CpuDevice(id=2),
 CpuDevice(id=3),
 CpuDevice(id=4),
 CpuDevice(id=5),
 CpuDevice(id=6),
 CpuDevice(id=7)]
```
This is especially useful for debugging and testing locally or even for prototyping in Colab since a CPU runtime is faster to (re-)start.
