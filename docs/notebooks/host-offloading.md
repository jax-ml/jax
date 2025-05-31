---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "bQbS50fIdHw1"}

(host-offloading)=
# JAX Memories and Host Offloading

<!--* freshness: { reviewed: '2025-04-10' } *-->

This tutorial provides a practical introduction to host offloading techniques in JAX, focusing on:

- Activation offloading
- Parameter offloading

By applying offloading strategies, you can better manage memory resources and reduce memory pressure on your devices. To implement these strategies effectively, you'll need to understand JAX's core mechanisms for data placement and movement.

## Building Blocks for Offloading

JAX provides several key components for controlling where and how data are stored and moved between the host and the device memory. In the following sections, you'll explore:

- How to specify data distribution with sharding
- How to control memory placement between host and device
- How to manage data movement in jitted functions

### NamedSharding and Memory Kinds

{class}`~jax.sharding.NamedSharding` defines how data are distributed across devices. It includes:

- Basic data distribution configuration
- `memory_kind` parameter for specifying memory type (`device` or `pinned_host`)
- By default, `memory_kind` is set to `device` memory
- `with_memory_kind` method for creating new sharding with modified memory type

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: f-6sxUlqrlBn
outputId: 691a3df2-8341-44a9-a4a0-5521c2d891e3
---
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

# Create mesh
# 1x1 mesh represents a single device with two named dimensions (x and y)
mesh = Mesh(np.array(jax.devices()[0]).reshape(1, 1), ('x', 'y'))

# Device sharding - partitions data along x and y dimensions
s_dev = NamedSharding(mesh, P('x', 'y'), memory_kind="device")

# Host sharding - same partitioning but in pinned host memory
s_host = s_dev.with_memory_kind('pinned_host')

print(s_dev)   # Shows device memory sharding
print(s_host)  # Shows pinned host memory sharding
```

+++ {"id": "R_pB9465VoMP"}

### Data Placement with device_put

{func}`jax.device_put` is a function that explicitly transfers arrays to a specified memory location according to a sharding specification.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: OJFnf7FGp6Lj
outputId: c762e1df-2453-4ed9-9d53-0defb6a05ce2
---
# Create a 2x4 array
arr = jnp.arange(8.0).reshape(2, 4)

# Move arrays to different memory locations based on sharding objects
arr_host = jax.device_put(arr, s_host)  # Places in pinned host memory
arr_dev = jax.device_put(arr, s_dev)    # Places in device memory

# Verify memory locations
print(arr_host.sharding.memory_kind)  # Output: pinned_host
print(arr_dev.sharding.memory_kind)   # Output: device
```

+++ {"id": "HHXvBpQKTMCR"}

### Output Sharding Controls

Shardings determine how data is split across devices. JAX provides `out_shardings` to control how output arrays are partitioned when leaving a jitted function.

Key Features:
  - Can differ from input sharding
  - Allows different memory kinds for outputs

Examples:

#### Device Output Sharding

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ZXNj9NUeaIdX
outputId: 399321ef-082a-4a77-c33a-9de3421f429b
---
f = jax.jit(lambda x:x, out_shardings=s_dev)
out_dev = f(arr_host)
print("Result value of H2D: \n", out_dev)
```

+++ {"id": "iYXC5ix384XP"}

Moving data from host to device memory when needed for computation is the essence of host offloading. Use {func}`jax.device_put` to perform this transfer in this example to optimize performance.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: cmM6tJTS84XQ
outputId: 40c353a1-fb55-44bc-bac9-dffc09852f49
---
# Instead of the lambda function, you can define add_func to explicitly
# move data to device before computation
def add_func(x):  # Move data to device and add one
    x = jax.device_put(x, s_dev)
    return x + 1

f = jax.jit(add_func, out_shardings=s_dev)
out_dev = f(arr_host)
print("Result value of H2D and add 1 in device memory: \n", out_dev)
```

+++ {"id": "EbE-eBrJTBuS"}

#### Host Output Sharding

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: FjZzkxI8ky4r
outputId: 2a1b6e7a-1c29-4347-c020-7b47c27a5cc3
---
f = jax.jit(lambda x: x, out_shardings=s_dev)
out_host = f(arr_host)      # Input arrays in the device memory while output arrays in the host memory
print("Result value of D2H: \n", out_host)
```

+++ {"id": "UhLVvRO2p6Lj"}

## Activation Offloading

The detailed coverage of activation offloading can be found in the {ref}`gradient-checkpointing` tutorial. Activation offloading helps manage memory by moving intermediate activations to host memory after the forward pass, and bringing them back to device memory during the backward pass when needed for gradient computation.

To implement activation offloading effectively, you need to understand checkpoint names and policies. Here's how they work in a simple example:

### Checkpoint Names

The {func}`checkpoint_name` function allows you to label activations for memory management during computation. Here's a simple example:

```{code-cell} ipython3
:id: sLO9ceS6p6Lj

from jax.ad_checkpoint import checkpoint_name

def layer(x, w):
  w1, w2 = w
  x = checkpoint_name(x, "x")
  y = x @ w1
  return y @ w2, None
```

+++ {"id": "-_T92oCOp6Lk"}

This example shows:

* A simple neural network layer with two matrix multiplications
* Labeling of input activation x with identifier `"x"`
* Sequential operations:
  1. First multiplication: `x @ w1`
  2. Second multiplication: `y @ w2`

The checkpoint name helps the system decide whether to:
* Keep the activation in device memory or
* Offload it to host memory during computation

This pattern is common in neural networks, where multiple transformations are applied sequentially to input data.


### Checkpoint Policies

The {func}`jax.remat` transformation manages memory by handling intermediate values through three strategies:

1. Recomputing during backward pass (default behavior)
2. Storing on device
3. Offloading to host memory after forward pass and loading back during backward pass

Example of setting an offloading checkpoint policy:

```{code-cell} ipython3
:id: W8Usw_wOp6Lk

from jax import checkpoint_policies as cp

policy = cp.save_and_offload_only_these_names(
    names_which_can_be_saved=[],          # No values stored on device
    names_which_can_be_offloaded=["x"],   # Offload activations labeled "x"
    offload_src="device",                 # Move from device memory
    offload_dst="pinned_host"             # To pinned host memory
)
```

+++ {"id": "iuDRCXu7ky4r"}

Since {func}`jax.lax.scan` is commonly used in JAX for handling sequential operations (like RNNs or transformers), you need to know how to apply your offloading strategy in this context.

Key components:
* {func}`jax.remat` applies our checkpoint policy to the layer function
* `prevent_cse=False` enables XLA's common subexpression elimination for better performance
* {func}`jax.lax.scan` iterates the rematerialized layer along an axis

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: xCrxjTx_p6Lk
outputId: 13d46584-9b25-4622-b3c3-f50c1dac02c2
---
def scanned(w, x):
  remat_layer = jax.remat(layer,
                          policy=policy,     # Use our offloading policy
                          prevent_cse=False) # Allow CSE optimizations
  result = jax.lax.scan(remat_layer, x, w)[0]
  return jnp.sum(result)

# Initialize input and weights with small values (0.0001)
input = jnp.ones((256, 256), dtype=jnp.float32) * 0.001  # Input matrix: 256 x 256
w1 = jnp.ones((10, 256, 1024), dtype=jnp.float32) * 0.001 # 10 layers of 256 x 1024 matrices
w2 = jnp.ones((10, 1024, 256), dtype=jnp.float32) * 0.001 # 10 layers of 1024 x 256 matrices

# Compile and compute gradients of the scanned function
f = jax.jit(jax.grad(scanned))  # Apply JIT compilation to gradient computation
result_activation = f((w1, w2), input)     # Execute the function with weights and input
print("Sample of results: ", result_activation[0][0, 0, :5])
```

+++ {"id": "0tx7aara42pY"}

### Summary of Activation Offloading

Activation offloading provides a powerful way to manage memory in large computations by:

* Using checkpoint names to mark specific activations
* Applying policies to control where and how activations are stored
* Supporting common JAX patterns like scan operations
* Moving selected activations to host memory when device memory is under budget

This approach is particularly useful when working with large models that would otherwise exceed device memory capacity.

## Parameter Offloading

Model parameters (also known as weights) can be offloaded to the host memory to optimize device memory usage during initialization. This is achieved by using {func}`jax.jit` with a sharding strategy that specifies host memory kind.

While parameter offloading and activation offloading are distinct memory optimization techniques, the following example demonstrates parameter offloading built upon the activation offloading implementation shown earlier.

### Parameter Placement for Computation

Different from the earlier `layer` function, {func}`jax.device_put` is applied to move parameter `w1` and `w2` to the device before the  matrix multiplications. This ensures the parameters are available on the device for both forward and backward passes.

Note that the activation offloading implementation remains unchanged, using the same:
* Checkpoint name `"x"`
* Checkpoint policy
* `scanned` function combining {func}`jax.remat` and {func}`jax.lax.scan`

### Parameter Initialization with Host Offloading

During the initialization, parameter `w1` and `w2` are placed on host memory before being passed to the {func}`jax.jit` function `f`, while keeping the `input` variable on the device.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 1qGN2hBQdheo
outputId: 48c09658-f8b6-4be3-ef0e-02e0e2566e10
---
# Hybrid version: Both activation and parameter offloading
def hybrid_layer(x, w):
  # Move model parameters w1 and w2 to host memory via device_put
  w1, w2 = jax.tree.map(lambda x: jax.device_put(x, s_dev), w)
  x = checkpoint_name(x, "x")  # Offload activation x to host memory
  y = x @ w1
  return y @ w2, None

def hybrid_scanned(w, x):
  remat_layer = jax.remat(hybrid_layer,     # Use hybrid_layer instead of layer
                          policy=policy,     # Use offloading policy
                          prevent_cse=False) # Allow CSE optimizations
  result = jax.lax.scan(remat_layer, x, w)[0]
  return jnp.sum(result)

# Move model parameters w1 and w2 to the host via device_put
# Initialize input and weights with small values (0.0001)
wh1 = jax.device_put(w1, s_host)
wh2 = jax.device_put(w2, s_host)

# Compile and compute gradients of the scanned function
f = jax.jit(jax.grad(hybrid_scanned))  # Apply JIT compilation to gradient computation
result_both = f((wh1, wh2), input) # Execute with both activation and parameter offloading

# Verify numerical correctness
are_close = jnp.allclose(
    result_activation[0],    # Result from activation offloading only
    result_both[0],         # Result from both activation and parameter offloading
    rtol=1e-5,
    atol=1e-5
)
print(f"Results match within tolerance: {are_close}")
```

+++ {"id": "SVpozzwHflQk"}

The matching results verify that initializing parameters on host memory maintains computational correctness.

### Limitation of Parameter Offloading

{func}`jax.lax.scan` is crucial for effective parameter management. Using an explicit for loop would cause parameters to continuously occupy device memory, resulting in the same memory usage as without parameter offloading. While {func}`jax.lax.scan` allows specifying the scan axis, parameter offloading currently works only when scanning over axis 0. Scanning over other axes generates a `transpose` operation during compilation before returning parameters to the device, which is expensive and not supported on all platforms.

## Tools for Host Offloading

For device memory analysis, refer to :doc:`device_memory_profiling`. The profiling tools described in {ref}`profiling` can help measure memory savings and performance impact from host offloading.
