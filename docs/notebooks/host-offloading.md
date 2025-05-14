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

+++ {"id": "bQbS50fIdHw1"}

(host-offloading)=
# JAX Memories and Host Offloading

<!--* freshness: { reviewed: '2025-04-10' } *-->

This tutorial provides a practical introduction to host offloading techniques in JAX, focusing on:

- Activation offloading
- Parameter offloading
- Optimizer state offloading

By applying offloading strategies, developers can better manage memory resources and reduce memory pressure on devices. To implement these strategies effectively, understanding JAX's core mechanisms for data placement and movement is essential.

## Building Blocks for Offloading

JAX provides several key components for controlling where and how data are stored and moved between the host and the device memory. The following sections explore:

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
# Instead of the lambda function, add_func can be defined explicitly
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
out_host = f(arr_host)      # Input arrays in hte device memory while output arrays in the host memory
print("Result value of D2H: \n", out_host)
```

+++ {"id": "UhLVvRO2p6Lj"}

## Activation Offloading

Before diving into activation offloading, let's first take a look at the baseline code.

This code implements a simple neural network with 10 layers, each consisting of two linear transformations. The code demonstrates basic memory usage patterns and provides a foundation for comparing offloading optimization techniques.

Key components:
- Each layer consists of two sequential linear operations:
  1. First multiplication: `x @ w1`
  2. Second multiplication: `y @ w2`
- 10-layer network using JAX's scan operation
- Memory usage analysis
- Gradient computation with JIT compilation

To analyze memory usage in JAX, the :func:`jax.stages.Compiled.memory_analysis` method can be used on a compiled function. This provides detailed statistics about memory consumption during computation. The key metrics include temporary memory size, argument size, output size, and alias size. To calculate the total memory usage, sum the temporary, argument, and output sizes, then subtract the alias size to avoid double-counting the same memory multiple times. This provides a summarized view of how the device memory is utilized across different aspects of the computation.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: UEt0dtxukkaz
outputId: 22bb32b7-8491-4100-f212-e56c50f44cfa
---
# Initialize input and weights with small values (0.0001)
input = jnp.ones((256, 256), dtype=jnp.float32) * 0.001  # Input matrix: 256 x 256
w1 = jnp.ones((10, 256, 1024), dtype=jnp.float32) * 0.001 # 10 layers of 256 x 1024 matrices
w2 = jnp.ones((10, 1024, 256), dtype=jnp.float32) * 0.001 # 10 layers of 1024 x 256 matrices

def two_layers(x, w):
  # Simple two-layer linear transformation
  w1, w2 = w
  y = x @ w1
  return y @ w2, None

def scanned(w, x):
  # Applies the layer function 10 times using JAX's scan operation
  # Input: w (tuple of weight matrices), x (input matrix)
  # Output: sum of the final layer's output
  result = jax.lax.scan(two_layers, x, w)[0]
  return jnp.sum(result)

# Compile and compute gradients of the scanned function
f = jax.jit(jax.grad(scanned))  # Apply JIT compilation to gradient computation

# Analyze memory usage
compiled_step = f.lower((w1, w2), input).compile()
compiled_stats = compiled_step.memory_analysis()

if compiled_stats is not None:
  # Calculate total memory usage including temporary storage, arguments, and outputs
  # Subtract alias size to avoid double-counting memory shared between different components
  total = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
      + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
  print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB")
  print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB")
  print(f"Total size: {total/(1024**2):.2f} MB")

# Execute the function and print sample results
result = f((w1, w2), input)     # Execute the function with weights and input
print("Sample of results: ", result[0][0, 0, :5])
```

+++ {"id": "DnFyRt2nkkaz"}

The detailed coverage of activation offloading can be found in the {ref}`gradient-checkpointing` tutorial. Activation offloading helps manage memory by moving intermediate activations to host memory after the forward pass, and bringing them back to device memory during the backward pass when needed for gradient computation.

To implement activation offloading effectively, it is important to understand checkpoint names and policies. Here's how they work in a simple example:

### Checkpoint Names

The {func}`checkpoint_name` function allows labeling activations for memory management during computation. Here's a simple example that a checkpoint name `x` is specified.

```{code-cell} ipython3
:id: sLO9ceS6p6Lj

from jax.ad_checkpoint import checkpoint_name

def layer_name(x, w):
  w1, w2 = w
  x = checkpoint_name(x, "x")
  y = x @ w1
  return y @ w2, None
```

+++ {"id": "-_T92oCOp6Lk"}

The checkpoint name helps the system decide whether to:
* Keep the activation in device memory or
* Offload it to host memory during computation

This pattern is common in neural networks, where multiple transformations are applied sequentially to input data.

### Checkpoint Policies

This checkpoint policy implements a memory management strategy that optimizes memory usage during computation. It manages memory by handling intermediate values through three strategies:
1. Recomputing during backward pass (default behavior)
2. Storing on device
3. Offloading to host memory after forward pass and loading back during backward pass

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

{func}`jax.lax.scan` is commonly used in JAX for handling sequential operations (like RNNs or transformers). It can be integrated with JAX's rematerialization to process sequential data.

Key components:
* {func}`jax.remat` creates a rematerialized version of the layer function using {func}`jax.remat` and applies the checkpoint policy to the layer function
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
  remat_layer = jax.remat(layer_name,
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

# Analyze memory usage
compiled_step = f.lower((w1, w2), input).compile()
compiled_stats = compiled_step.memory_analysis()

if compiled_stats is not None:
  total = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
      + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
  print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB")
  print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB")
  print(f"Total size: {total/(1024**2):.2f} MB")

result_activation = f((w1, w2), input)     # Execute the function with weights and input
# Verify numerical correctness
are_close = jnp.allclose(
    result_activation[0],    # Result from activation offloading only
    result[0],         # Result from both activation and parameter offloading
    rtol=1e-5,
    atol=1e-5
)
print(f"Results match within tolerance: {are_close}")
print("Sample of results: ", result_activation[0][0, 0, :5])
```

+++ {"id": "0tx7aara42pY"}

Activation offloading reduces temporary memory usage from 17.25 MB to 6.5 MB while input and output argument sizes remain the same. Totally 10.75 MB is saved. It is achieved by offloading activation `x` to host memory after the forward pass and loading it back to device memory before the backward pass.

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

# Analyze memory usage
compiled_step = f.lower((wh1, wh2), input).compile()
compiled_stats = compiled_step.memory_analysis()

if compiled_stats is not None:
  total = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
      + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
  print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB")
  print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB")
  print(f"Total size: {total / (1024**2):.2f} MB")

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

This implementation demonstrates how offloading model parameters together with activation offloading to host memory can significantly reduce device memory usage.

### Memory Analysis

**Baseline Memory Usage:**
- Input tensor: 0.25 MB (256 × 256 × 4 bytes)
- Model parameters (w1, w2): 10 MB each (256 × 1024 × 4 bytes ≈ 1 MB per layer × 10 layers)

**Memory Usage Comparison:**
- Argument size without parameter offloading: 20.25 MB (0.25 + 10 + 10)
- Argument size with parameter offloading: 0.25 MB (only input remains)
- Temporary memory without activation offloading: 17.25 MB
- Temporary memory with activation offloading: 6.50 MB
- Temporary memory with activation and parameter offloading: 4.75 MB

#### Key Optimizations

1. **Parameter Offloading**: Moving parameters (w1, w2) to host memory reduces argument size by 20 MB (from 20.25 MB to 0.25 MB).

2. **Activation Offloading**: Moving activations to host memory reduces temporary memory usage by 10.75 MB (from 17.25 to 6.50 MB).

3. **Hybrid Strategy**: The rematerialization of activation offloading helps avoid keeping weights on the device and reduce temporary memory usage by 1.75 MB (from 6.50 MB to 4.75 MB). Without it, JAX would be eager to keep the on-device copies of the weights alive for the backward pass.

#### Results

**Total Memory Savings**: 33.5 MB (20 MB + 10.75 MB + 1.75 MB)

This hybrid approach demonstrates that parameter and activation offloading work synergistically to achieve significant memory reductions while maintaining computational correctness.  

### Limitations of Parameter Offloading

{func}`jax.lax.scan` is crucial for effective parameter management. Using an explicit for loop would cause parameters to continuously occupy device memory, resulting in the same memory usage as without parameter offloading. While {func}`jax.lax.scan` allows specifying the scan axis, parameter offloading currently works only when scanning over axis 0. Scanning over other axes generates a `transpose` operation during compilation before returning parameters to the device, which is expensive and not supported on all platforms.

The offloading performance can vary for different device types. It may degrade performance due to memory transfers between host and device, so it's important to consider this trade-off when designing your optimization strategy.

# Optimizer State Offloading

Optimizer state offloading is a memory management technique that stores optimizer states in host memory instead of device memory. This approach is particularly useful when optimizer states are large, as it reduces device memory usage.

A basic JAX implementation using the Adam optimizer can serve as a starting point, where all tensors are stored on the device. This will serve as a reference implementation before introducing optimizer state offloading.

### Basic Implementation

This section, let's implement a simple model with the Adam optimizer. This implementation helps establish the baseline behavior before exploring optimizer state offloading. It is particularly useful for understanding memory patterns in large-scale neural network training.

In the code example below, a neural network training loop is included to use JAX and Optax's Adam optimizer. The network consists of four linear layers with GELU activation functions, processing large matrices of size 7168x7168. The training process involves:
- Forward pass: The input flows through four layers, each applying a linear transformation followed by GELU activation
- Loss computation: Calculates mean squared error between output and input, plus L2 regularization
- Backward pass: Computes gradients using automatic differentiation
- Optimization step: Updates parameters using Adam optimizer with gradient clipping

The code uses JIT compilation to optimize performance and includes memory usage analysis to monitor the computational resources required during training. The memory analysis provides insights into temporary memory usage, argument sizes, and total memory consumption during the optimization step.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: ujvC0YJ2VOyV
outputId: d237ca0a-89ae-4e14-edd3-36cc38890349
---
import optax

DIM = 7168

# Initialize data and parameter w1, w2, w3 and w4
input = jnp.ones((DIM, DIM))
params = {f'w{i}': jnp.ones((DIM, DIM)) for i in range(1, 5)}

# Initialize optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.1)
)
opt_state = optimizer.init(params)

def gelu(x):
  return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

def single_layer(x, w):
  return x @ w

def forward(params, x):
  for i in range(1, 5):
    x = gelu(single_layer(x, params[f'w{i}']))
  return x

def compute_loss(params, inputs):
  outputs = forward(params, inputs)
  loss = jnp.mean((outputs - inputs) ** 2)
  l2_reg = 0.001 * sum(jnp.sum(w ** 2) for w in jax.tree_util.tree_leaves(params))
  return loss + l2_reg

def step(params, opt_state, inputs):
  grads = jax.grad(lambda p: compute_loss(p, inputs))(params)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  return optax.apply_updates(params, updates), new_opt_state

# JIT compile the step function with proper sharding
step = jax.jit(step, donate_argnums=(0, 1))

# Run a optimization step
new_params, new_opt_state = step(params, opt_state, input)

# Analyze memory usage
compiled_step = step.lower(params, opt_state, input).compile()
compiled_stats = compiled_step.memory_analysis()

if compiled_stats is not None:
  total = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
      + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
  print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**3):.2f} GB")
  print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**3):.2f} GB")
  print(f"Total size: {total / (1024**3):.2f} GB")
```

+++ {"id": "oW4Qm6E5VOyV"}

Optimizer state offloading can be implemented as follows.

### Setting Up Sharding and Memory Kinds

{func}`jax.sharding.SingleDeivceSharding` is adopted to simplify the shardings for both device and host memory kinds. During the model state initialization, move the optimizer state to the host using {func}`device_put`.

### Model and Training Step Implementation

Next, define the model architecture, loss function, and training step. The key addition here is moving the optimizer state to device memory via {func}`device_put` at the beginning of each training step, as it's needed for the parameter update on the device.

### Running and Comparing Results

After setting up the sharding, the optimizer state is moved to host memory and the step function is run with {func}`jax.jit`.

The JIT compilation of the step function uses several important parameters:
- `donate_argnums=(0,)`: Indicates that the first argument (parameters) can be modified in-place, allowing JAX to reuse its memory
- `out_shardings`: Specifies how output tensors should be sharded across the mesh (devices and hosts)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: fEDTasJZVOyW
outputId: b36cedd6-cf30-4d36-f4fd-32b2fdfd7564
---
# Create sharding specifications for device and host memory
s_dev = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="device")
s_host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")

def step(params, opt_state, inputs):
  grads = jax.grad(lambda p: compute_loss(p, inputs))(params)
  opt_state = jax.device_put(opt_state, s_dev)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state

params = {f'w{i}': jnp.ones((DIM, DIM)) for i in range(1, 5)}
opt_state = optimizer.init(params)

# Initialize optimizer
optimizer = optax.chain(
  optax.clip_by_global_norm(1.0),
  optax.adam(learning_rate=0.1)
)

# Optimizer state is placed on the host during initialization
opt_state = jax.device_put(opt_state, s_host)

# JIT compile the step function with proper sharding and memory optimization
step = jax.jit(
  step,
  donate_argnums=(0,),
  out_shardings=(s_dev, s_host)
)

# Run an optimization step
new_params, offload_opt_state = step(params, opt_state, input)

# Analyze memory usage
compiled_step = step.lower(params, opt_state, input).compile()
compiled_stats = compiled_step.memory_analysis()
if compiled_stats is not None:
  total = compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes \
      + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes
  print(f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**3):.2f} GB")
  print(f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**3):.2f} MB")
  print(f"Total size: {total / (1024**3):.2f} GB")
```

+++ {"id": "vKo8qYnQVOyW"}

This implementation demonstrates how to:
1. Set up sharding specifications for `device` and `pinned_host`
2. Move optimizer states between host and device memory via {func}`jax.device_put`
3. Use `out_shardings` to ensure proper memory placement
4. Show the memory usage

This implementation demonstrates how offloading optimizer state to host memory can reduce device memory usage through a trade-off between argument size and temporary memory.

Memory Analysis:
1. Argument Size Reduction:
   - The optimizer states are arguments of the {func}`jax.jit` function
   - By offloading these states to host memory, the argument size on device is reduced

2. Temporary Memory Impact:
   - Offloading increases temporary memory usage
   - This is because outputs of optimizer states need memory buffers before being copied to host
   - The memory live ranges for these temporary buffers are extended due to the host-device transfers

3. Latency Hiding Scheduling:
   - JAX uses XLA's latency hiding scheduling to overlap computation with host-device transfers
   - The overlapping can cause tensors to have larger live ranges, which increases memory pressure on the device
   - This adaptive behavior helps maintain stable memory usage while still providing some performance benefits

4. Memory Trade-off:
   - Total memory size with offloading: 2.87 GB
   - Total memory size without offloading: 4.59 GB
   - Net memory saving: 1.72 GB

while offloading increases temporary memory usage, the reduction in argument size more than compensates for this increase, resulting in an overall reduction in device memory usage. 

Note: The optimizer states can be compared for numerical equivalence using `jax.tree_util.tree_map` and `jnp.allclose`, but this verification step is omitted here for brevity.

## Tools for Host Offloading

:func:`jax.stages.Compiled.memory_analysis` API is utilized above to get memory usage information. For device memory analysis, refer to :doc:`device_memory_profiling`. The profiling tools described in {ref}`profiling` can help measure memory savings and performance impact from host offloading.
