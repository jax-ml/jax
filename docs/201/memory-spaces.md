(jax-201-memory-spaces)=
# Memory spaces and host offloading

<!--* freshness: { reviewed: '2026-08-27' } *-->

An accelerator's device memory is fast but small; the host's memory is slow
but large. JAX lets an array live in either one: every sharding carries a
**memory kind**, either `"device"` (the default) or `"pinned_host"`, and
moving data between the two spaces is just {func}`jax.device_put` with a
sharding of the other kind. *Host offloading* uses this to trade transfer
time for device-memory capacity: park model parameters, activations, or
optimizer state in host memory, and bring them back to the device only when
computation needs them.

```{note}
The examples on this page are shown with outputs from an accelerator
platform; memory-kind support varies by platform, so these snippets are
illustrative rather than executed in place. Offloading can also cost real
performance in host-device transfers — measure before committing to it.
```

## Building blocks

A sharding's `memory_kind` says which space its arrays live in, and
`with_memory_kind` derives a sharding in the other space:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices()[:1], 'x')
s_dev = NamedSharding(mesh, P('x'), memory_kind="device")
s_host = s_dev.with_memory_kind('pinned_host')
print(s_dev)   # NamedSharding(..., memory_kind=device)
print(s_host)  # NamedSharding(..., memory_kind=pinned_host)
```

{func}`jax.device_put` places (or moves) an array into the space a sharding
names:

```python
arr = jnp.arange(8.0).reshape(2, 4)
arr_host = jax.device_put(arr, s_host)
arr_dev = jax.device_put(arr, s_dev)
print(arr_host.sharding.memory_kind)  # pinned_host
print(arr_dev.sharding.memory_kind)   # device
```

Compiled functions can consume and produce arrays in either space: `jit`'s
`out_shardings` accepts memory kinds, so a jitted function can return its
outputs directly to host memory (device-to-host) or take host-resident
inputs and produce device outputs (host-to-device):

```python
f = jax.jit(lambda x: x, out_shardings=s_host)
out_host = f(arr_dev)   # inputs on device, outputs land in host memory

g = jax.jit(lambda x: x + 1, out_shardings=s_dev)
out_dev = g(arr_host)   # host-resident input, device-resident result
```

Inside a jitted function, {func}`jax.device_put` moves values between
spaces mid-computation — this is the workhorse for the offloading patterns
below.

## Offloading activations

Residuals saved for the backward pass are often the dominant memory cost of
training. Rather than keeping them in device memory or recomputing them,
autodiff can *offload* them: move named residuals to host memory after the
forward pass and fetch them back for the backward pass, using
{func}`jax.checkpoint` with the
`jax.checkpoint_policies.save_and_offload_only_these_names` policy. That
machinery belongs to rematerialization, and is covered with the rest of the
remat story in {ref}`jax-301-remat-offload`. As a taste of the effect, for
the 10-layer scanned MLP used as a running example below, offloading the
layer activations cuts temporary memory from 17.25 MB to 6.50 MB.

## Offloading parameters

Model parameters can live in host memory, with each layer fetching its
weights to the device just in time. The pattern: initialize (or load)
parameters into host memory, and apply {func}`jax.device_put` inside the
layer function before the weights are used:

```python
from jax.ad_checkpoint import checkpoint_name
from jax import checkpoint_policies as cp

policy = cp.save_and_offload_only_these_names(
    names_which_can_be_saved=[],
    names_which_can_be_offloaded=["x"],
    offload_src="device",
    offload_dst="pinned_host",
)

def hybrid_layer(x, w):
  # Move this layer's parameters to device memory just in time.
  w1, w2 = jax.tree.map(lambda w: jax.device_put(w, s_dev), w)
  x = checkpoint_name(x, "x")   # offload this activation (see the remat docs)
  y = x @ w1
  return y @ w2, None

def hybrid_scanned(w, x):
  remat_layer = jax.remat(hybrid_layer, policy=policy, prevent_cse=False)
  result = jax.lax.scan(remat_layer, x, w)[0]
  return jnp.sum(result)

input = jnp.ones((256, 256), dtype=jnp.float32) * 0.001
w1 = jnp.ones((10, 256, 1024), dtype=jnp.float32) * 0.001
w2 = jnp.ones((10, 1024, 256), dtype=jnp.float32) * 0.001

# Parameters live in host memory...
wh1 = jax.device_put(w1, s_host)
wh2 = jax.device_put(w2, s_host)

# ...and the input stays on the device.
f = jax.jit(jax.grad(hybrid_scanned))
result = f((wh1, wh2), input)
```

For this example, {func}`jax.stages.Compiled.memory_analysis` reports (on
TPU):

```text
Temp size: 4.75 MB
Argument size: 0.25 MB
Total size: 25.00 MB
```

against a no-offloading baseline of 17.25 MB temporary and 20.25 MB argument
memory. Three effects combine:

1. **Parameter offloading** removes the weights from device argument memory
   (20.25 MB → 0.25 MB: only the input remains).
2. **Activation offloading** cuts temporary memory (17.25 MB → 6.50 MB).
3. **Their interaction** saves a bit more (6.50 MB → 4.75 MB):
   rematerialization keeps JAX from holding on-device copies of the weights
   alive for the backward pass.

Two limitations to know about. {func}`jax.lax.scan` is load-bearing in this
pattern: with an explicit Python loop, the parameters would continuously
occupy device memory, giving no saving. And parameter offloading currently
works only when scanning over axis 0 — other axes insert an expensive
`transpose` when returning parameters to the device, and aren't supported
on all platforms.

## Offloading optimizer state

Optimizer state (like Adam's moments) is device memory spent on values used
only briefly in each step. The same pattern applies: keep the state in host
memory between steps, move it to the device inside the step, and send the
updated state back to host memory via `out_shardings`:

```python
import optax

s_dev = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="device")
s_host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=0.1))

# (network and loss definitions elided)
def step(params, opt_state, inputs):
  grads = jax.grad(lambda p: compute_loss(p, inputs))(params)
  opt_state = jax.device_put(opt_state, s_dev)   # fetch state to the device
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state

params = init_params()                          # on device
opt_state = optimizer.init(params)
opt_state = jax.device_put(opt_state, s_host)   # state lives on the host

step = jax.jit(
    step,
    donate_argnums=(0,),
    out_shardings=(s_dev, s_host),   # params to device, state back to host
)
new_params, new_opt_state = step(params, opt_state, input)
```

For a four-layer 7168×7168 MLP with Adam, memory analysis reports 4.59 GB
total without offloading and 2.87 GB with it — a 1.72 GB saving, almost all
from the optimizer state leaving device argument memory. Note the
trade-off's structure: offloading can *add* temporary memory (updated state
needs device buffers before it's copied out to the host, and XLA's
latency-hiding scheduling extends buffer live ranges to overlap transfers
with compute), but the argument-memory saving typically dominates.

## Measuring

{func}`jax.stages.Compiled.memory_analysis`, used throughout this page,
reports a compiled function's memory breakdown before you run it: sum the
temporary, argument, and output sizes, minus the alias size, for the total.
For runtime measurement — including verifying that transfers overlap with
compute — see the device memory profiling and tracing tools in
{doc}`profiling`.
