---
jupytext:
  formats: ipynb,py,md:myst
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

(ffi-tutorial)=

# Foreign function interface (FFI)

_This tutorial requires JAX v0.4.31 or newer._

While a wide range of numerical operations can be easily and efficiently implemented using JAX's built in `jax.numpy` and `jax.lax` interfaces, it can sometimes be useful to explicitly call out to external compiled libraries via a "foreign function interface" (FFI).
This can be particularly useful when particular operations have been previously implemented in an optimized C or CUDA library, and it would be non-trivial to reimplement these computations directly using JAX, but it can also be useful for optimizing runtime or memory performance of JAX programs.
That being said, the FFI should typically be considered a last resort option because the XLA compiler that sits in the backend, or the Pallas kernel language, which provides lower level control, typically produce performant code with a lower development and maintenance cost.

One point that should be taken into account when considering use of the FFI is that _JAX doesn't automatically know how to differentiate through foreign functions_.
This means that if you want to use JAX's autodifferentiation capabilities alongside a foreign function, you'll also need to provide an implementation of the relevant differentiation rules.
We will discuss some possible approaches below, but it is important to call this limitation out right from the start!

JAX's FFI support is provided in two parts:

1. A header-only C++ library from XLA which is packaged as part of JAX as of v0.4.29 or available from the [openxla/xla](https://github.com/openxla/xla) project, and
2. A Python front end, available in the `jax.ffi` submodule.

In this tutorial we demonstrate the use of both of these components using a simple example, and then go on to discuss some lower-level extensions for more complicated use cases.
We start by presenting the FFI on CPU, and discuss generalizations to GPU or multi-device environments below.

The end-to-end code for this example and some other more advanced use cases can be found in the JAX FFI examples project on GitHub at [`examples/ffi` in the JAX repository](https://github.com/jax-ml/jax/tree/main/examples/ffi).

Because we demonstrate how FFI calls can be sharded at the end of this tutorial, let's first set up our environment to be treated by JAX as having multiple CPUs:

```{code-cell} ipython3
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
```

## A simple example

To demonstrate the use of the FFI interface, we will implement a simple "root-mean-square (RMS)" normalization function.
RMS normalization takes an array $x$ with shape $(N,)$ and returns

$$
y_n = \frac{x_n}{\sqrt{\frac{1}{N}\sum_{n=1}^N {x_n}^2 + \epsilon}}
$$

where $\epsilon$ is a tuning parameter used for numerical stability.

This is a somewhat silly example, because it can be easily implemented using JAX as follows:

```{code-cell} ipython3
import jax
import jax.numpy as jnp


def rms_norm_ref(x, eps=1e-5):
  scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
  return x / scale
```

But, it's just non-trivial enough to be useful for demonstrating some key details of the FFI, while still being straightforward to understand.
We will use this reference implementation to test our FFI version below.

## Backend code

To begin with, we need an implementation of RMS normalization in C++ that we will expose using the FFI.
This isn't meant to be particularly performant, but you could imagine that if you had some new better implementation of RMS normalization in a C++ library, it might have an interface like the following.
So, here's a simple implementation of RMS normalization in C++:

```c++
#include <cmath>
#include <cstdint>

float ComputeRmsNorm(float eps, int64_t size, const float *x, float *y) {
  float sm = 0.0f;
  for (int64_t n = 0; n < size; ++n) {
    sm += x[n] * x[n];
  }
  float scale = 1.0f / std::sqrt(sm / float(size) + eps);
  for (int64_t n = 0; n < size; ++n) {
    y[n] = x[n] * scale;
  }
  return scale;
}
```

and, for our example, this is the function that we want to expose to JAX via the FFI.

+++

### C++ interface

To expose our library function to JAX and XLA, we need to write a thin wrapper using the APIs provided by the header-only library in the [`xla/ffi/api`](https://github.com/openxla/xla/tree/main/xla/ffi/api) directory of the [XLA project](https://github.com/openxla/xla).
For more information about this interface, take a look at [the XLA custom call documentation](https://openxla.org/xla/custom_call).
The full source listing can be downloaded [here](https://github.com/jax-ml/jax/blob/main/examples/ffi/src/jax_ffi_example/rms_norm.cc), but the key implementation details are reproduced here:

```c++
#include <functional>
#include <numeric>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function `ComputeRmsNorm` above. This function handles the batch
// dimensions by calling `ComputeRmsNorm` within a loop.
ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("RmsNorm input must be an array");
  }
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeRmsNorm(eps, lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
  }
  return ffi::Error::Success();
}

// Wrap `RmsNormImpl` and specify the interface to XLA. If you need to declare
// this handler in a header, you can use the `XLA_FFI_DECLARE_HANDLER_SYMBOL`
// macro: `XLA_FFI_DECLARE_HANDLER_SYMBOL(RmsNorm)`.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNorm, RmsNormImpl,
    ffi::Ffi::Bind()
        .Attr<float>("eps")
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);
```

Starting at the bottom, we're using the XLA-provided macro `XLA_FFI_DEFINE_HANDLER_SYMBOL` to generate some boilerplate which will expand into a function called `RmsNorm` with the appropriate signature.
But, the important stuff here is all in the call to `ffi::Ffi::Bind()`, where we define the input and output types, and the types of any parameters.

Then, in `RmsNormImpl`, we accept `ffi::Buffer` arguments which include information about the buffer shape, and pointers to the underlying data.
In this implementation, we treat all leading dimensions of the buffer as batch dimensions, and perform RMS normalization over the last axis.
`GetDims` is a helper function providing support for this batching behavior.
We discuss this batching behavior in more detail [below](ffi-call-vmap), but the general idea is that it can be useful to transparently handle batching in the left-most dimensions of the input arguments.
In this case, we treat all but the last axis as batch dimensions, but other foreign functions may require a different number of non-batch dimensions.

+++

### Building and registering an FFI handler

Now that we have our minimal FFI wrapper implemented, we need to expose this function (`RmsNorm`) to Python.
In this tutorial, we compile `RmsNorm` into a shared library and load it using [ctypes](https://docs.python.org/3/library/ctypes.html), but another common pattern is to use [nanobind](https://nanobind.readthedocs.io/) or [pybind11](https://pybind11.readthedocs.io/) as discussed below.

To compile the shared library, we're using CMake here, but you should be able to use your favorite build system without too much trouble.

```{code-cell} ipython3
:tags: [hide-output]

!cmake -DCMAKE_BUILD_TYPE=Release -B ffi/_build ffi
!cmake --build ffi/_build
!cmake --install ffi/_build
```

With this compiled library in hand, we now need to register this handler with XLA via the {func}`~jax.ffi.register_ffi_target` function.
This function expects our handler (a function pointer to the C++ function `RmsNorm`) to be wrapped in a [`PyCapsule`](https://docs.python.org/3/c-api/capsule.html).
JAX provides a helper function {func}`~jax.ffi.pycapsule` to help with this:

```{code-cell} ipython3
import ctypes
from pathlib import Path

path = next(Path("ffi").glob("librms_norm*"))
rms_norm_lib = ctypes.cdll.LoadLibrary(path)
jax.ffi.register_ffi_target(
    "rms_norm", jax.ffi.pycapsule(rms_norm_lib.RmsNorm), platform="cpu")
```

```{tip}
If you're familiar with the legacy "custom call" API, it's worth noting that you can also use {func}`~jax.ffi.register_ffi_target` to register a custom call target by manually specifying the keyword argument `api_version=0`. The default `api_version` for {func}`~jax.ffi.register_ffi_target` is `1`, the new "typed" FFI API that we're using here.
```

**An alternative approach**:
A common alternative pattern for exposing handlers to Python is to use [nanobind](https://nanobind.readthedocs.io/) or [pybind11](https://pybind11.readthedocs.io/) to define a tiny Python extension which can be imported.
For our example here, the nanobind code would be:

```c++
#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(rms_norm, m) {
  m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
}
```

Then, in Python we can register this handler using:

```python
# Assuming that we compiled a nanobind extension called `rms_norm`:
import rms_norm as rms_norm_lib

jax.ffi.register_ffi_target("rms_norm", rms_norm_lib.rms_norm(), platform="cpu")
```

+++

## Frontend code

Now that we have registered our FFI handler, it is straightforward to call our C++ library from JAX using the {func}`~jax.ffi.ffi_call` function:

```{code-cell} ipython3
import numpy as np


def rms_norm(x, eps=1e-5):
  # We only implemented the `float32` version of this function, so we start by
  # checking the dtype. This check isn't strictly necessary because type
  # checking is also performed by the FFI when decoding input and output
  # buffers, but it can be useful to check types in Python to raise more
  # informative errors.
  if x.dtype != jnp.float32:
    raise ValueError("Only the float32 dtype is implemented by rms_norm")

  call = jax.ffi.ffi_call(
    # The target name must be the same string as we used to register the target
    # above with `jax.ffi.register_ffi_target`
    "rms_norm",

    # In this case, the output of our FFI function is just a single array with
    # the same shape and dtype as the input. We discuss a case with a more
    # interesting output type below.
    jax.ShapeDtypeStruct.like(x),
  )

  # Note that here we're use `numpy` (not `jax.numpy`) to specify a dtype for
  # the attribute `eps`. Our FFI function expects this to have the C++ `float`
  # type (which corresponds to numpy's `float32` type), and it must be a
  # static parameter (i.e. not a JAX array).
  return call(x, eps=np.float32(eps))


# Test that this gives the same result as our reference implementation
x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
np.testing.assert_allclose(rms_norm(x), rms_norm_ref(x), rtol=1e-5)
```

This code cell includes a lot of inline comments which should explain most of what is happening here, but there are a few points that are worth explicitly highlighting.
Most of the heavy lifting here is done by the {func}`~jax.ffi.ffi_call` function, which tells JAX how to call the foreign function for a particular set of inputs.
It's important to note that the first argument to {func}`~jax.ffi.ffi_call` must be a string that matches the target name that we used when calling {func}`~jax.ffi.register_ffi_target` above.

Any attributes (defined using `Attr` in the C++ wrapper above) should be passed as keyword arguments to {func}`~jax.ffi.ffi_call`.
Note that we explicitly cast `eps` to `np.float32` because our FFI library expects a C `float`, and we can't use `jax.numpy` here, because these parameters must be static arguments.

+++

(ffi-call-vmap)=

## Supporting transformations like `vmap` and `grad`

The `rms_norm` function above works whenever we evaluate it directly, but as written it doesn't yet support JAX's function transformations.
As far as JAX is concerned, an {func}`~jax.ffi.ffi_call` is an opaque black box: JAX can't look inside it to work out how it should behave under {func}`~jax.vmap`, or how to differentiate it.
So, for example, trying to differentiate `rms_norm` as defined above would fail.

To teach JAX how to transform our foreign function, we wrap it in a _HiJAX primitive_: a custom JAX operation defined by subclassing `VJPHiPrimitive` from the experimental `jax.experimental.hijax` module.
On the primitive, we define a handful of methods:

* `expand` implements the operation in terms of other ("lojax") JAX operations. Here, that's just a call to {func}`~jax.ffi.ffi_call`, and it is what runs when the primitive isn't being transformed.
* `vjp_fwd` and `vjp_bwd_retval` together define the reverse-mode automatic differentiation rule (used by {func}`~jax.grad`, {func}`~jax.vjp`, and friends).
* `batch` defines the {func}`~jax.vmap` rule.

```{note}
HiJAX is a new and experimental API. The details of the interface may change in future releases of JAX.
```

To support differentiation, we implement two additional FFI targets for the forward and backward passes:

1. `rms_norm_fwd` returns two outputs: (a) the "primal" result, and (b) the "residuals" which are saved for use on the backwards pass, and
2. `rms_norm_bwd` takes the residuals and the output co-tangents, and returns the input co-tangents.

We won't get into the details of the RMS normalization backwards pass, but take a look at the [C++ source code](https://github.com/jax-ml/jax/blob/main/examples/ffi/src/jax_ffi_example/rms_norm.cc) to see how these functions are implemented on the back end.
The main point to emphasize here is that the "residual" computed by `rms_norm_fwd` has a different shape than the primal output, so in the {func}`~jax.ffi.ffi_call` to `rms_norm_fwd`, the output type has two elements with different shapes.

```{code-cell} ipython3
from jax.experimental.hijax import VJPHiPrimitive

jax.ffi.register_ffi_target(
  "rms_norm_fwd", jax.ffi.pycapsule(rms_norm_lib.RmsNormFwd), platform="cpu")
jax.ffi.register_ffi_target(
  "rms_norm_bwd", jax.ffi.pycapsule(rms_norm_lib.RmsNormBwd), platform="cpu")


class RMSNorm(VJPHiPrimitive):
  def __init__(self, aval, eps):
    if aval.dtype != jnp.float32:
      raise ValueError("Only the float32 dtype is implemented by rms_norm")
    self.in_avals = (aval,)
    self.out_aval = aval
    self.params = dict(eps=np.float32(eps))
    super().__init__()

  def expand(self, x):
    # The plain implementation, in terms of other JAX operations: the FFI call.
    return jax.ffi.ffi_call("rms_norm", self.out_aval)(x, eps=self.eps)

  def vjp_fwd(self, nzs_in, x):
    # The forward pass returns the primal output together with the residuals
    # that the backward pass needs.
    res_aval = jax.ShapeDtypeStruct(x.shape[:-1], x.dtype)
    y, res = jax.ffi.ffi_call(
      "rms_norm_fwd", (self.out_aval, res_aval))(x, eps=self.eps)
    return y, (res, x)

  def vjp_bwd_retval(self, res, ct):
    # The backward pass maps the output co-tangents to input co-tangents.
    res, x = res
    return jax.ffi.ffi_call("rms_norm_bwd", self.in_avals)(res, x, ct)

  def batch(self, axis_data, args, dims):
    # Our handler already treats all leading axes as batch dimensions, so to
    # support `vmap` we move the mapped axis to the front and reapply `rms_norm`.
    x, = args
    bdim, = dims
    return rms_norm(jnp.moveaxis(x, bdim, 0), eps=self.eps), 0


def rms_norm(x, eps=1e-5):
  return RMSNorm(jax.typeof(x), eps)(x)
```

Now `rms_norm` produces the same values as before, but it also transforms correctly under {func}`~jax.vjp` and {func}`~jax.vmap`:

```{code-cell} ipython3
x = jnp.linspace(-0.5, 0.5, 32).reshape((8, 4))
np.testing.assert_allclose(rms_norm(x), rms_norm_ref(x), rtol=1e-5)

# The gradient now matches the reference implementation
ct_y = jnp.ones_like(x)
np.testing.assert_allclose(
  jax.vjp(rms_norm, x)[1](ct_y), jax.vjp(rms_norm_ref, x)[1](ct_y), rtol=1e-5)

# As does the batched version
xs = jnp.linspace(-0.5, 0.5, 96).reshape((3, 8, 4))
np.testing.assert_allclose(
  jax.vmap(rms_norm)(xs), jax.vmap(rms_norm_ref)(xs), rtol=1e-5)
```

## Sharding

Most large scale users of JAX use its APIs for distributed computation across multiple devices.
As discussed in {ref}`parallel`, parallelism in JAX is controlled by sharding data across devices.
The story is a little more complicated for FFI calls, though: since the internals of an FFI call are opaque to both JAX and XLA, an FFI call won't typically partition well when its inputs are sharded.

To see why this matters, recall that our implementation treats all leading axes of the input as _batch_ dimensions, and normalizes along the last axis.
If the data are sharded along a batch dimension (but replicated along the last axis), the normalization is embarrassingly parallel and no communication is required.
XLA can take advantage of this for the pure-JAX `rms_norm_ref`, but it can't see inside our FFI call to do the same: if we naively shard the input, XLA will first gather the whole array onto every device with an `all-gather`, run the FFI call redundantly on the full data, and then slice out each device's portion.

We can do better by handling the sharding _inside_ the primitive, in its `expand` rule.
When using JAX's explicit sharding mode (see {ref}`parallel`), the output sharding of each operation is determined by the shardings of its inputs, and our primitive's `out_aval` carries that sharding.
The idea is to use {func}`~jax.shard_map` to drop into manual ("per-device") mode inside `expand`, so that the FFI call only ever sees the local shard of the data.
Because RMS normalization only reduces over the last (replicated) axis, running the FFI call on each shard independently computes exactly the right answer, with no communication.

First, let's create a mesh and make it the active mesh:

```{code-cell} ipython3
assert len(jax.devices()) == 4  # Set using the XLA_FLAGS environment variable
jax.set_mesh(jax.make_mesh((4,), ("x",)))
```

Then we redefine our primitive so that `expand` wraps the FFI call in a {func}`~jax.shard_map`.
To keep the example focused on sharding we only show the forward pass here, but the same `shard_map` wrapping can be applied to the `vjp_fwd` and `vjp_bwd_retval` rules above to make the differentiated program partition well too.

```{code-cell} ipython3
class RMSNorm(VJPHiPrimitive):
  def __init__(self, aval, eps):
    if aval.dtype != jnp.float32:
      raise ValueError("Only the float32 dtype is implemented by rms_norm")
    self.in_avals = (aval,)
    self.out_aval = aval
    self.params = dict(eps=np.float32(eps))
    super().__init__()

  def expand(self, x):
    body = lambda x: jax.ffi.ffi_call("rms_norm", jax.typeof(x))(x, eps=self.eps)
    return jax.shard_map(body, out_specs=self.out_aval.sharding.spec)(x)


def rms_norm(x, eps=1e-5):
  return RMSNorm(jax.typeof(x), eps)(x)
```

Sharding the input along its batch dimension, the compiled program runs the FFI call directly on each shard, with no `all-gather`, `all-reduce`, or other collectives:

```{code-cell} ipython3
x_batch_shd = jax.device_put(x, jax.P("x"))
np.testing.assert_allclose(rms_norm(x_batch_shd), rms_norm_ref(x), rtol=1e-5)

hlo = jax.jit(rms_norm).lower(x_batch_shd).compile().as_text()
assert "all-" not in hlo
```

## FFI calls on a GPU

So far, we have been interfacing only with foreign functions running on the CPU, but JAX's FFI also supports calls to GPU code.
Since this documentation page is automatically generated on a machine without access to a GPU, we can't execute any GPU-specific examples here, but we will go over the key points.

When defining our FFI wrapper for CPU, the function signature that we used was:

```c++
ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y)
```

To update this to interface with a CUDA kernel, this signature becomes:

```c++
ffi::Error RmsNormImpl(cudaStream_t stream, float eps,
                       ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y)
```

And the handler definition is updated to include a `Ctx` in its binding:

```c++
XLA_FFI_DEFINE_HANDLER(
    RmsNorm, RmsNormImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<float>("eps")
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);
```

Then, the `RmsNormImpl` can use the CUDA stream to launch CUDA kernels.

On the front end, the registration code would be updated to specify the appropriate platform:

```python
jax.ffi.register_ffi_target(
  "rms_norm_cuda", rms_norm_lib_cuda.rms_norm(), platform="CUDA"
)
```

+++

### Supporting multiple platforms

Suppose we have registered both the CPU target `rms_norm` and the GPU target `rms_norm_cuda`.
To support running our function on either platform, we can choose the FFI target inside `expand`, based on the platform that the computation will run on.
In explicit sharding mode the active mesh carries an abstract description of the target device, which we can query to pick the right target name:

```{code-cell} ipython3
class RMSNorm(VJPHiPrimitive):
  def __init__(self, aval, eps):
    if aval.dtype != jnp.float32:
      raise ValueError("Only the float32 dtype is implemented by rms_norm")
    self.in_avals = (aval,)
    self.out_aval = aval
    self.params = dict(eps=np.float32(eps))
    super().__init__()

  def expand(self, x):
    platform = jax.sharding.get_abstract_mesh().abstract_device.platform
    name = {"cpu": "rms_norm", "cuda": "rms_norm_cuda"}[platform]
    body = lambda x: jax.ffi.ffi_call(name, jax.typeof(x))(x, eps=self.eps)
    return jax.shard_map(body, out_specs=self.out_aval.sharding.spec)(x)


def rms_norm(x, eps=1e-5):
  return RMSNorm(jax.typeof(x), eps)(x)


# On this CPU-only machine, the `cpu` target is selected:
np.testing.assert_allclose(rms_norm(x), rms_norm_ref(x), rtol=1e-5)
```

Even without a GPU, we can confirm that the `cuda` target would be selected when lowering for a GPU.
To do this, we lower (but don't compile or run) the function under an abstract mesh that stands in for a GPU device:

```{code-cell} ipython3
gpu_device = jax.sharding.AbstractDevice("gpu", 1, "cuda")
gpu_mesh = jax.sharding.AbstractMesh(
  (4,), ("x",), (jax.sharding.AxisType.Explicit,), abstract_device=gpu_device)
with jax.sharding.use_abstract_mesh(gpu_mesh):
  hlo = jax.jit(rms_norm).lower(x).as_text()
print(hlo)
```

As you can see in the lowered program above, the FFI call now targets `rms_norm_cuda`, even though it was traced on a CPU-only machine.

+++

## Advanced topics

This tutorial covers most of the basic steps that are required to get up and running with JAX's FFI, but advanced use cases may require more features.
We will leave these topics to future tutorials, but here are some possibly useful references:

* **Supporting multiple dtypes**: In this tutorial's example, we restricted to only support `float32` inputs and outputs, but many use cases require supporting multiple different input types. One option to handle this is to register different FFI targets for all supported input types and then use Python to select the appropriate target for {func}`jax.ffi.ffi_call` depending on the input types. But, this approach could get quickly unwieldy depending on the combinatorics of the supported cases. So it is also possible to define the C++ handler to accept `ffi::AnyBuffer` instead of `ffi::Buffer<Dtype>`. Then, the input buffer will include a `element_type()` method which can be used to define the appropriate dtype dispatching logic in the backend.

* **Stateful foreign functions**: It is also possible to use the FFI to wrap functions with associated state. There is a [low-level example included in the XLA test suite](https://github.com/openxla/xla/blob/737a7da3c5405583dc95773ac0bb11b1349fc9ea/xla/service/gpu/custom_call_test.cc#L794-L845), and a future tutorial will include more details.
