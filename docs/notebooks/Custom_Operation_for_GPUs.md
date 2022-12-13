---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# JAX custom operation for GPUs

JAX, a popular framework for machine learning, comes with a large number of operations, however, users occasionally run into a situation where they need a new operation that is not supported by JAX.
To accommodate such scenarios, JAX allows users to define custom operations and this tutorial is to explain how we can define one for GPUs and use it in single-GPU and multi-GPU environments.

This tutorial contains information from https://jax.readthedocs.io/en/latest/index.html and https://github.com/dfm/extending-jax.

+++

# RMS Normalization

For this tutorial, we are going to add the RMS normalization as a custom operation in JAX.
The CUDA code in `rms_norm_kernels.cu` for this operation has been borrowed from https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu and adapted to eliminate any dependency on PyTorch.

+++

`rms_norm_kernels.cu` defines the following functions, which are declared with the XLA custom function signature.
These functions are responsible for launching RMS normalization kernels with the given `buffers` on the specified `stream`.

```cpp
namespace gpu_ops {
    
void rms_forward_affine_mixed_dtypes(cudaStream_t stream, void **buffers,
                                     const char *opaque,
                                     std::size_t opaque_len);

void rms_backward_affine(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len);

} // namespace gpu_ops
```

* `stream` is the CUDA stream to be used to execute any kernel on the GPU.
* `buffers` has all pointers to input buffers followed by all pointers to output buffers.
* `opaque` is a buffer for any extra information that is being passed to the custom functions and `opaque_len` is the length of `opaque`.

For this tutorial, an `RMSNormDescriptor` object will be passed to these functions as `opaque`.

```cpp
namespace gpu_ops {

enum ElementType { BF16, F16, F32, F64 };

struct RMSNormDescriptor {
  int n1;
  int n2;
  double eps;
  ElementType x_type;
  ElementType w_type;
  int part_grad_size;
};

} // namespace gpu_ops
```

+++

Now, we need to expose these functions as well as `ElementType` and `RMSNormDescriptor` as a Python module, `gpu_ops`, through `pybind11`.

```cpp
pybind11::dict RMSNormRegistrations() {
  pybind11::dict dict;
  dict["rms_forward_affine_mixed_dtype"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_forward_affine_mixed_dtypes);
  dict["rms_backward_affine"] =
      gpu_ops::EncapsulateFunction(gpu_ops::rms_backward_affine);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("rms_norm_registrations", &RMSNormRegistrations);
  m.def("rms_norm_descriptor",
        [](int n1, int n2, double eps, gpu_ops::ElementType x_type,
           gpu_ops::ElementType w_type, int part_grad_size) {
          return gpu_ops::PackDescriptor(gpu_ops::RMSNormDescriptor{
              n1, n2, eps, x_type, w_type, part_grad_size});
        });

  pybind11::enum_<gpu_ops::ElementType>(m, "ElementType")
      .value("BF16", gpu_ops::ElementType::BF16)
      .value("F16", gpu_ops::ElementType::F16)
      .value("F32", gpu_ops::ElementType::F32)
      .value("F64", gpu_ops::ElementType::F64);

}
```

+++

# Build `gpu_ops` extension module

We build the `gpu_ops` Python extension module with the aforementioned code.

```{code-cell} ipython3
!python -m pip install pybind11==2.10.1
import pybind11
import sys

!mkdir -p build
!nvcc --threads 4 -Xcompiler -Wall -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -Xcompiler -O3 --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] --generate-code=arch=compute_80,code=[compute_80,sm_80] --generate-code=arch=compute_86,code=[compute_86,sm_86] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -x cu -c gpu_ops/rms_norm_kernels.cu -o build/rms_norm_kernels.cu.o
!c++ -I/usr/local/cuda/include -I{pybind11.get_include()} $({sys.executable}-config --cflags) -O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects -o build/gpu_ops.cpp.o -c gpu_ops/gpu_ops.cpp
!c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared  -o build/gpu_ops$({sys.executable}-config --extension-suffix) build/gpu_ops.cpp.o build/rms_norm_kernels.cu.o -L/usr/local/cuda/lib64  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
!strip build/gpu_ops$({sys.executable}-config --extension-suffix)
!ls build
```

# Add RMS normalization to JAX as custom call

`gpu_ops` is just a Python extension module and we need more work to plug it into JAX.

+++

## Create primitive ops

We first create primitive operations, `_rms_norm_fwd_p` and `_rms_norm_bwd_p`, which the custom functions can be mapped to.
We set the `multipe_results` attribute to `True` for these operations, which means that the operation produces multipel outputs collected in an list.
When it is set to `False`, the operation produces a single output without a list.

```{code-cell} ipython3
%%capture
from functools import partial

from build import gpu_ops
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client


# Create _rms_norm_fwd_p for forward operation.
_rms_norm_fwd_p = core.Primitive("rms_norm_fwd")
_rms_norm_fwd_p.multiple_results = True
_rms_norm_fwd_p.def_impl(partial(xla.apply_primitive, _rms_norm_fwd_p))


def rms_norm_fwd(x, weight, eps=1e-05):
    output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
    return output


# Create _rms_norm_bwd_p for backward operation.
_rms_norm_bwd_p = core.Primitive("rms_norm_bwd")
_rms_norm_bwd_p.multiple_results = True
_rms_norm_bwd_p.def_impl(partial(xla.apply_primitive, _rms_norm_bwd_p))


def rms_norm_bwd(g, invvar, x, weight, eps):
    grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
        g, invvar, x, weight, eps=eps
    )
    return grad_input, grad_weight
```

## Lowering to MLIR custom call

To map the custom functions to the new primitive operations, `_rms_norm_fwd_p` and `_rms_norm_bwd_p`, we need to:

* Register custom functions as custom call targets with `xla_client.register_custom_call_target`, and
* Register lowering functions that lower the primitive operations to MLIR custom calls with the registered custom call targets.

The functions `_rms_norm_fwd_cuda_lowering` and `_rms_norm_bwd_cuda_lowering` below lower the primitive operations to MLIR custom call operations with the custom targets from `gpu_ops`.  These functions are registered with `jax.interpreters.mlir.register_lowering`.

Note that an `RMSNormDescriptor` object is created in the lowering function, and passed to the custom call as `opaque`.

```{code-cell} ipython3
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.mhlo_helpers import custom_call


# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in gpu_ops.rms_norm_registrations().items():          
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        ir.F64Type.get(): gpu_ops.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _rms_norm_fwd_cuda_lowering(ctx, x, weight, eps):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(weight.type)
    w_shape = w_type.shape
    iv_element_type = (
        ir.F32Type.get()
        if x_type.element_type in [ir.F16Type.get(), ir.BF16Type.get()]
        else x_type.element_type
    )

    n2 = reduce(lambda x, y: x * y, w_shape)
    n1 = reduce(lambda x, y: x * y, x_shape) // n2

    opaque = gpu_ops.rms_norm_descriptor(
        n1,
        n2,
        eps,
        element_type_to_descriptor_type_mapping(x_type.element_type),
        element_type_to_descriptor_type_mapping(w_type.element_type),
        0,  # unused
    )
    out = custom_call(
        b"rms_forward_affine_mixed_dtype",
        out_types=[
            ir.RankedTensorType.get(x_shape, w_type.element_type),
            ir.RankedTensorType.get((n1,), iv_element_type),
        ],
        operands=[x, weight],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape, w_shape),
        result_layouts=default_layouts(x_shape, (n1,)),
    )
    return out


mlir.register_lowering(
    _rms_norm_fwd_p,
    _rms_norm_fwd_cuda_lowering,
    platform="gpu",
)


def _rms_norm_bwd_cuda_lowering(ctx, grad_output, invvar, x, weight, eps):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(weight.type)
    w_shape = w_type.shape
    iv_type = ir.RankedTensorType(invvar.type)

    n2 = reduce(lambda x, y: x * y, w_shape)
    n1 = reduce(lambda x, y: x * y, x_shape) // n2

    part_grad_shape = ctx.avals_out[-1].shape

    opaque = gpu_ops.rms_norm_descriptor(
        n1,
        n2,
        eps,
        element_type_to_descriptor_type_mapping(x_type.element_type),
        element_type_to_descriptor_type_mapping(w_type.element_type),
        part_grad_shape[0],
    )
    out = custom_call(
        b"rms_backward_affine",
        out_types=[
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
            ir.RankedTensorType.get(part_grad_shape, iv_type.element_type),
        ],
        operands=[grad_output, invvar, x, weight],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape, (n1,), x_shape, w_shape),
        result_layouts=default_layouts(x_shape, w_shape, part_grad_shape),
    )
    return out


mlir.register_lowering(
    _rms_norm_bwd_p,
    _rms_norm_bwd_cuda_lowering,
    platform="gpu",
)
```

# Let's test it

```{code-cell} ipython3
import jax


x = jax.random.normal(jax.random.PRNGKey(0), shape=(jax.local_device_count() * 4, 512, 512), dtype=jax.numpy.float16)
norm_shape = x.shape[-2:]
weight = jax.numpy.ones(norm_shape, dtype=jax.numpy.float16)
```

## Test forward function

```{code-cell} ipython3
:tags: [raises-exception]

out = rms_norm_fwd(x, weight)
```

# Abstract evaluation

The test above failed with `NotImplementedError: Abstract evaluation for 'rms_norm_fwd' not implemented`.  Why did the test fail?  What does it mean?

As part of the execution, JAX performs abstract evaluation.  As JAX has no knowledge about the new primitive operations, it doesn't know how to compute the ouptut shapes and outut data types, thus can't evaluate these operations abstractly.

We need to provide a function for abstract evaluation of each primitive operation.
These abstract evaluation functions compute the shape and the data type of the outputs, but don't compute actual values for the operations.

These functions are passed to `.def_abstract_eval` method to be registered with the corresponding primitive operations.

See https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html#abstract-evaluation-rules for more information on abstract evaluation.

```{code-cell} ipython3
%%capture
from functools import reduce
from operator import mul

import jax.numpy as jnp
from jax.abstract_arrays import ShapedArray


def _rms_norm_fwd_abstract(x, weight, eps):
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    iv_dtype = dtypes.canonicalize_dtype(x.dtype)
    if iv_dtype in [jnp.float16, jnp.bfloat16]:
        iv_dtype = jnp.float32
    n2 = reduce(mul, weight.shape)
    n1 = reduce(mul, x.shape) // n2
    return (
        ShapedArray(x.shape, w_dtype, named_shape=x.named_shape),  # output
        ShapedArray((n1,), iv_dtype, named_shape=x.named_shape),  # invvar
    )


_rms_norm_fwd_p.def_abstract_eval(_rms_norm_fwd_abstract)


def _rms_norm_bwd_abstract(grad_output, invvar, x, weight, eps):
    iv_dtype = dtypes.canonicalize_dtype(invvar.dtype)
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    n2 = reduce(lambda x, y: x * y, weight.shape)
    n1 = reduce(lambda x, y: x * y, x.shape) // n2
    part_grad_shape = (16, n2)
    assert dtypes.canonicalize_dtype(grad_output.dtype) == w_dtype
    assert grad_output.shape == x.shape
    assert invvar.shape == (n1,)
    assert (
        iv_dtype == jnp.float32 if x_dtype in [jnp.float16, jnp.bfloat16] else x_dtype
    )
    assert grad_output.named_shape == x.named_shape
    weight_named_shape = (
        weight_named_shape if weight.named_shape else x.named_shape
    )
    return (
        ShapedArray(
            x.shape, x_dtype, named_shape=x.named_shape
        ),  # grad input
        ShapedArray(
            weight.shape, w_dtype, named_shape=weight_named_shape
        ),  # grad weight
        ShapedArray(
            part_grad_shape, iv_dtype, named_shape=weight_named_shape
        ),  # part grad
    )


_rms_norm_bwd_p.def_abstract_eval(_rms_norm_bwd_abstract)
```

# Let's test it again

+++

## Test forward function

```{code-cell} ipython3
out = rms_norm_fwd(x, weight)
```

## Test backward function

Now let's test the backward operation using `jax.grad`.

```{code-cell} ipython3
:tags: [raises-exception]

def loss(x, weight):
    predictions = rms_norm_fwd(x, weight)
    return -jax.numpy.mean(predictions**2)


loss_grad = jax.grad(loss)
out = loss_grad(x, weight)
```

# Differentation rule

The backward operation failed with the error `NotImplementedError: Differentiation rule for 'rms_norm_fwd' not implemented`.  It means that, although we have defined `rms_norm_fwd` and `rms_norm_bwd`, JAX doesn't know the relationship between them.

We can teach JAX that `rms_norm_bwd` is the backward operation for `rms_norm_fwd`, using `jax.custom_vjp` and its convention.  As the first step, we need to refine the definition of `rms_norm_fwd` and `rms_norm_bwd`.

```{code-cell} ipython3
# rms_norm_fwd was previously defined as
#
# def rms_norm_fwd(x, weight, eps=1e-05):
#     output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
#     return output
#
def rms_norm_fwd(x, weight, eps=1e-05):
    output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
    return output, (invvar, x, weight)


# rms_norm_bwd was previously defined as
#
# def rms_norm_bwd(g, invvar, x, weight, eps):
#     grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
#         g, invvar, x, weight, eps=eps
#     )
#     return grad_input, grad_weight
#
def rms_norm_bwd(eps, res, g):
    invvar, x, weight = res
    grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
        g, invvar, x, weight, eps=eps
    )
    return grad_input, grad_weight
```

`rms_norm_fwd` now returns an extra output `(invvar, x, weight)` for the residual data and `rms_norm_bwd` takes `eps`, `res`, and `g` as the parameters.

Once the relationship between `rms_norm_fwd` and `rms_norm_bwd` is estabilished through `jax.custom_vjp`, JAX will ensure that the residual data from `rms_norm_fwd` is passed to `rms_norm_bwd` as `res` for backward operation.
For non-differentiable parameters such as `eps`, JAX ensures that they are passed to the backward operation before the residual data.  That's why `eps` precedes `res` in the parameter list of `rms_norm_bwd`.

Now that `rms_norm_fwd` returns the residual data, which is not needed for simple RMS normalization operation, we define a wrapper around it, `rms_norm`.  It simply calls `rms_norm_fwd` and returns only `output`.  Note that `rms_norm` is annotated with `@partial(jax.custom_vjp, nondiff_argnums=(2,))` and we are passing `rms_norm_fwd` and `rms_norm_bwd` to `rms_norm.defvjp`.  It teaches JAX that, when `rms_norm` is differentiated, `rms_norm_fwd` is to be used for forward operation, and `rms_norm_bwd` is to be used for backward operation.

See https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#use-jax-custom-vjp-to-define-custom-reverse-mode-only-rules for more information on `jax.custom_vjp`.

```{code-cell} ipython3
@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rms_norm(x, weight, eps=1e-05):
    output, _ = rms_norm_fwd(x, weight, eps=eps)
    return output


rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)
```

With the refinement we have made, the backward operation test works with a modification: `loss` now calls `rms_norm` instead of `rms_norm_fwd`.

```{code-cell} ipython3
def loss(x, weight):
    predictions = rms_norm(x, weight)
    return -jax.numpy.mean(predictions**2)


loss_grad = jax.grad(loss)
out = loss_grad(x, weight)
```

# Let's test it on multiple devices

We are using `jax.experimental.pjit.pjit` for parallel execution on multiple devices, and we produce reference values with sequential execution on a single device.

+++

## Test forward function

Let's first test the forward operation on multiple devices.  We are creating a simple 1D mesh and sharding `x` on all devices.

```{code-cell} ipython3
import numpy
from jax.experimental.maps import Mesh
from jax.experimental.pjit import PartitionSpec, pjit


mesh = Mesh(numpy.array(jax.local_devices()).reshape(-1), ("a",))
ref = rms_norm(x, weight)
pjitted = pjit(
    rms_norm,
    in_axis_resources=(PartitionSpec("a", None, None), PartitionSpec(None, None)),
    out_axis_resources=PartitionSpec("a", None, None),
)

with mesh:
    print(pjitted.lower(x, weight).compile().runtime_executable().hlo_modules()[0].to_string())
    out = pjitted(x, weight)

numpy.allclose(ref, out)
```

The values have been computed correctly for forward operation, however, the generated HLO modules shows an `all-gather` operation to replicate `x` on all devices, incurring large communication overhead.

As XLA does not have enough knowledge about the custom functions to shard input tensors, it decides to replicate them to produce correct values before making the custom call.

To avoid this overhead, we need to use the xmap manual sharding with the following configuration updates

```{code-cell} ipython3
jax.config.update("experimental_xmap_spmd_lowering", True)
jax.config.update("experimental_xmap_spmd_lowering_manual", True)
```

We need to modify the test code to use the xmap manual sharding with the custom operation.

We first define a function that wraps `rms_norm` with `xmap`.  As the size of the data axis that is being sharded must match the size of the corresponding mesh axis in the xmap manual sharding mode, we reshape `x` with the new shape `(device_count, x.shape[0] // device_count, *x.shape[1:])`, and `device_count` represents the size of the corresponding mesh axis.

After running `rms_norm` through `xmap`, we reshape the output to match the shape of `x` to match the expectation from clients.

```{code-cell} ipython3
from jax.experimental.maps import xmap


def xmap_rms_norm(x, weight, *, device_count):
    reshaped = x.reshape(device_count, x.shape[0] // device_count, *x.shape[1:])
    xmap_axes = (("a", None, None, None), (None, None))
    xmapped = xmap(
        rms_norm,
        in_axes=xmap_axes,
        out_axes=xmap_axes[0],
        axis_resources={"a": "a"},
    )
    reshaped_out = xmapped(reshaped, weight)
    return reshaped_out.reshape(x.shape)
```

Now we need to run `xmap_rms_norm`, not `rms_norm` through `pjit`.

```{code-cell} ipython3
with mesh:

    pjitted = pjit(
        partial(xmap_rms_norm, device_count=jax.local_device_count()),
        in_axis_resources=(
            PartitionSpec("a", None, None),
            PartitionSpec(None, None),
        ),
        out_axis_resources=PartitionSpec("a", None, None),
    )
    print(pjitted.lower(x, weight).compile().runtime_executable().hlo_modules()[0].to_string())
    out = pjitted(x, weight)

numpy.allclose(ref, out)
```

With this modification, the `all-gather` operation is eliminated and the custom call is made on each shard of `x`.

+++

## Test backward function

We are moving onto the the backward operation using `jax.grad` on multiple devices.

Similiarly to the forward operation test, we are creating a simple 1D mesh and sharding `x` on all devices.

We also define the `loss` function with `xmap_rms_norm` instead of `rms_norm`

```{code-cell} ipython3
def loss_ref(x, weight):
    predictions = rms_norm(x, weight)
    return -jax.numpy.mean(predictions**2)


ref = jax.grad(loss_ref, argnums=(0, 1))(x, weight)


# Re-define loss to use xmap_rms_norm instead of rms_norm
def loss(x, weight, *, device_count):
    predictions = xmap_rms_norm(x, weight, device_count=device_count)
    return -jax.numpy.mean(predictions**2)


with mesh:

    pjitted = pjit(
        jax.grad(partial(loss, device_count=jax.local_device_count()), argnums=(0, 1)),
        in_axis_resources=(
            PartitionSpec("a", None, None),
            PartitionSpec(None, None),
        ),
        out_axis_resources=(
            PartitionSpec("a", None, None),
            PartitionSpec(None, None),
        ),
    )
    out = pjitted(x, weight)

for r, o in zip(ref, out):
    print(numpy.allclose(r, o))
```

We can inspect the generated jaxpr, which is the JAX internal representation, to make sure `jax.grad` inserts a `psum` for the gradient accumulation across the devices when needed.

```{code-cell} ipython3
with mesh:
    
    print(jax.make_jaxpr(pjitted)(x, weight))
```

We see that `bm:f16[512,512] = psum[axes=('a',) axis_index_groups=None] bl` has been added after the call to `rms_norm_bwd` to reduce `grad_weight` across the devics on the axis `a`, but there is no `psum` for `grad_input`.

This is controlled by `named_shape` passed to the `ShapedArray` construction in abstract evaluation and the axes given to `xmap`.

The following code snippet from `_rms_norm_bwd_abstract` shows that `grad_input` has the exact same shape, type, and named shape as `x` does, which means `grad_input` is sharded the same way as `x`, hence no need for a `psum` for `grad_input`.
In contrast, `grad_weight` has the same shape and type as `weight` does, but, when `weight.named_shape` is empty, `x.named_shape` is used for `grad_weight`.  In `in_axes` of our `xmap` call, `weight` has no named axis and `weight.named_shape` is empty, but `grad_weight` now has a named axis `"a"` in `grad_weight.named_shape`.
This makes `jax.grad` insert `psum` on the axis `"a"` for `grad_weight`.

```
weight_named_shape = (
    weight_named_shape if weight.named_shape else x.named_shape
)
...
return (
    ShapedArray(
        x.shape, x_dtype, named_shape=x.named_shape
    ),  # grad input
    ShapedArray(
        weight.shape, w_dtype, named_shape=weight_named_shape
    ),  # grad weight
    ....
)
```

+++

# Let's put it together

Here is the complete code that is functional.

```{code-cell} ipython3
%reset -f

from functools import partial, reduce
from operator import mul

import jax
import jax.numpy as jnp
from build import gpu_ops
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.experimental.maps import Mesh, xmap
from jax.experimental.pjit import PartitionSpec, pjit
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir
from jax.lib import xla_client
from jaxlib.mhlo_helpers import custom_call


# Create _rms_norm_fwd_p for forward operation.
_rms_norm_fwd_p = core.Primitive("rms_norm_fwd")
_rms_norm_fwd_p.multiple_results = True
_rms_norm_fwd_p.def_impl(partial(xla.apply_primitive, _rms_norm_fwd_p))


def rms_norm_fwd(x, weight, eps=1e-05):
    output, invvar = _rms_norm_fwd_p.bind(x, weight, eps=eps)
    return output, (invvar, x, weight)


# Create _rms_norm_bwd_p for backward operation.
_rms_norm_bwd_p = core.Primitive("rms_norm_bwd")
_rms_norm_bwd_p.multiple_results = True
_rms_norm_bwd_p.def_impl(partial(xla.apply_primitive, _rms_norm_bwd_p))


def rms_norm_bwd(eps, res, g):
    invvar, x, weight = res
    grad_input, grad_weight, part_grad = _rms_norm_bwd_p.bind(
        g, invvar, x, weight, eps=eps
    )
    return grad_input, grad_weight


####################
# Lowering to MLIR #
####################


# Register functions defined in gpu_ops as custom call target for GPUs
for _name, _value in gpu_ops.rms_norm_registrations().items():          
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F16Type.get(): gpu_ops.ElementType.F16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
        ir.F64Type.get(): gpu_ops.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _rms_norm_fwd_cuda_lowering(ctx, x, weight, eps):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(weight.type)
    w_shape = w_type.shape
    iv_element_type = (
        ir.F32Type.get()
        if x_type.element_type in [ir.F16Type.get(), ir.BF16Type.get()]
        else x_type.element_type
    )

    n2 = reduce(lambda x, y: x * y, w_shape)
    n1 = reduce(lambda x, y: x * y, x_shape) // n2

    opaque = gpu_ops.rms_norm_descriptor(
        n1,
        n2,
        eps,
        element_type_to_descriptor_type_mapping(x_type.element_type),
        element_type_to_descriptor_type_mapping(w_type.element_type),
        0,  # unused
    )
    out = custom_call(
        b"rms_forward_affine_mixed_dtype",
        out_types=[
            ir.RankedTensorType.get(x_shape, w_type.element_type),
            ir.RankedTensorType.get((n1,), iv_element_type),
        ],
        operands=[x, weight],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape, w_shape),
        result_layouts=default_layouts(x_shape, (n1,)),
    )
    return out


mlir.register_lowering(
    _rms_norm_fwd_p,
    _rms_norm_fwd_cuda_lowering,
    platform="gpu",
)


def _rms_norm_bwd_cuda_lowering(ctx, grad_output, invvar, x, weight, eps):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    w_type = ir.RankedTensorType(weight.type)
    w_shape = w_type.shape
    iv_type = ir.RankedTensorType(invvar.type)

    n2 = reduce(lambda x, y: x * y, w_shape)
    n1 = reduce(lambda x, y: x * y, x_shape) // n2

    part_grad_shape = ctx.avals_out[-1].shape

    opaque = gpu_ops.rms_norm_descriptor(
        n1,
        n2,
        eps,
        element_type_to_descriptor_type_mapping(x_type.element_type),
        element_type_to_descriptor_type_mapping(w_type.element_type),
        part_grad_shape[0],
    )
    out = custom_call(
        b"rms_backward_affine",
        out_types=[
            ir.RankedTensorType.get(x_shape, x_type.element_type),
            ir.RankedTensorType.get(w_shape, w_type.element_type),
            ir.RankedTensorType.get(part_grad_shape, iv_type.element_type),
        ],
        operands=[grad_output, invvar, x, weight],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape, (n1,), x_shape, w_shape),
        result_layouts=default_layouts(x_shape, w_shape, part_grad_shape),
    )
    return out


mlir.register_lowering(
    _rms_norm_bwd_p,
    _rms_norm_bwd_cuda_lowering,
    platform="gpu",
)


#######################
# Abstract evaluation #
#######################


def _rms_norm_fwd_abstract(x, weight, eps):
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    iv_dtype = dtypes.canonicalize_dtype(x.dtype)
    if iv_dtype in [jnp.float16, jnp.bfloat16]:
        iv_dtype = jnp.float32
    n2 = reduce(mul, weight.shape)
    n1 = reduce(mul, x.shape) // n2
    return (
        ShapedArray(x.shape, w_dtype, named_shape=x.named_shape),  # output
        ShapedArray((n1,), iv_dtype, named_shape=x.named_shape),  # invvar
    )


_rms_norm_fwd_p.def_abstract_eval(_rms_norm_fwd_abstract)


def _rms_norm_bwd_abstract(grad_output, invvar, x, weight, eps):
    iv_dtype = dtypes.canonicalize_dtype(invvar.dtype)
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    n2 = reduce(lambda x, y: x * y, weight.shape)
    n1 = reduce(lambda x, y: x * y, x.shape) // n2
    part_grad_shape = (16, n2)
    assert dtypes.canonicalize_dtype(grad_output.dtype) == w_dtype
    assert grad_output.shape == x.shape
    assert invvar.shape == (n1,)
    assert (
        iv_dtype == jnp.float32 if x_dtype in [jnp.float16, jnp.bfloat16] else x_dtype
    )
    assert grad_output.named_shape == x.named_shape
    weight_named_shape = (
        weight_named_shape if weight.named_shape else grad_output.named_shape
    )
    return (
        ShapedArray(
            x.shape, x_dtype, named_shape=x.named_shape
        ),  # grad input
        ShapedArray(
            weight.shape, w_dtype, named_shape=weight_named_shape
        ),  # grad weight
        ShapedArray(
            part_grad_shape, iv_dtype, named_shape=weight_named_shape
        ),  # part grad
    )


_rms_norm_bwd_p.def_abstract_eval(_rms_norm_bwd_abstract)


#######################################
# Top-level interface with custom vjp #
#######################################


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rms_norm(x, weight, eps=1e-05):
    output, _ = rms_norm_fwd(x, weight, eps=eps)
    return output


rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)


######################
# RMS norm with xmap #
######################


jax.config.update("experimental_xmap_spmd_lowering", True)
jax.config.update("experimental_xmap_spmd_lowering_manual", True)


def xmap_rms_norm(x, weight, *, device_count):
    reshaped = x.reshape(device_count, x.shape[0] // device_count, *x.shape[1:])
    xmap_axes = (("a", None, None, None), (None, None))
    xmapped = xmap(
        rms_norm,
        in_axes=xmap_axes,
        out_axes=xmap_axes[0],
        axis_resources={"a": "a"},
    )
    reshaped_out = xmapped(reshaped, weight)
    return reshaped_out.reshape(x.shape)


########
# Test #
########


import jax
import numpy


x = jax.random.normal(jax.random.PRNGKey(0), shape=(jax.local_device_count() * 4, 512, 512), dtype=jax.numpy.float16)
norm_shape = x.shape[-2:]
weight = jax.numpy.ones(norm_shape, dtype=jax.numpy.float16)


def loss_ref(x, weight):
    predictions = rms_norm(x, weight)
    return -jax.numpy.mean(predictions**2)


ref = jax.grad(loss_ref, argnums=(0, 1))(x, weight)


def loss(x, weight, *, device_count):
    predictions = xmap_rms_norm(x, weight, device_count=device_count)
    return -jax.numpy.mean(predictions**2)


with Mesh(numpy.array(jax.local_devices()).reshape(-1), ("a",)):

    pjitted = pjit(
        jax.grad(partial(loss, device_count=jax.local_device_count()), argnums=(0, 1)),
        in_axis_resources=(
            PartitionSpec("a", None, None),
            PartitionSpec(None, None),
        ),
        out_axis_resources=(
            PartitionSpec("a", None, None),
            PartitionSpec(None, None),
        ),
    )
    out = pjitted(x, weight)

for r, o in zip(ref, out):
    print(numpy.allclose(r, o))
```
