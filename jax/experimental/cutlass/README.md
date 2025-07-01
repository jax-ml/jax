# Jax + CuTe DSL

The experimental primitive `cutlass_call` provides a simple API to call kernels written with [CuTe DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

Note that CuTe DSL is still in its beta phase and may change significantly from release to release. As needed this API may be updated or modified as required.

## Installation of CuTe DSL

`pip install nvidia-cutlass-dsl` to get the latest version.

## Calling a Kernel

`cutlass_call` enables compilation and execution of a kernel and a host function from `jax.jit` functions. We assume static layout and shape for `cute.Tensor`s passed from Jax programs. This allows for each instance of a program to be compiled against the exact shapes for maximum efficiency and minimal overhead between Jax and the CuTe DSL program.

```
@cute.kernel
def kernel(in: cute.Tensor, out: cute.Tensor):
    ...

@cute.jit
def launch(stream: cuda.CUstream, input: cute.Tensor, out: cute.Tensor):
    kernel(a, b, c, const_a, const_b).launch(
            grid=[a.shape[-1], 1, 1],
            block=[a.shape[-1], 1, 1],
            stream=stream)

call = cutlass_call(launch, output_shape_dtype=jax.ShapeDtypeStruct((128, 64), jnp.float16))
out = call(input)
```

## Host Function Signature

`cutlass_call` requires a specific function signature to bind arrays and constant values provided by Jax. It is recommend that you annotate the signature with the appropriate or expected types.

All functions must take as the first argument a `cuda.CUstream` that will be used to launch and synchronize the kernel. Following the stream must be input tensors, then input/output tensors, then output tensors. Constexpr values must be passed as named keyword arguments last.

If the kernel signature does not exactly match there are two options: change the host function or use a small wrapper to aid in binding the parameters to the kernel.

```
@cute.jit
def launch(out: cute.Tensor, constval: cutlass.Constexpr, input: cute.Tensor, stream: cuda.CUstream):
    ...

x = cutlass_call(
    lambda stream, input, output, **kwargs: launch(output, kwargs["constval"], input, stream),
    constval=1.0,
    ...
)
```

## Layouts and Modes

`cutlass_call` accepts two optional parameters to aid in converting `jax.Array` into a `cute.Tensor`:

_Layout_: Describes the physical order of axis strides for the source `jax.Array`. If omitted the array is assumed to have row-major layout.
_Mode_: Specifies the order axes and strides of the `cute.Tensor`.

The following example demonstrates how the layout and mode impact the `cute.Tensor` layout.

```
@cute.kernel
def kernel(input: cute.Tensor, out: cute.Tensor):
    cute.printf(input.layout)
    cute.printf(out.layout)

@cute.jit
def launch(stream, input: cute.Tensor, out: cute.Tensor):
    kernel(input, out).launch(grid=[1, 1, 1], block=[1, 1, 1], stream=stream)

a = jnp.zeros((128, 512, 64)) # batch, row, column
call = cutlass_call(launch, output_shape_dtype=jax.ShapeDtypeStruct(a.shape, jnp.float32))
out = call(a)

(128,512,64):(32768,64,1) # default row major layout note shape and strides
(128,512,64):(32768,64,1)

call = cutlass_call(launch, output_shape_dtype=jax.ShapeDtypeStruct(a.shape, jnp.float32), input_mode=((0, 2, 1),))
out = call(a)

(128,64,512):(32768,1,64) # shape and stride reordered to reflect mode.
(128,512,64):(32768,64,1)

call = cutlass_call(launch, output_shape_dtype=jax.ShapeDtypeStruct(a.shape, jnp.float32), input_mode=((0, 2, 1),), output_mode=((2, 1, 0),))
out = call(a)

(128,64,512):(32768,1,64)
(64,512,128):(1,64,32768) # output and input layout can differ

call = cutlass_call(launch, output_shape_dtype=jax.ShapeDtypeStruct(a.shape, jnp.float32), input_layout=((2, 0, 1),))
out = call(a)

(128,512,64):(32768,1,512) # Column major input array
(128,512,64):(32768,64,1)
```

A common example of when to use modes is to implement logical layouts for gemm operations. For example we want the `cute.Tensor`s to follow a consistent layout specification of `[M][K][L]` for A, `[N][K][L]` for B and `[M][N][L]` for C. Strides of each dimension are set to properly offset into physical memory. 


### Complex Layouts and Modes

If your kernel requires more complex layouts at the function boundary e.g. tiled, composed or hierarchical it is recommended that the kernel be wrapped in a `cute.jit` function. Once you are outside of Jax you can built arbitrary layouts as needed using CuTe DSL.

## Compilation Cache

Like Jax, we maintain a cache of the compiled functions. When a kernel is compiled we check if it was previously seen for the given shapes, dtypes and constant values. For example if your kernel is called in a series of homogeneous layer (e.g a transformer model) it will only need to compile once and that instance can be reused as needed.

## Limitations

There are several limitations to highlight to avoid unexpected errors or behavior. Over time we hope to improve these as CuTe DSL matures.

### Jit Function Argument Types

`cutlass_call` does not allow for arbitrary python types to be passed between Jax and the DSL kernel. If the kernel interface depends on a complex Python type it is recommended that a wrapper function be used to bind together `jax.Array` and other compile time constants that can be passed.

```
@cute.jit
def launch(stream: cuda.CUstream, x: CustomType):
    kernel(x).launch(grid=x.get_grid(), block=x.get_block(), stream=stream)

@cute.jit
def wrapper(stream: cuda.CUstream, a: cute.Tensor, b: cute.Tensor, *, constval: float):
    x = CustomType(a, b, constval)
    launch(stream, x)

out = cutlass_call(wrapper, ..., constval=1.0)(a, b)
```

A useful trick can be to flatten a `PyTree` structure outside the call then unflatten inside the call by passing the `PyTree` as a constexpr argument.

#### Variable Length Arguments

Variable length positional arguments are not supported in the `cute.jit` function signature however you can emulate variable length input using `list` or `tuple` types. Its important to keep in mind that these lists are static in length and do not behave like dynamic containers.

#### kwargs

kwargs may not be used to pass `jax.Array`s to the `cutlass_call` they are only used to pass constant/static values.

#### Dictionary Types

Dictionary types are not supported for passing `jax.Array` to the `cutlass_call`.

### Closures and Nested Functions

Closures are not fully supported and may result in unexpected memory consumption or tracer leaks. If you need to capture state, its recommend to wrap your function in a class with global scope. The class can be instantiated deeper into your program with the appropriate values provided.

```
class MyFunction:
    def __init__(self, v0, v1):
        self.v0 = v0
        self.v1 = v1

    @cute.jit
    def __call__(self, stream, ...):
        # use self.v0, self.v1
        ...

def make_function(v0, v1):
    return MyFunction(v0, v1)
```

One common exception is if you need to 

### Autotuning

`cutlass_call` will not autotune arguments to the function. If there are multiple possible configurations you will need to sweep them in a separate program to find the optimal settings for your kernel.

### AoT Compilation and Cache Persistence

There is no support for AoT compilation or compile cache persistence.
