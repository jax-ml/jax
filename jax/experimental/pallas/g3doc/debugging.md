# Debugging Pallas

<!--internal:0-->

<!--*
freshness: { owner: 'slebedev' reviewed: '2025-11-12' }
*-->

[TOC]

This document contains a collection of tips and tricks for debugging Pallas
programs. For any specific requests or ideas for improvement, please create
a ticket on https://github.com/jax-ml/jax/issues.

## Debugging Tools

### Interpret (HLO) Mode

Passing in `interpret=True` into `pl.pallas_call` or `pl.core_map` will run the kernel in HLO instead of lowering to Mosaic/Triton. This is useful for checking correctness of your program and prototyping on smaller block sizes (as TPUs kernels require block sizes of at least 8x128). HLO is also more feature-complete so sometimes kernels will run in interpret mode but fail otherwise - this will make sure the bug is not in your kernel but in Pallas.

Note that interpret mode will not be able to fully replicate the behavior or programs that use communication (DMAs) between devices. This is because low-level communication APIs are more general than the interface that XLA provides via SPMD collective operations.

### TPU Interpret Mode

TPU interpret mode is similar to [interpret (HLO) mode](#interpret-hlo-mode),
but TPU interpret mode explicitly simulates accesses to TPU memory (HBM, VMEM,
SMEM, etc.), communication via remote DMAs, TPU synchronization operations
(e.g., barriers and semaphores), and parallel execution of kernels distributed
across
[multiple TPUs](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html) and
[Megacore cores](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#megacore).

TPU interpret mode is slower than interpret (HLO) mode, but it can be useful for
developing and debugging distributed TPU kernels with explicit communication and
synchronization. With this mode, kernels can be run on CPU -- enabling local
development (with no TPU), using a debugger and inspecting the state of
simulated TPU buffers and semaphores, etc.

To use TPU interpret mode, pass `interpret=pltpu.InterpretParams()` into
`pl.pallas_call` or `pl.core_map`. For examples, see
`test_matmul_example` in
[tpu_pallas_interpret_test.py](https://github.com/jax-ml/jax/blob/main/tests/pallas/tpu_pallas_interpret_test.py#:~:text=test_matmul_example)
and
`test_right_permute_example` and the other tests in
[tpu_pallas_interpret_distributed_test.py](https://github.com/jax-ml/jax/blob/main/tests/pallas/tpu_pallas_interpret_distributed_test.py#:~:text=test_right_permute_example).

The behavior of TPU interpret mode can be configured via arguments to
[`pltpu.InterpretParams`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret.py#:~:text=class%20InterpretParams). For example, use `num_cores_per_device=2`
to simulate Megacore or `uninitialized_memory='zero'` to initialize simuluated
TPU buffers with zeros instead of NaNs.

### debug_print

The `pl.debug_print` function can be used to print runtime values inside of a kernel.

For TPUs only, the kernel must be compiled with the 'xla_tpu_enable_log_recorder' option.
<!--internal:1-->

```python
kernel = pl.pallas_call(...)
compiled_kernel = (
       jax.jit(kernel)
       .lower(x)
       .compile({'xla_tpu_enable_log_recorder': 'true'})
 )
result = compiled_kernel(x)
```

### Runtime Asserts

Checkify can be used to insert runtime asserts, nan checks, out of bounds errors, etc. inside of a kernel.
Pallas implements two options for assertions: a *hard assert* which will crash the TPU if failed, and a *functionalized assertion* which will simulate a runtime assertion that can be thrown
as a Python error after the kernel has successfully executed.

#### Hard assertion

Hard assertions can be inserted with `pl.debug_check`
and running your program with the `--jax_pallas_enable_debug_checks` flag.

Your code will look like the following:

```python
def kernel(...):
  pl.debug_check(x > y, "Check x > y failed")  # Will halt if x <= y
```

This will print a relatively lengthy dump which resembles the following:

```
E1001 15:22:33.275768    4353 real_program_continuator.cc:1350] 0x0x0_TC0: [Physical location: dldgr4:pe1:1] generic::internal: Core halted unexpectedly: INTERNAL: Accelerator device halted prematurely, perhaps due to an on-device check-failure. Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x169 (from TensorCoreSequencer:1:0x213): Check x > y failed HLO: main; HLO computation: main.3
```

The benefit of a hard assertion is that it is guaranteed to either pass or
halt the TPU. The kernel will never proceed past the assertion if it fails.
However, the downside is that if the assertion fails you will
likely have to restart the program in order to run any other TPU operations,
and there is no Python error thrown that can be caught.

#### Functionalized assertion
Functionalized asserts can be performed by checkify-ing the `pl.pallas_call` op like so:

```python
from jax.experimental import checkify

def kernel(...):
  pl.debug_check(x > y, "Check x > y failed")  # Will throw an error if x <= y

kernel = pl.pallas_call(...)
checkified_kernel = checkify.checkify(kernel, errors=checkify.all_checks)
error, result = checkified_kernel(x)
error.throw()
```

This will throw a Python error if any checks failed, such as if a NaN occurred
or if an out-of-bounds index was accessed.

The benefit of a functionalized assert is that it will throw Python errors
that can be caught, and it will not interfere with downstream TPU operations.
However, it requires the kernel to successfully complete, meaning if your
error would have caused a TPU crash, the crash would still happen and
the error would not be thrown.


### Dumping Jaxprs

Passing in `debug=True` into `pl.pallas_call` will print out the Jaxpr of the kernel as well as the lowered Mosaic code.

```python
def kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]

x = jnp.ones((8, 128), dtype=jnp.float32)
pl.pallas_call(
  kernel,
  out_shape=jax.ShapeDTypeStruct((8, 128), jnp.float32)
  debug=True,
  name="my_call",
)(x, x)
```

This will output:

```
The kernel jaxpr for the pallas_call my_call for kernel function kernel at ...:1000:
{ lambda ; a:MemRef<None>{float32[8,128]} b:MemRef<None>{float32[8,128]} c:MemRef<None>{float32[8,128]}. let
    d:f32[8,128] <- a[:,:]
    e:f32[8,128] <- b[:,:]
    f:f32[8,128] = add d e
    c[:,:] <- f
  in () }

The Mosaic module for the pallas_call my_call for kernel function kernel at ...:1000:
module {
  func.func @main(%arg0: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg1: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg2: memref<8x128xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0_0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %c0_1 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %1 = vector.load %arg1[%c0_1, %c0_2] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %2 = arith.addf %0, %1 : vector<8x128xf32>
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %3 = vector.load %arg2[%c0_3, %c0_4] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    vector.store %2, %arg2[%c0_3, %c0_4] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    return
  }
}
```

### Dumping Mosaic Passes

Mosaic is the underlying TPU compiler for Pallas. It can be useful to dump Mosaic if you are running into errors that are originating from the Mosaic compiler to see what code is actually being generated.

Passing the `--xla_mosaic_dump_to=<directory>` argument will dump the output of all intermediate Mosaic passes. The names of the files contain either the parameter `name` passed to the `pallas_call`, or the name of the kernel function. A useful option is to dump to Sponge with `--test_arg=--xla_mosaic_dump_to=sponge` after which you will see all passes under the “Artifacts” tab in sponge.

### Dynamic Race Detection

[TPU Interpret Mode](#tpu-interpret-mode) includes a dynamic race detector.
While running a kernel, it can detect and log data races -- pairs of accesses
to shared memory (HBM, VMEM, SMEM, etc.) that are not properly synchronized.

To enable the dynamic race detector, use the option `detect_races=True` in the
`pltpu.InterpretParams` passed to `pl.pallas_call`:

```python
pl.pallas_call(
  kernel,
  ...,
  intepret=pltpu.InterpretParams(..., detect_races=True),
)
```

If any data races are detected while running the kernel, a message will be
printed -- for example:

```
RACE DETECTED
  write ... from ...jax/tests/pallas/tpu_pallas_interpret_distributed_test.py:1038:10 (InterpretDistributedTest.test_race_detection.<locals>.kernel.<locals>._)
  write ... from .../jax/tests/pallas/tpu_pallas_interpret_distributed_test.py:1038:10 (InterpretDistributedTest.test_race_detection.<locals>.kernel.<locals>._)
```

<!--internal:3-->

## Useful Command line flags

* OOB Checks: `--xla_mosaic_on_device_checks=bounds`
* Poison VMEM allocations: `--xla_jf_poison_vmem_allocations=true`
<!--internal:4-->
* Dump Mosaic: `--xla_mosaic_dump_to=<directory>`
* Enable trace markers in XProf: `--xla_enable_transpose_trace`

## Common Errors

### INTERNAL Mosaic failed to compile TPU Kernel

`INTERNAL Mosaic failed to compile TPU Kernel: Not implemented X`

This error means that you hit an unimplemented case in the underlying Mosaic compiler.
Our recommended course of action here is to file a ticket if one does not already
exist for your specific error.

In some cases, your error may be due to an operation which cannot be implemented
efficiently in the compiler, in which your best course of action is to find a workaround. This
is most commonly seen in `layout` and `shape_cast` errors. The important tip
to remember regarding layouts is that the last 2 dimensions of arrays in Pallas
are physically tiled into registers, so any reshapes, slicing, transposes, etc.
on the last 2 dimensions may trigger a relayout.


### VerificationError

A verification error indicates that Pallas produced invalid code for Mosaic.

This is a bug in Pallas, so please file a bug under https://github.com/jax-ml/jax/issues.

### LoweringError

This is a catch-all error type during Pallas to Mosaic lowering and can have many causes.
In most cases the error message should hint at what is wrong.

For specific errors:

* `Mixed dtype operands in cmp` when using `jnp.mod`: Use lax.rem instead of jnp.mod
