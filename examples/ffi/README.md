# End-to-end example usage for JAX's foreign function interface

This directory includes an example project demonstrating the use of JAX's
foreign function interface (FFI). The JAX docs provide more information about
this interface in [the FFI tutorial](https://docs.jax.dev/en/latest/ffi.html),
but the example in this directory complements that document by demonstrating
(and testing!) the full packaging workflow, and some more advanced use cases.
Within the example project, there are several example calls:

1. `rms_norm`: This is the example from the tutorial on the JAX docs, and it
   demonstrates the most basic use of the FFI. It also includes customization of
   behavior under automatic differentiation using `jax.custom_vjp`.

2. `cpu_examples`: This submodule includes several smaller examples:

   * `counter`: This example demonstrates a common pattern for how an FFI call
     can use global cache to maintain state between calls. This pattern is
     useful when an FFI call requires an expensive initialization step which
     shouldn't be run on every execution, or if there is other shared state
     that could be reused between calls. In this simple example we just count
     the number of times the call was executed.
   * `attrs`: An example demonstrating the different ways that attributes can be
     passed to the FFI. For example, we can pass arrays, variadic attributes,
     and user-defined types. Full support of user-defined types isn't yet
     supported by XLA, so that example will be added in the future.

3. `cuda_examples`: An end-to-end example demonstrating the use of the JAX FFI
   with CUDA. The specifics of the kernels are not very important, but the
   general structure, and packaging of the extension are useful for testing.
