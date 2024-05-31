# Compatibility windows for exported artifacts
*necula@*

*May 2024*

**WORK IN PROGRESS**

## Motivation



You should not use the raw StableHLO that is obtained from just lowering
(`jax.jit(f).lower(1.).compiler_ir()`)
for archival and compilation in another process, for several reasons.

First, the compilation may use a different version of the compiler, with a
different version of StableHLO. The serialization mechanism takes 
care of this by using the
[portable-artifact feature of StableHLO](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md)
to deal with the possible evolution of the StableHLO opset.

### Compatibility guarantees for custom calls

Second, the raw StableHLO may contain custom calls referencing C++
functions.
JAX natively uses custom calls for lowering of certain primitives.
The most common example is for the implementation of PRNG on GPUs,
where we get better performance with a custom call (`cu_threefry32`)
than if we use native StableHLO. Another class of examples are for
some linear algebra primitives (e.g., QR decomposition).
The C++ implementations of these functions
can change and do not fall under the compatibility
guarantees of StableHLO.

The JAX team guarantees **backwards compatibility** for custom calls as follows:
a serialized artifact can be compiled
and executed with a compiler and runtime system that are 
**up to 6 months newer** than the version
of JAX used for lowering and serialization. 

To be precise, in the rare occasions when we need to change the
C++ implementation of the custom call targets, we create a
new C++ function to be used by the new JAX lowering rules.
Say that we changed the JAX lowering rules to use the new
function on January 1st, and this is released
on February 1st. We may delete the old C++ function
no earlier than August 1st. To get the longest
backwards compatibility window you should use
the most recent JAX release for serialization.

We guarantee **forwards compatibility** as follows:
a serialized artifact can be compiled and executed with
a compiler and runtime system that are **up to 1 month older**
than the version used for lowering and serialization.

Only a subset of custom calls are guaranteed stable. We continuously
add more custom call targets to the allowed list along with backwards
compatibility tests. If you try to serialize
code that invokes other custom call targets you will get an error
during exporting.

If you want to disable this safety check for a specific custom call
with target `my_target`, you can add
`export.DisabledSafetyCheck.custom_call("my_target")` to the
`disabled_checks` parameter of the `export` method,
as in the following example:

```python
>>> import jax
>>> from jax import lax
>>> from jax import stages
>>> from jax._src.interpreters import mlir

### Start monkey-patch; to be removed once we finish importing jax.experimental.export into AOT API
>>> from jax.experimental import export
>>> from jax.experimental.export import _export
>>> jax.jit(lambda x: x).lower(1.).__class__.export = lambda self, disabled_checks=(): _export._export_lowered(self, disabled_checks=disabled_checks)
>>> stages.Exported = export.Exported
>>> stages.DisabledSafetyCheck = export.DisabledSafetyCheck
>>> jax.jit(lambda x: x).lower(1.).export().__class__.serialize = export.serialize
>>> stages.Exported.deserialize = export.deserialize
>>> stages.Exported.call = lambda self, *args: export.call(self)(*args)

### End monkey-patch

# override the lowering rule for sin to use a custom call `my_new_sin`
>>> _ = mlir.register_lowering(lax.sin_p, lambda ctx, o: mlir.custom_call("my_new_sin", operands=[o], result_types=[o.type]).results)
>>> print(jax.jit(lax.sin).lower(1.).compiler_ir())
module @jit_sin attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f32> {mhlo.layout_mode = "default"}) -> (tensor<f32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @my_new_sin(%arg0) {api_version = 2 : i32} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

# If we try to export the lowering, we get an error
>>> jax.jit(lax.sin).lower(1.).export()  # doctest: +SKIP
Traceback (most recent call last):
ValueError: Cannot serialize code with custom calls whose targets have no compatibility guarantees: my_new_sin

# We can avoid the error if we pass a `DisabledSafetyCheck.custom_call`
>>> exp = jax.jit(lax.sin).lower(1.).export(disabled_checks=[stages.DisabledSafetyCheck.custom_call("my_new_sin")])

```
