# Primitives with limited support for jax2tf

*Last generated on (YYYY-MM-DD): 2021-02-02*

This document summarizes known limitations of the jax2tf conversion.
There are several kinds of limitations.

  * There are some JAX primitives that are converted to TF ops that have incomplete coverage
  for data types on different kinds of devices,
  see [below](#generated-summary-of-primitives-with-unimplemented-support-in-tensorflow).

  * There are some cases when the converted program computes different results than
  the JAX program, see [below](#generated-summary-of-primitives-with-known-numerical-discrepancies-in-tensorflow).

Note that automated tests will fail if new limitations appear, but
they won't when limitations are fixed. If you see a limitation that
you think it does not exist anymore, please ask for this file to
be updated.

## Generated summary of primitives with unimplemented support in Tensorflow

The following JAX primitives are converted to Tensorflow but the result of the
conversion may trigger runtime errors when run on certain devices and with
certain data types.

This table is organized by JAX primitive, but the actual errors described
in the table are for the Tensorflow ops to which the primitive is converted to.
In general, each JAX primitive is mapped
to one Tensorflow op, e.g., `sin` is mapped to `tf.math.sin`.

The errors apply only for certain devices and compilation modes ("eager",
"graph", and "compiled"). In general, "eager" and "graph" mode share the same errors.
On TPU only the "compiled" mode is relevant.

This table only shows errors for cases that are working in JAX (see [separate
list of unsupported or partially-supported primitives](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/g3doc/jax_primitives_coverage.md) )

We do not yet have support for `pmap` (with its collective primitives),
nor for `sharded_jit` (SPMD partitioning).

We use the following abbreviations for sets of dtypes:

  * `signed` = `int8`, `int16`, `int32`, `int64`
  * `unsigned` = `uint8`, `uint16`, `uint32`, `uint64`
  * `integer` = `signed`, `unsigned`
  * `floating` = `float16`, `bfloat16`, `float32`, `float64`
  * `complex` = `complex64`, `complex128`
  * `inexact` = `floating`, `complex`
  * `all` = `integer`, `inexact`, `bool`

More detailed information can be found in the
[source code for the limitation specification](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/tests/primitives_test.py).


| Affected primitive | Description of limitation | Affected dtypes | Affected devices | Affected compilation modes |
| --- | --- | --- | --- | --- | ---|
| acos | TF error: op not defined for dtype | complex128 | cpu, gpu | eager, graph |
| acos | TF error: op not defined for dtype | bfloat16, complex64, float16 | cpu, gpu | eager, graph |
| acosh | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| asin | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| asin | TF error: op not defined for dtype | complex | cpu, gpu, tpu | compiled, eager, graph |
| asinh | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| atan | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| atan | TF error: op not defined for dtype | complex | cpu, gpu, tpu | compiled, eager, graph |
| atan2 | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| atanh | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| bessel_i0e | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| bessel_i1e | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| bitcast_convert_type | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| cholesky | TF test skipped: Not implemented in JAX: unimplemented | float16 | cpu, gpu | compiled, eager, graph |
| cholesky | TF error: function not compilable | complex | cpu, gpu | compiled |
| cholesky | TF error: op not defined for dtype | complex | tpu | compiled, graph |
| clamp | TF error: op not defined for dtype | int8, uint16, uint32, uint64 | cpu, gpu, tpu | compiled, eager, graph |
| conv_general_dilated | TF error: jax2tf BUG: batch_group_count > 1 not yet converted | all | cpu, gpu, tpu | compiled, eager, graph |
| cosh | TF error: op not defined for dtype | float16 | cpu, gpu | eager, graph |
| cummax | TF test skipped: Not implemented in JAX: unimplemented | complex64 | tpu | compiled, eager, graph |
| cummax | TF error: op not defined for dtype | complex128 | cpu, gpu | compiled, eager, graph |
| cummax | TF error: op not defined for dtype | complex64 | cpu, gpu | compiled, eager, graph |
| cummin | TF test skipped: Not implemented in JAX: unimplemented | complex64 | tpu | compiled, eager, graph |
| cummin | TF error: op not defined for dtype | complex128, uint64 | cpu, gpu | compiled, eager, graph |
| cummin | TF error: op not defined for dtype | complex64, int8, uint16, uint32 | cpu, gpu, tpu | compiled, eager, graph |
| cumprod | TF test skipped: Not implemented in JAX: unimplemented | complex64 | tpu | compiled, eager, graph |
| cumsum | TF error: op not defined for dtype | complex64 | tpu | compiled, eager, graph |
| digamma | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| div | TF error: TF integer division fails if divisor contains 0; JAX returns NaN | integer | cpu, gpu, tpu | compiled, eager, graph |
| div | TF error: op not defined for dtype | int16, int8, unsigned | cpu, gpu, tpu | compiled, eager, graph |
| dot_general | TF error: op not defined for dtype | int64 | cpu, gpu | compiled |
| dot_general | TF error: op not defined for dtype | bool, int16, int8, unsigned | cpu, gpu, tpu | compiled, eager, graph |
| eig | TF test skipped: Not implemented in JAX: only supported on CPU in JAX | all | gpu, tpu | compiled, eager, graph |
| eig | TF test skipped: Not implemented in JAX: unimplemented | bfloat16, float16 | cpu | compiled, eager, graph |
| eig | TF error: TF Conversion of eig is not implemented when both compute_left_eigenvectors and compute_right_eigenvectors are set to True | all | cpu, gpu, tpu | compiled, eager, graph |
| eig | TF error: function not compilable | all | cpu | compiled |
| eigh | TF test skipped: Not implemented in JAX: complex eigh not supported  | complex | tpu | compiled, eager, graph |
| eigh | TF test skipped: Not implemented in JAX: unimplemented | bfloat16, float16 | cpu, gpu, tpu | compiled, eager, graph |
| eigh | TF error: function not compilable | complex | cpu, gpu, tpu | compiled |
| erf | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| erf_inv | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu | eager, graph |
| erfc | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| fft | TF test skipped: Not implemented in JAX: only 1D FFT is currently supported b/140351181. | all | tpu | compiled, eager, graph |
| fft | TF error: TF function not compileable | complex128, float64 | cpu, gpu | compiled |
| ge | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| gt | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| igamma | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu, tpu | compiled, eager, graph |
| igammac | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu, tpu | compiled, eager, graph |
| integer_pow | TF error: op not defined for dtype | unsigned | cpu, gpu, tpu | compiled, eager, graph |
| le | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| lgamma | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| lt | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| lu | TF test skipped: Not implemented in JAX: unimplemented | bfloat16, float16 | cpu, gpu, tpu | compiled, eager, graph |
| lu | TF error: op not defined for dtype | complex64 | tpu | compiled, eager, graph |
| max | TF error: op not defined for dtype | complex128 | cpu, gpu | compiled, eager, graph |
| max | TF error: op not defined for dtype | int8, uint16, uint32, uint64 | cpu, gpu | eager, graph |
| max | TF error: op not defined for dtype | bool, complex64 | cpu, gpu, tpu | compiled, eager, graph |
| min | TF error: op not defined for dtype | complex128 | cpu, gpu | compiled, eager, graph |
| min | TF error: op not defined for dtype | bool, complex64, int8, uint16, uint32, uint64 | cpu, gpu, tpu | compiled, eager, graph |
| neg | TF error: op not defined for dtype | unsigned | cpu, gpu, tpu | compiled, eager, graph |
| nextafter | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu, tpu | compiled, eager, graph |
| population_count | TF error: op not defined for dtype | uint32, uint64 | cpu, gpu | eager, graph |
| qr | TF test skipped: Not implemented in JAX: unimplemented | bfloat16, float16 | cpu, gpu | compiled, eager, graph |
| qr | TF error: op not defined for dtype | bfloat16 | tpu | compiled, eager, graph |
| reduce_max | TF error: op not defined for dtype | complex128 | cpu, gpu, tpu | compiled, eager, graph |
| reduce_max | TF error: op not defined for dtype | complex64 | cpu, gpu, tpu | compiled, eager, graph |
| reduce_min | TF error: op not defined for dtype | complex128 | cpu, gpu, tpu | compiled, eager, graph |
| reduce_min | TF error: op not defined for dtype | complex64 | cpu, gpu, tpu | compiled, eager, graph |
| reduce_window_add | TF error: op not defined for dtype | complex64 | tpu | compiled, eager, graph |
| reduce_window_max | TF test skipped: Not implemented in JAX: unimplemented in XLA | complex64 | tpu | compiled, eager, graph |
| reduce_window_max | TF error: op not defined for dtype | complex128 | cpu, gpu | compiled, eager, graph |
| reduce_window_max | TF error: op not defined for dtype | bool, complex64 | cpu, gpu, tpu | compiled, eager, graph |
| reduce_window_min | TF test skipped: Not implemented in JAX: unimplemented in XLA | complex64 | tpu | compiled, eager, graph |
| reduce_window_min | TF error: op not defined for dtype | bool, complex, int8, uint16, uint32, uint64 | cpu, gpu, tpu | compiled, eager, graph |
| reduce_window_mul | TF test skipped: Not implemented in JAX: unimplemented in XLA | complex64 | tpu | compiled, eager, graph |
| regularized_incomplete_beta | TF error: op not defined for dtype | bfloat16, float16 | cpu, gpu, tpu | compiled, eager, graph |
| rem | TF error: TF integer division fails if divisor contains 0; JAX returns NaN | integer | cpu, gpu, tpu | compiled, eager, graph |
| rem | TF error: op not defined for dtype | float16 | cpu, gpu | eager, graph |
| rem | TF error: op not defined for dtype | int16, int8, unsigned | cpu, gpu, tpu | compiled, eager, graph |
| rev | TF error: op not defined for dtype | uint32, uint64 | cpu, gpu, tpu | compiled, eager, graph |
| round | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| rsqrt | TF error: op not defined for dtype | bfloat16 | cpu, gpu | eager, graph |
| scatter_add | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| scatter_add | TF error: op not defined for dtype | complex64 | tpu | compiled, eager, graph |
| scatter_max | TF test skipped: Not implemented in JAX: unimplemented | complex64 | tpu | compiled, eager, graph |
| scatter_max | TF error: op not defined for dtype | bool, complex | cpu, gpu, tpu | compiled, eager, graph |
| scatter_min | TF test skipped: Not implemented in JAX: unimplemented | complex64 | tpu | compiled, eager, graph |
| scatter_min | TF error: op not defined for dtype | bool, complex, int8, uint16, uint32, uint64 | cpu, gpu, tpu | compiled, eager, graph |
| scatter_mul | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| scatter_mul | TF error: op not defined for dtype | complex64 | tpu | compiled, eager, graph |
| select_and_gather_add | TF error: This JAX primitives is not not exposed directly in the JAX API but arises from JVP of `lax.reduce_window` for reducers `lax.max` or `lax.min`. It also arises from second-order VJP of the same. Implemented using XlaReduceWindow | float32 | tpu | compiled, eager, graph |
| select_and_gather_add | TF error: jax2tf unimplemented for 64-bit inputs because the current implementation relies on packing two values into a single value. This can be fixed by using a variadic XlaReduceWindow, when available | float64 | cpu, gpu | compiled, eager, graph |
| select_and_scatter_add | TF test skipped: Not implemented in JAX: works only for 2 or more inactive dimensions | all | tpu | compiled, eager, graph |
| sign | TF error: op not defined for dtype | int16, int8, unsigned | cpu, gpu, tpu | compiled, eager, graph |
| sinh | TF error: op not defined for dtype | float16 | cpu, gpu | eager, graph |
| sort | TF error: op not defined for dtype | bool | cpu, gpu, tpu | compiled, eager, graph |
| svd | TF test skipped: Not implemented in JAX: complex not implemented. Works in JAX for CPU and GPU with custom kernels | complex | tpu | compiled, eager, graph |
| svd | TF test skipped: Not implemented in JAX: unimplemented | bfloat16, float16 | cpu, gpu | compiled, eager, graph |
| svd | TF error: function not compilable. Implemented using `tf.linalg.svd` and `tf.linalg.adjoint` | complex | cpu, gpu | compiled |
| svd | TF error: op not defined for dtype | bfloat16 | tpu | compiled, eager, graph |
| tie_in | TF test skipped: Not implemented in JAX: requires omnistaging to be disabled | all | cpu, gpu, tpu | compiled, eager, graph |
| top_k | TF error: op not defined for dtype | int64, uint64 | cpu, gpu | compiled |
| triangular_solve | TF test skipped: Not implemented in JAX: unimplemented | float16 | gpu | compiled, eager, graph |
| triangular_solve | TF error: op not defined for dtype | bfloat16 | cpu, gpu, tpu | compiled, eager, graph |
| triangular_solve | TF error: op not defined for dtype | float16 | cpu, gpu | eager, graph |

## Generated summary of primitives with known numerical discrepancies in Tensorflow

In general, we expect a JAX program to produce the same exact answer as its conversion
with jax2tf. The following table lists that cases when this does not quite hold:


| Affected primitive | Description of limitation | Affected dtypes | Affected devices | Affected compilation modes |
| --- | --- | --- | --- | --- | ---|
| acos | May return different but still correct results | complex | cpu, gpu, tpu | eager, graph |
| acosh | May return different but still correct results | complex | cpu, gpu, tpu | eager, graph |
| asin | May return different but still correct results | complex | cpu, gpu, tpu | eager, graph |
| asinh | May return different but still correct results | complex | cpu, gpu, tpu | eager, graph |
| atan | May return different but still correct results | complex | cpu, gpu, tpu | eager, graph |
| atanh | May return different but still correct results | complex | cpu, gpu, tpu | eager, graph |
| cholesky | May return different values in the strictly upper triangular part of the result. This does not matter for correctness, because this part of the matrix is not considered in the result. | all | cpu, gpu, tpu | compiled, eager, graph |
| custom_linear_solve | Numeric comparision disabled: TODO: large numerical discrepancy | float32 | tpu | compiled, eager, graph |
| digamma | May return different results at singularity points 0 and -1.JAX returns nan and TF returns inf | bfloat16 | cpu, gpu, tpu | eager, graph |
| eig | May return the eigenvalues and eigenvectors in a potentially different order. The eigenvectors may also be different, but equally valid. | all | cpu, gpu, tpu | eager, graph |
| eigh | May return the eigenvalues and eigenvectors in a potentially different order. The eigenvectors may also be different, but equally valid. | all | cpu, gpu, tpu | compiled, eager, graph |
| eigh | Numeric comparision disabled: TODO: numeric discrepancies | float64 | cpu, gpu | compiled |
| eigh | Numeric comparision disabled: TODO: numeric discrepancies | float16 | tpu | compiled, eager, graph |
| erf_inv | May return different results at undefined points (< -1 or > 1): JAX returns `NaN` and TF returns `+inf` or `-inf`. | float32, float64 | cpu, gpu, tpu | compiled, eager, graph |
| igamma | May return different results at undefined points (both arguments 0). JAX returns `NaN` and TF returns 0 or JAX returns 1 and TF returns `NaN` | all | cpu, gpu, tpu | eager, graph |
| igammac | May return different results at undefined points (both arguments less or equal 0). JAX returns `NaN` and TF returns 0 or JAX returns 1 and TF returns `NaN` | all | cpu, gpu | eager, graph |
| integer_pow | Numeric comparision disabled: Different overflow behavior for large exponents.  | complex64, float32, signed | cpu, gpu, tpu | compiled, eager, graph |
| integer_pow | custom numeric comparison | complex | cpu, gpu, tpu | compiled, eager, graph |
| lu | May return different, but also correct, results when the decomposition is not unique | all | cpu, gpu, tpu | compiled, eager, graph |
| max | May return different values when one of the values is NaN. JAX always returns NaN, while TF returns the value NaN is compared with. | all | cpu, gpu, tpu | compiled, eager, graph |
| min | May return different values when one of the values is NaN. JAX always returns NaN, while TF returns the value NaN is compared with. | all | cpu, gpu, tpu | compiled, eager, graph |
| pow | custom numeric comparison | complex | cpu, gpu, tpu | compiled, eager, graph |
| sort | Numeric comparision disabled: TODO: TF non-stable multiple-array sort | all | gpu | compiled, eager, graph |
| svd | custom numeric comparison | all | cpu, gpu, tpu | compiled, eager, graph |
| top_k | Produces different results when the array contains `inf` and `NaN` (they are sorted differently in TF vs. XLA). | floating | cpu, gpu, tpu | compiled, eager, graph |

## Updating the documentation

To update this documentation, run the following command:

```
  JAX_ENABLE_X64=1 JAX_OUTPUT_LIMITATIONS_DOC=1 python jax/experimental/jax2tf/tests/primitives_test.py JaxPrimitiveTest.test_generate_limitations_doc
```
