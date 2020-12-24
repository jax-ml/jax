# Primitives with limited JAX support

*Last generated on: 2020-12-27* (YYYY-MM-DD)

## Supported data types for primitives

We use a set of 2305 test harnesses for testing
the implementation of 122 numeric JAX primitives.
Not all primitives are supported in JAX at all
data types. The following table shows the dtypes at which
**primitives are NOT supported on any device**.
(In reality, this shows for each primitive what dtypes are not covered
by the current harnesses on any device.)

Note also that the set of supported dtypes include 64-bit types
(`float64`, `int64`, `uint64`, `complex128`) only of the
flag `--jax_enable_x64` is set (or the JAX_ENABLE_X64 environment
variable).

We use the following abbreviations for sets of dtypes:

  * `all_signed_integers` = `int8`, `int16`, `int32`, `int64`
  * `all_unsigned_integers` = `uint8`, `uint16`, `uint32`, `uint64`
  * `all_integers` = `all_signed_integers`, `all_unsigned_integers`
  * `all_float` = `float16`, `bfloat16`, `float32`, `float64`
  * `all_complex` = `complex64`, `complex128`
  * `all_inexact` = `all_float`, `all_complex`
  * `all` = `all_integers`, `all_inexact`, `bool`

In order to experiment with increased coverage, add more harnesses for
more data types.


| Primitive | Nr. test harnesses | dtypes supported on at least one device | dtypes NOT tested on any device |
| --- | --- | --- | --- | --- |
| abs | 10 | all_inexact, all_signed_integers | all_unsigned_integers, bool |
| acos | 6 | all_inexact | all_integers, bool |
| acosh | 6 | all_inexact | all_integers, bool |
| add | 16 | all_inexact, all_integers | bool |
| add_any | 14 | all_inexact, all_integers | bool |
| and | 11 | all_integers, bool | all_inexact |
| argmax | 22 | all_float, all_integers, bool | all_complex |
| argmin | 22 | all_float, all_integers, bool | all_complex |
| asin | 6 | all_inexact | all_integers, bool |
| asinh | 6 | all_inexact | all_integers, bool |
| atan | 6 | all_inexact | all_integers, bool |
| atan2 | 6 | all_float | all_complex, all_integers, bool |
| atanh | 6 | all_inexact | all_integers, bool |
| bessel_i0e | 4 | all_float | all_complex, all_integers, bool |
| bessel_i1e | 4 | all_float | all_complex, all_integers, bool |
| bitcast_convert_type | 41 | all |  |
| broadcast | 17 | all |  |
| broadcast_in_dim | 19 | all |  |
| ceil | 4 | all_float | all_complex, all_integers, bool |
| cholesky | 30 | all_inexact | all_integers, bool |
| clamp | 17 | all_float, all_integers | all_complex, bool |
| complex | 4 | float32, float64 | all_complex, all_integers, bfloat16, bool, float16 |
| concatenate | 17 | all |  |
| conj | 5 | all_complex, float32, float64 | all_integers, bfloat16, bool, float16 |
| conv_general_dilated | 58 | all_inexact | all_integers, bool |
| convert_element_type | 201 | all |  |
| cos | 6 | all_inexact | all_integers, bool |
| cosh | 6 | all_inexact | all_integers, bool |
| cummax | 17 | all_inexact, all_integers | bool |
| cummin | 17 | all_inexact, all_integers | bool |
| cumprod | 17 | all_inexact, all_integers | bool |
| cumsum | 17 | all_inexact, all_integers | bool |
| custom_linear_solve | 4 | float32, float64 | all_complex, all_integers, bfloat16, bool, float16 |
| device_put | 16 | all |  |
| digamma | 4 | all_float | all_complex, all_integers, bool |
| div | 20 | all_inexact, all_integers | bool |
| dot_general | 125 | all |  |
| dynamic_slice | 32 | all |  |
| dynamic_update_slice | 21 | all |  |
| eig | 72 | all_inexact | all_integers, bool |
| eigh | 36 | all_inexact | all_integers, bool |
| eq | 17 | all |  |
| erf | 4 | all_float | all_complex, all_integers, bool |
| erf_inv | 4 | all_float | all_complex, all_integers, bool |
| erfc | 4 | all_float | all_complex, all_integers, bool |
| exp | 6 | all_inexact | all_integers, bool |
| expm1 | 6 | all_inexact | all_integers, bool |
| fft | 20 | all_complex, float32, float64 | all_integers, bfloat16, bool, float16 |
| floor | 4 | all_float | all_complex, all_integers, bool |
| gather | 37 | all |  |
| ge | 15 | all_float, all_integers, bool | all_complex |
| gt | 15 | all_float, all_integers, bool | all_complex |
| igamma | 6 | all_float | all_complex, all_integers, bool |
| igammac | 6 | all_float | all_complex, all_integers, bool |
| imag | 2 | all_complex | all_float, all_integers, bool |
| integer_pow | 34 | all_inexact, all_integers | bool |
| iota | 16 | all_inexact, all_integers | bool |
| is_finite | 4 | all_float | all_complex, all_integers, bool |
| le | 15 | all_float, all_integers, bool | all_complex |
| lgamma | 4 | all_float | all_complex, all_integers, bool |
| log | 6 | all_inexact | all_integers, bool |
| log1p | 6 | all_inexact | all_integers, bool |
| lt | 15 | all_float, all_integers, bool | all_complex |
| lu | 18 | all_inexact | all_integers, bool |
| max | 29 | all |  |
| min | 29 | all |  |
| mul | 16 | all_inexact, all_integers | bool |
| ne | 17 | all |  |
| neg | 14 | all_inexact, all_integers | bool |
| nextafter | 6 | all_float | all_complex, all_integers, bool |
| or | 11 | all_integers, bool | all_inexact |
| pad | 90 | all |  |
| population_count | 8 | all_integers | all_inexact, bool |
| pow | 10 | all_inexact | all_integers, bool |
| qr | 60 | all_inexact | all_integers, bool |
| random_gamma | 4 | float32, float64 | all_complex, all_integers, bfloat16, bool, float16 |
| random_split | 5 | uint32 | all |
| real | 2 | all_complex | all_float, all_integers, bool |
| reduce_and | 1 | bool | all_inexact, all_integers |
| reduce_max | 15 | all |  |
| reduce_min | 15 | all |  |
| reduce_or | 1 | bool | all_inexact, all_integers |
| reduce_prod | 14 | all_inexact, all_integers | bool |
| reduce_sum | 14 | all_inexact, all_integers | bool |
| reduce_window_add | 33 | all_inexact, all_integers | bool |
| reduce_window_max | 37 | all |  |
| reduce_window_min | 15 | all |  |
| reduce_window_mul | 42 | all_inexact, all_integers | bool |
| regularized_incomplete_beta | 4 | all_float | all_complex, all_integers, bool |
| rem | 18 | all_float, all_integers | all_complex, bool |
| reshape | 19 | all |  |
| rev | 19 | all |  |
| round | 7 | all_float | all_complex, all_integers, bool |
| rsqrt | 6 | all_inexact | all_integers, bool |
| scatter_add | 14 | all_inexact, all_integers | bool |
| scatter_max | 15 | all |  |
| scatter_min | 19 | all |  |
| scatter_mul | 14 | all_inexact, all_integers | bool |
| select | 16 | all |  |
| select_and_gather_add | 15 | all_float | all_complex, all_integers, bool |
| select_and_scatter_add | 27 | all_float, all_integers, bool | all_complex |
| shift_left | 10 | all_integers | all_inexact, bool |
| shift_right_arithmetic | 10 | all_integers | all_inexact, bool |
| shift_right_logical | 10 | all_integers | all_inexact, bool |
| sign | 14 | all_inexact, all_integers | bool |
| sin | 6 | all_inexact | all_integers, bool |
| sinh | 6 | all_inexact | all_integers, bool |
| slice | 24 | all |  |
| sort | 21 | all |  |
| sqrt | 6 | all_inexact | all_integers, bool |
| squeeze | 23 | all |  |
| stop_gradient | 15 | all |  |
| sub | 16 | all_inexact, all_integers | bool |
| svd | 120 | all_inexact | all_integers, bool |
| tan | 6 | all_inexact | all_integers, bool |
| tanh | 6 | all_inexact | all_integers, bool |
| tie_in | 15 | all |  |
| top_k | 15 | all_float, all_integers, bool | all_complex |
| transpose | 17 | all |  |
| triangular_solve | 26 | all_inexact | all_integers, bool |
| xor | 11 | all_integers, bool | all_inexact |
| zeros_like | 15 | all |  |

## Partially implemented data types for primitives

In some cases, a primitive is supported at a given data type but
it may be missing implementations for some of the devices.
For example, the eigen decomposition (`lax.eig`) is implemented
in JAX using custom kernels only on CPU and GPU. There is no
TPU implementation. In other cases, there are either bugs or
not-yet-implemented cases in the XLA compiler for different
devices.

The following table shows which of the supported data types
are partially implemented for each primitive. This table already
excludes the data types not supported (previous table).

In order to see the actual errors for all entries above look at
the logs of the `test_jax_implemented` from `jax_primitives_coverage_test.py`.


| Affected primitive | Description of limitation | Affected dtypes | Affected devices |
| --- | --- | --- | --- | --- |
|cholesky|unimplemented|float16|cpu, gpu|
|cummax|not implemented|complex64|tpu|
|cummin|not implemented|complex64|tpu|
|cumprod|not implemented|complex64|tpu|
|eig|only supported on CPU in JAX|all|tpu, gpu|
|eig|unimplemented|bfloat16, float16|cpu|
|eigh|complex eigh not supported |all_complex|tpu|
|eigh|unimplemented|float16|cpu|
|eigh|unimplemented|float16|gpu|
|fft|only 1D FFT is currently supported b/140351181.|all|tpu|
|igamma|XLA internal error|bfloat16, float16|cpu, gpu, tpu|
|igammac|XLA internal error|bfloat16, float16|cpu, gpu, tpu|
|lu|unimplemented|bfloat16, float16|cpu, gpu, tpu|
|nextafter|XLA internal error, implicit broadcasting not implemented|all|cpu, gpu, tpu|
|qr|unimplemented|bfloat16, float16|cpu, gpu|
|reduce_window_max|unimplemented in XLA|complex64|tpu|
|reduce_window_min|unimplemented in XLA|complex64|tpu|
|reduce_window_mul|unimplemented in XLA|complex64|tpu|
|scatter_add|not implemented|complex64|tpu|
|scatter_max|not implemented|complex64|tpu|
|scatter_min|not implemented|complex64|tpu|
|scatter_mul|not implemented|complex64|tpu|
|select_and_scatter_add|works only for 2 or more inactive dimensions|all|tpu|
|svd|unimplemented|bfloat16, float16|cpu, gpu|
|svd|complex not implemented|all_complex|tpu|
|tie_in|requires omnistaging to be disabled|all|cpu, gpu, tpu|
|triangular_solve|not implemented|float16|gpu|

## Table generation

To regenerate this table run on a CPU machine::

```
  JAX_OUTPUT_LIMITATIONS_DOC=1 JAX_ENABLE_X64=1 python jax/experimental/jax2tf/tests/jax_primitives_coverage_test.py JaxPrimitiveTest.test_generate_primitives_coverage_doc
```
