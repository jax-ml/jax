# Primitives with limited JAX support

*Last generated on: 2020-12-28* (YYYY-MM-DD)

## Supported data types for primitives

We use a set of 2305 test harnesses for testing
the implementation of 122 numeric JAX primitives.
Not all primitives are supported in JAX at all
data types. The following table shows the dtypes at which
**primitives are NOT supported on any device**.
(In reality, this shows for each primitive what dtypes are not covered
by the current harnesses on any device.)

Note also that the set of supported dtypes include 64-bit types
(`float64`, `int64`, `uint64`, `complex128`) only if the
flag `--jax_enable_x64` is set (or the JAX_ENABLE_X64 environment
variable).

We use the following abbreviations for sets of dtypes:

  * `signed` = `int8`, `int16`, `int32`, `int64`
  * `unsigned` = `uint8`, `uint16`, `uint32`, `uint64`
  * `integer` = `signed`, `unsigned`
  * `floating` = `float16`, `bfloat16`, `float32`, `float64`
  * `complex` = `complex64`, `complex128`
  * `inexact` = `floating`, `complex`
  * `all` = `integer`, `inexact`, `bool`

In order to experiment with increased coverage, add more harnesses for
more data types.


| Primitive | Total test harnesses | dtypes supported on at least one device | dtypes NOT tested on any device |
| --- | --- | --- | --- | --- |
| abs | 10 | complex, floating, signed | bool, unsigned |
| acos | 6 | complex, floating | bool, integer |
| acosh | 6 | complex, floating | bool, integer |
| add | 16 | complex, floating, integer | bool |
| add_any | 14 | complex, floating, integer | bool |
| and | 11 | bool, integer | complex, floating |
| argmax | 22 | bool, floating, integer | complex |
| argmin | 22 | bool, floating, integer | complex |
| asin | 6 | complex, floating | bool, integer |
| asinh | 6 | complex, floating | bool, integer |
| atan | 6 | complex, floating | bool, integer |
| atan2 | 6 | floating | bool, complex, integer |
| atanh | 6 | complex, floating | bool, integer |
| bessel_i0e | 4 | floating | bool, complex, integer |
| bessel_i1e | 4 | floating | bool, complex, integer |
| bitcast_convert_type | 41 | bool, complex, floating, integer |  |
| broadcast | 17 | bool, complex, floating, integer |  |
| broadcast_in_dim | 19 | bool, complex, floating, integer |  |
| ceil | 4 | floating | bool, complex, integer |
| cholesky | 30 | complex, floating | bool, integer |
| clamp | 17 | floating, integer | bool, complex |
| complex | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| concatenate | 17 | bool, complex, floating, integer |  |
| conj | 5 | complex, float32, float64 | bfloat16, bool, float16, integer |
| conv_general_dilated | 58 | complex, floating | bool, integer |
| convert_element_type | 201 | bool, complex, floating, integer |  |
| cos | 6 | complex, floating | bool, integer |
| cosh | 6 | complex, floating | bool, integer |
| cummax | 17 | complex, floating, integer | bool |
| cummin | 17 | complex, floating, integer | bool |
| cumprod | 17 | complex, floating, integer | bool |
| cumsum | 17 | complex, floating, integer | bool |
| custom_linear_solve | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| device_put | 16 | bool, complex, floating, integer |  |
| digamma | 4 | floating | bool, complex, integer |
| div | 20 | complex, floating, integer | bool |
| dot_general | 125 | bool, complex, floating, integer |  |
| dynamic_slice | 32 | bool, complex, floating, integer |  |
| dynamic_update_slice | 21 | bool, complex, floating, integer |  |
| eig | 72 | complex, floating | bool, integer |
| eigh | 36 | complex, floating | bool, integer |
| eq | 17 | bool, complex, floating, integer |  |
| erf | 4 | floating | bool, complex, integer |
| erf_inv | 4 | floating | bool, complex, integer |
| erfc | 4 | floating | bool, complex, integer |
| exp | 6 | complex, floating | bool, integer |
| expm1 | 6 | complex, floating | bool, integer |
| fft | 20 | complex, float32, float64 | bfloat16, bool, float16, integer |
| floor | 4 | floating | bool, complex, integer |
| gather | 37 | bool, complex, floating, integer |  |
| ge | 15 | bool, floating, integer | complex |
| gt | 15 | bool, floating, integer | complex |
| igamma | 6 | floating | bool, complex, integer |
| igammac | 6 | floating | bool, complex, integer |
| imag | 2 | complex | bool, floating, integer |
| integer_pow | 34 | complex, floating, integer | bool |
| iota | 16 | complex, floating, integer | bool |
| is_finite | 4 | floating | bool, complex, integer |
| le | 15 | bool, floating, integer | complex |
| lgamma | 4 | floating | bool, complex, integer |
| log | 6 | complex, floating | bool, integer |
| log1p | 6 | complex, floating | bool, integer |
| lt | 15 | bool, floating, integer | complex |
| lu | 18 | complex, floating | bool, integer |
| max | 29 | bool, complex, floating, integer |  |
| min | 29 | bool, complex, floating, integer |  |
| mul | 16 | complex, floating, integer | bool |
| ne | 17 | bool, complex, floating, integer |  |
| neg | 14 | complex, floating, integer | bool |
| nextafter | 6 | floating | bool, complex, integer |
| or | 11 | bool, integer | complex, floating |
| pad | 90 | bool, complex, floating, integer |  |
| population_count | 8 | integer | bool, complex, floating |
| pow | 10 | complex, floating | bool, integer |
| qr | 60 | complex, floating | bool, integer |
| random_gamma | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| random_split | 5 | uint32 | bool, complex, floating, integer |
| real | 2 | complex | bool, floating, integer |
| reduce_and | 1 | bool | complex, floating, integer |
| reduce_max | 15 | bool, complex, floating, integer |  |
| reduce_min | 15 | bool, complex, floating, integer |  |
| reduce_or | 1 | bool | complex, floating, integer |
| reduce_prod | 14 | complex, floating, integer | bool |
| reduce_sum | 14 | complex, floating, integer | bool |
| reduce_window_add | 33 | complex, floating, integer | bool |
| reduce_window_max | 37 | bool, complex, floating, integer |  |
| reduce_window_min | 15 | bool, complex, floating, integer |  |
| reduce_window_mul | 42 | complex, floating, integer | bool |
| regularized_incomplete_beta | 4 | floating | bool, complex, integer |
| rem | 18 | floating, integer | bool, complex |
| reshape | 19 | bool, complex, floating, integer |  |
| rev | 19 | bool, complex, floating, integer |  |
| round | 7 | floating | bool, complex, integer |
| rsqrt | 6 | complex, floating | bool, integer |
| scatter_add | 14 | complex, floating, integer | bool |
| scatter_max | 15 | bool, complex, floating, integer |  |
| scatter_min | 19 | bool, complex, floating, integer |  |
| scatter_mul | 14 | complex, floating, integer | bool |
| select | 16 | bool, complex, floating, integer |  |
| select_and_gather_add | 15 | floating | bool, complex, integer |
| select_and_scatter_add | 27 | bool, floating, integer | complex |
| shift_left | 10 | integer | bool, complex, floating |
| shift_right_arithmetic | 10 | integer | bool, complex, floating |
| shift_right_logical | 10 | integer | bool, complex, floating |
| sign | 14 | complex, floating, integer | bool |
| sin | 6 | complex, floating | bool, integer |
| sinh | 6 | complex, floating | bool, integer |
| slice | 24 | bool, complex, floating, integer |  |
| sort | 21 | bool, complex, floating, integer |  |
| sqrt | 6 | complex, floating | bool, integer |
| squeeze | 23 | bool, complex, floating, integer |  |
| stop_gradient | 15 | bool, complex, floating, integer |  |
| sub | 16 | complex, floating, integer | bool |
| svd | 120 | complex, floating | bool, integer |
| tan | 6 | complex, floating | bool, integer |
| tanh | 6 | complex, floating | bool, integer |
| tie_in | 15 | bool, complex, floating, integer |  |
| top_k | 15 | bool, floating, integer | complex |
| transpose | 17 | bool, complex, floating, integer |  |
| triangular_solve | 26 | complex, floating | bool, integer |
| xor | 11 | bool, integer | complex, floating |
| zeros_like | 15 | bool, complex, floating, integer |  |

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
excludes the unsupported data types (previous table).

In order to see the actual errors for all entries above look at
the logs of the `test_jax_implemented` from `jax_primitives_coverage_test.py`.


| Affected primitive | Description of limitation | Affected dtypes | Affected devices |
| --- | --- | --- | --- | --- |
|cholesky|unimplemented|float16|cpu, gpu|
|cummax|unimplemented|complex64|tpu|
|cummin|unimplemented|complex64|tpu|
|cumprod|unimplemented|complex64|tpu|
|eig|only supported on CPU in JAX|all|tpu, gpu|
|eig|unimplemented|bfloat16, float16|cpu|
|eigh|complex eigh not supported |complex|tpu|
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
|scatter_add|unimplemented|complex64|tpu|
|scatter_max|unimplemented|complex64|tpu|
|scatter_min|unimplemented|complex64|tpu|
|scatter_mul|unimplemented|complex64|tpu|
|select_and_scatter_add|works only for 2 or more inactive dimensions|all|tpu|
|svd|unimplemented|bfloat16, float16|cpu, gpu|
|svd|complex not implemented|complex|tpu|
|tie_in|requires omnistaging to be disabled|all|cpu, gpu, tpu|
|triangular_solve|unimplemented|float16|gpu|

## Table generation

To regenerate this table run on a CPU machine::

```
  JAX_OUTPUT_LIMITATIONS_DOC=1 JAX_ENABLE_X64=1 python jax/experimental/jax2tf/tests/jax_primitives_coverage_test.py JaxPrimitiveTest.test_generate_primitives_coverage_doc
```
