# Primitives with limited JAX support

*Last generated on: 2021-03-03* (YYYY-MM-DD)

## Supported data types for primitives

We use a set of 2313 test harnesses to test
the implementation of 122 numeric JAX primitives.
We consider a JAX primitive supported for a particular data
type if it is supported on at least one device type.
The following table shows the dtypes at which primitives
are supported in JAX and at which dtypes they are not
supported on any device.
(In reality, this shows for each primitive what dtypes are not covered
by the current test harnesses on any device.)

The set of supported dtypes include 64-bit types
(`float64`, `int64`, `uint64`, `complex128`) only if the
flag `--jax_enable_x64` or the JAX_ENABLE_X64 environment
variable are set.

We use below the following abbreviations for sets of dtypes:

  * `signed` = `int8`, `int16`, `int32`, `int64`
  * `unsigned` = `uint8`, `uint16`, `uint32`, `uint64`
  * `integer` = `signed`, `unsigned`
  * `floating` = `float16`, `bfloat16`, `float32`, `float64`
  * `complex` = `complex64`, `complex128`
  * `inexact` = `floating`, `complex`
  * `all` = `integer`, `inexact`, `bool`

See the comment in `primitive_harness.py` for a description
of test harnesses and their definitions.

Note that automated tests will fail if new limitations appear, but
they won't when limitations are fixed. If you see a limitation that
you think it does not exist anymore, please ask for this file to
be updated.


| Primitive | Total test harnesses | dtypes supported on at least one device | dtypes NOT tested on any device |
| --- | --- | --- | --- | --- |
| abs | 10 | inexact, signed | bool, unsigned |
| acos | 6 | inexact | bool, integer |
| acosh | 6 | inexact | bool, integer |
| add | 16 | inexact, integer | bool |
| add_any | 14 | inexact, integer | bool |
| and | 11 | bool, integer | inexact |
| argmax | 22 | bool, floating, integer | complex |
| argmin | 22 | bool, floating, integer | complex |
| asin | 6 | inexact | bool, integer |
| asinh | 6 | inexact | bool, integer |
| atan | 6 | inexact | bool, integer |
| atan2 | 6 | floating | bool, complex, integer |
| atanh | 6 | inexact | bool, integer |
| bessel_i0e | 4 | floating | bool, complex, integer |
| bessel_i1e | 4 | floating | bool, complex, integer |
| bitcast_convert_type | 41 | all |  |
| broadcast | 17 | all |  |
| broadcast_in_dim | 19 | all |  |
| ceil | 4 | floating | bool, complex, integer |
| cholesky | 30 | inexact | bool, integer |
| clamp | 17 | floating, integer | bool, complex |
| complex | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| concatenate | 17 | all |  |
| conj | 5 | complex, float32, float64 | bfloat16, bool, float16, integer |
| conv_general_dilated | 58 | inexact | bool, integer |
| convert_element_type | 201 | all |  |
| cos | 6 | inexact | bool, integer |
| cosh | 6 | inexact | bool, integer |
| cummax | 17 | inexact, integer | bool |
| cummin | 17 | inexact, integer | bool |
| cumprod | 17 | inexact, integer | bool |
| cumsum | 17 | inexact, integer | bool |
| custom_linear_solve | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| device_put | 16 | all |  |
| digamma | 4 | floating | bool, complex, integer |
| div | 20 | inexact, integer | bool |
| dot_general | 125 | all |  |
| dynamic_slice | 32 | all |  |
| dynamic_update_slice | 21 | all |  |
| eig | 72 | inexact | bool, integer |
| eigh | 36 | inexact | bool, integer |
| eq | 17 | all |  |
| erf | 4 | floating | bool, complex, integer |
| erf_inv | 4 | floating | bool, complex, integer |
| erfc | 4 | floating | bool, complex, integer |
| exp | 6 | inexact | bool, integer |
| expm1 | 6 | inexact | bool, integer |
| fft | 20 | complex, float32, float64 | bfloat16, bool, float16, integer |
| floor | 4 | floating | bool, complex, integer |
| gather | 37 | all |  |
| ge | 15 | bool, floating, integer | complex |
| gt | 15 | bool, floating, integer | complex |
| igamma | 6 | floating | bool, complex, integer |
| igammac | 6 | floating | bool, complex, integer |
| imag | 2 | complex | bool, floating, integer |
| integer_pow | 34 | inexact, integer | bool |
| iota | 16 | inexact, integer | bool |
| is_finite | 4 | floating | bool, complex, integer |
| le | 15 | bool, floating, integer | complex |
| lgamma | 4 | floating | bool, complex, integer |
| log | 6 | inexact | bool, integer |
| log1p | 6 | inexact | bool, integer |
| lt | 15 | bool, floating, integer | complex |
| lu | 18 | inexact | bool, integer |
| max | 29 | all |  |
| min | 29 | all |  |
| mul | 16 | inexact, integer | bool |
| ne | 17 | all |  |
| neg | 14 | inexact, integer | bool |
| nextafter | 6 | floating | bool, complex, integer |
| or | 11 | bool, integer | inexact |
| pad | 90 | all |  |
| population_count | 8 | integer | bool, inexact |
| pow | 10 | inexact | bool, integer |
| qr | 60 | inexact | bool, integer |
| random_gamma | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| random_split | 5 | uint32 | all |
| real | 2 | complex | bool, floating, integer |
| reduce_and | 1 | bool | inexact, integer |
| reduce_max | 15 | all |  |
| reduce_min | 15 | all |  |
| reduce_or | 1 | bool | inexact, integer |
| reduce_prod | 14 | inexact, integer | bool |
| reduce_sum | 14 | inexact, integer | bool |
| reduce_window_add | 33 | inexact, integer | bool |
| reduce_window_max | 37 | all |  |
| reduce_window_min | 15 | all |  |
| reduce_window_mul | 42 | inexact, integer | bool |
| regularized_incomplete_beta | 4 | floating | bool, complex, integer |
| rem | 18 | floating, integer | bool, complex |
| reshape | 19 | all |  |
| rev | 19 | all |  |
| round | 7 | floating | bool, complex, integer |
| rsqrt | 6 | inexact | bool, integer |
| scatter_add | 14 | inexact, integer | bool |
| scatter_max | 15 | all |  |
| scatter_min | 19 | all |  |
| scatter_mul | 14 | inexact, integer | bool |
| select | 16 | all |  |
| select_and_gather_add | 15 | floating | bool, complex, integer |
| select_and_scatter_add | 27 | bool, floating, integer | complex |
| shift_left | 10 | integer | bool, inexact |
| shift_right_arithmetic | 10 | integer | bool, inexact |
| shift_right_logical | 10 | integer | bool, inexact |
| sign | 14 | inexact, integer | bool |
| sin | 6 | inexact | bool, integer |
| sinh | 6 | inexact | bool, integer |
| slice | 24 | all |  |
| sort | 29 | all |  |
| sqrt | 6 | inexact | bool, integer |
| squeeze | 23 | all |  |
| stop_gradient | 15 | all |  |
| sub | 16 | inexact, integer | bool |
| svd | 120 | inexact | bool, integer |
| tan | 6 | inexact | bool, integer |
| tanh | 6 | inexact | bool, integer |
| tie_in | 15 | all |  |
| top_k | 15 | bool, floating, integer | complex |
| transpose | 17 | all |  |
| triangular_solve | 26 | inexact | bool, integer |
| xor | 11 | bool, integer | inexact |
| zeros_like | 15 | all |  |

## Partially implemented data types for primitives

In some cases, a primitive is supported in JAX at a given data type but
it may be missing implementations for some of the devices.
For example, the eigen decomposition (`lax.eig`) is implemented
in JAX using custom kernels only on CPU and GPU. There is no
TPU implementation. In other cases, there are either bugs or
not-yet-implemented cases in the XLA compiler for different
devices.

The following table shows which of the supported data types
are partially implemented for each primitive. This table already
excludes the unsupported data types (previous table).

In order to see the actual errors for all entries look through
the logs of the `test_jax_implemented` from `jax_primitives_coverage_test.py`
and search for "limitation".


| Affected primitive | Description of limitation | Affected dtypes | Affected devices |
| --- | --- | --- | --- | --- |
|cholesky|unimplemented|float16|cpu, gpu|
|cummax|unimplemented|complex64|tpu|
|cummin|unimplemented|complex64|tpu|
|cumprod|unimplemented|complex64|tpu|
|eig|only supported on CPU in JAX|all|tpu, gpu|
|eig|unimplemented|bfloat16, float16|cpu|
|eigh|complex eigh not supported |complex|tpu|
|eigh|unimplemented|bfloat16, float16|cpu, gpu|
|lu|unimplemented|bfloat16, float16|cpu, gpu, tpu|
|qr|unimplemented|bfloat16, float16|cpu, gpu|
|reduce_window_max|unimplemented in XLA|complex64|tpu|
|reduce_window_min|unimplemented in XLA|complex64|tpu|
|reduce_window_mul|unimplemented in XLA|complex64|tpu|
|scatter_max|unimplemented|complex64|tpu|
|scatter_min|unimplemented|complex64|tpu|
|select_and_scatter_add|works only for 2 or more inactive dimensions|all|tpu|
|svd|complex not implemented. Works in JAX for CPU and GPU with custom kernels|complex|tpu|
|svd|unimplemented|bfloat16, float16|cpu, gpu|
|tie_in|requires omnistaging to be disabled|all|cpu, gpu, tpu|
|triangular_solve|unimplemented|float16|gpu|

## Table generation

To regenerate this table run (on a CPU machine):

```
  JAX_OUTPUT_LIMITATIONS_DOC=1 JAX_ENABLE_X64=1 python jax/experimental/jax2tf/tests/jax_primitives_coverage_test.py JaxPrimitiveTest.test_generate_primitives_coverage_doc
```
