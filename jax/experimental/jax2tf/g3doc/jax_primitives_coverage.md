# Primitives with limited JAX support

*Last generated on: 2023-07-31* (YYYY-MM-DD)

## Supported data types for primitives

We use a set of 7554 test harnesses to test
the implementation of 133 numeric JAX primitives.
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
| --- | --- | --- | --- |
| abs | 10 | inexact, signed | bool, unsigned |
| acos | 6 | inexact | bool, integer |
| acosh | 6 | inexact | bool, integer |
| add | 16 | inexact, integer | bool |
| add_any | 14 | inexact, integer | bool |
| and | 11 | bool, integer | inexact |
| approx_top_k | 24 | floating | bool, complex, integer |
| argmax | 64 | bool, floating, integer | complex |
| argmin | 64 | bool, floating, integer | complex |
| asin | 6 | inexact | bool, integer |
| asinh | 6 | inexact | bool, integer |
| atan | 6 | inexact | bool, integer |
| atan2 | 6 | floating | bool, complex, integer |
| atanh | 6 | inexact | bool, integer |
| bessel_i0e | 4 | floating | bool, complex, integer |
| bessel_i1e | 4 | floating | bool, complex, integer |
| bitcast_convert_type | 41 | all |  |
| broadcast_in_dim | 19 | all |  |
| cbrt | 4 | floating | bool, complex, integer |
| ceil | 4 | floating | bool, complex, integer |
| cholesky | 30 | inexact | bool, integer |
| clamp | 20 | all |  |
| complex | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| concatenate | 17 | all |  |
| conj | 5 | complex, float32, float64 | bfloat16, bool, float16, integer |
| conv_general_dilated | 132 | inexact, signed | bool, unsigned |
| convert_element_type | 201 | all |  |
| cos | 6 | inexact | bool, integer |
| cosh | 6 | inexact | bool, integer |
| cumlogsumexp | 12 | float16, float32, float64 | bfloat16, bool, complex, integer |
| cummax | 34 | inexact, integer | bool |
| cummin | 34 | inexact, integer | bool |
| cumprod | 34 | inexact, integer | bool |
| cumsum | 34 | inexact, integer | bool |
| custom_linear_solve | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| device_put | 16 | all |  |
| digamma | 4 | floating | bool, complex, integer |
| div | 20 | inexact, integer | bool |
| dot_general | 400 | all |  |
| dynamic_slice | 68 | all |  |
| dynamic_update_slice | 46 | all |  |
| eig | 72 | inexact | bool, integer |
| eigh | 36 | inexact | bool, integer |
| eq | 17 | all |  |
| erf | 4 | floating | bool, complex, integer |
| erf_inv | 4 | floating | bool, complex, integer |
| erfc | 4 | floating | bool, complex, integer |
| exp | 6 | inexact | bool, integer |
| expm1 | 6 | inexact | bool, integer |
| fft | 32 | complex, float32, float64 | bfloat16, bool, float16, integer |
| floor | 4 | floating | bool, complex, integer |
| gather | 164 | all |  |
| ge | 17 | all |  |
| gt | 17 | all |  |
| igamma | 6 | floating | bool, complex, integer |
| igammac | 6 | floating | bool, complex, integer |
| imag | 2 | complex | bool, floating, integer |
| integer_pow | 108 | inexact, integer | bool |
| iota | 16 | inexact, integer | bool |
| iota_2x32_shape | 3 | uint32 | bool, inexact, signed, uint16, uint64, uint8 |
| is_finite | 4 | floating | bool, complex, integer |
| le | 17 | all |  |
| lgamma | 4 | floating | bool, complex, integer |
| log | 6 | inexact | bool, integer |
| log1p | 6 | inexact | bool, integer |
| logistic | 6 | inexact | bool, integer |
| lt | 17 | all |  |
| lu | 18 | inexact | bool, integer |
| max | 27 | all |  |
| min | 27 | all |  |
| mul | 16 | inexact, integer | bool |
| ne | 17 | all |  |
| neg | 14 | inexact, integer | bool |
| nextafter | 6 | floating | bool, complex, integer |
| or | 11 | bool, integer | inexact |
| pad | 180 | all |  |
| population_count | 8 | integer | bool, inexact |
| pow | 10 | inexact | bool, integer |
| qr | 60 | inexact | bool, integer |
| random_categorical | 12 | floating | bool, complex, integer |
| random_gamma | 4 | float32, float64 | bfloat16, bool, complex, float16, integer |
| random_randint | 12 | signed | bool, inexact, unsigned |
| random_split | 5 | uint32 | all |
| random_uniform | 12 | floating | bool, complex, integer |
| real | 2 | complex | bool, floating, integer |
| reduce | 33 | all |  |
| reduce_and | 1 | bool | inexact, integer |
| reduce_max | 15 | all |  |
| reduce_min | 15 | all |  |
| reduce_or | 1 | bool | inexact, integer |
| reduce_precision | 32 | floating | bool, complex, integer |
| reduce_prod | 14 | inexact, integer | bool |
| reduce_sum | 14 | inexact, integer | bool |
| reduce_window_add | 50 | inexact, integer | bool |
| reduce_window_max | 66 | all |  |
| reduce_window_min | 27 | all |  |
| reduce_window_mul | 42 | inexact, integer | bool |
| regularized_incomplete_beta | 4 | floating | bool, complex, integer |
| rem | 18 | floating, integer | bool, complex |
| reshape | 19 | all |  |
| rev | 19 | all |  |
| rng_bit_generator | 36 | uint32, uint64 | bool, inexact, signed, uint16, uint8 |
| round | 6 | floating | bool, complex, integer |
| rsqrt | 6 | inexact | bool, integer |
| scatter | 645 | all |  |
| scatter_add | 885 | all |  |
| scatter_max | 885 | all |  |
| scatter_min | 888 | all |  |
| scatter_mul | 885 | all |  |
| select_and_gather_add | 15 | floating | bool, complex, integer |
| select_and_scatter_add | 27 | bool, floating, integer | complex |
| select_n | 32 | all |  |
| shift_left | 10 | integer | bool, inexact |
| shift_right_arithmetic | 10 | integer | bool, inexact |
| shift_right_logical | 10 | integer | bool, inexact |
| sign | 28 | inexact, integer | bool |
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
| top_k | 15 | bool, floating, integer | complex |
| transpose | 17 | all |  |
| triangular_solve | 26 | inexact | bool, integer |
| tridiagonal_solve | 2 | float32, float64 | bfloat16, bool, complex, float16, integer |
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
| --- | --- | --- | --- |
|cholesky|unimplemented|float16|cpu, gpu|
|clamp|unimplemented|bool, complex|cpu, gpu, tpu|
|conv_general_dilated|preferred_element_type not implemented for integers|signed|gpu|
|dot_general|preferred_element_type must be floating for integer dtype|integer|gpu|
|dot_general|preferred_element_type must match dtype for floating point|inexact|gpu|
|eig|only supported on CPU in JAX|all|tpu, gpu|
|eig|unimplemented|bfloat16, float16|cpu|
|eigh|unimplemented|bfloat16, float16|cpu, gpu|
|lu|unimplemented|bfloat16, float16|cpu, gpu, tpu|
|qr|unimplemented|bfloat16, float16|cpu, gpu|
|scatter_add|unimplemented|bool|cpu, gpu, tpu|
|scatter_mul|unimplemented|bool|cpu, gpu, tpu|
|select_and_scatter_add|works only for 2 or more inactive dimensions|all|tpu|
|svd|unimplemented|bfloat16, float16|cpu, gpu|
|triangular_solve|unimplemented|float16|gpu|

## Table generation

To regenerate this table run (on a CPU machine):

```
  JAX_OUTPUT_LIMITATIONS_DOC=1 JAX_ENABLE_X64=1 python jax/experimental/jax2tf/tests/jax_primitives_coverage_test.py JaxPrimitiveTest.test_generate_primitives_coverage_doc
```
