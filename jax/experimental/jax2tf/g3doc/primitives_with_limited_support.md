# Primitives with limited support

*Last generated on (YYYY-MM-DD): 2020-12-28*

We do not yet have support for `pmap` (with its collective primitives),
nor for `sharded_jit` (SPMD partitioning).

A few JAX primitives are converted to TF ops that have incomplete coverage
for data types on different kinds of devices. The most [up-to-date list of
limitations is generated automatically](#generated-summary-of-primitives-with-limited-support)
by the jax2tf coverage tests.
More detailed information can be found in the
[source code of categorize](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/tests/correctness_stats.py)
for some cases.

Additionally, some primitives have numerical differences between JAX and TF in some corner cases:

  * `top_k` is implemented using `tf.math.top_k` which has limited functionality. There are
  silent errors when the array contains `inf` and `NaN` (they are sorted differently in TF vs. XLA).

  * `digamma` is converted to `tf.math.digamma` with the following limitations (all
  at the singularity points 0 and -1):
  At singularity points with dtype float32 JAX returns `NaN` and TF returns `inf`.

  * `igamma` and `igammac` are implemented using `tf.math.igamma` and `tf.math.igammac`,
  with the following limitations: At undefined points (both arguments 0 for `igamma` and
  both arguments less or equal to 0 for `igammac`) JAX returns `NaN` and TF returns 0 or
  JAX returns 1 and TF returns `NaN`.

  * `integer_pow` does not have the right overflow behavior in compiled mode
  for int32 and int64 dtypes. Additionally, the overflow behavior for floating
  point and complex numbers produces `NaN`s and `+inf`/`-inf` differently in
  JAX and TF.

  * `erf_inv` is converted to `tf.math.erfinv` with discrepancies at
  undefined points (< -1 or > 1): At undefined points with dtype float32 JAX
  returns `NaN` and TF returns `+inf` or `-inf`.

  * QR decomposition is implemented using `tf.linalg.qr` with
  the following limitations: QR for complex numbers will not work when using XLA
  because it is not implemented in XLA. (It works in JAX on CPU and GPU
  using custom calls to Lapack and Cusolver; it does not work in JAX on TPU.)
  It is likely that on CPU and GPU the TF version is slower than the JAX version.

  * `svd` is implemented using `tf.linalg.svd` and `tf.linalg.adjoint`:
  SVD weirdly works for bfloat16 on TPU for JAX, but fails for TF (this
  is related to a more general bfloat16 type casting problem).
  The conversion does not work for complex types because XLA does
  not implement support for complexes (once again, it works with JAX
  because the implementation there uses custom calls to cusolver).

  * `select_and_gather_add` is implemented using `XlaReduceWindow`:
  This JAX primitive is not exposed directly in the JAX API but
  arises from JVP of `lax.reduce_window` for reducers `lax.max` or
  `lax.min`. It also arises from second-order VJP of the same.

  * `select_and_scatter`: We decided to explicitly throw an error
  when trying to translate this operation. The reason is that
  the operation is not exposed in lax, and here mostly
  for completion purposes. It is not expected that this
  operation will ever have a use case.

  * `acos`, `asin`, `atan`, and their hyperbolic counterparts
  `acosh`, `asinh`, `atanh` return equally valid but possibly
  different results when the parameter has a complex dtype.

  * `min` and `max` return different values when one of the elements
  being compared is NaN. JAX always returns NaN, while TF returns
  the value NaN is compared with.

  * `cholesky` returns potentially different values in the strictly
  upper triangular part of the result. This does not matter for
  correctness, because this part of the matrix is not considered in
  the result.

  * `eig` and `eigh` return eigenvalues and eigenvectors in a
  potentially different order. The eigenvectors may also be
  different, but equally valid.

  * `lu` may return different but equally valid results when the LU
  decomposition is not unique.


## Generated summary of primitives with limited support in Tensorflow

The following JAX primitives are converted to Tensorflow but the result of the
conversion may trigger runtime errors when run on certain devices and with
certain data types.

This table is organized by JAX primitive, but the actual errors described
in the table are for the Tensorflow ops to which the primitive is converted to.
In general, each JAX primitive is mapped
to one Tensorflow op, e.g., `sin` is mapped to `tf.math.sin`.

The errors apply only for certain devices and compilation modes ("eager",
"graph", and "compiled"). In general, "eager" and "graph" mode share the same errors.
On TPU only "compiled" mode is ever used.


This table only shows errors for cases that are working in JAX (see [separate
list of unsupported or partially-supported primitives](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/g3doc/jax_primitives_coverage.md)

We use the following abbreviations for sets of dtypes:

  * `signed` = `int8`, `int16`, `int32`, `int64`
  * `unsigned` = `uint8`, `uint16`, `uint32`, `uint64`
  * `integer` = `signed`, `nsigned`
  * `floating` = `float16`, `bfloat16`, `float32`, `float64`
  * `complex` = `complex64`, `complex128`
  * `inexact` = `floating`, `complex`
  * `all` = `integer`, `inexact`, `bool`


| Affected primitive | Description of limitation | Affected dtypes | Affected devices | Affected compilation modes |
| --- | --- | --- | --- | --- | ---|
|acos|TF error: op not defined for dtype|bfloat16, complex64, float16|cpu, gpu|eager, graph|
|acos|TF error: op not defined for dtype|complex128|cpu, gpu|eager, graph|
|acosh|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|add|TF error: op not defined for dtype|uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|add|TF error: op not defined for dtype|uint64|cpu, gpu|compiled, eager, graph|
|add_any|TF error: op not defined for dtype|uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|asin|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|asin|TF error: op not defined for dtype|complex|cpu, gpu, tpu|compiled, eager, graph|
|asinh|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|asinh|TF error: op not defined for dtype|complex|cpu, gpu, tpu|compiled, eager, graph|
|atan|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|atan|TF error: op not defined for dtype|complex|cpu, gpu, tpu|compiled, eager, graph|
|atan2|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|atanh|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|bessel_i0e|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|bessel_i1e|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|bitcast_convert_type|TF error: op not defined for dtype|bool|cpu, gpu, tpu|compiled, eager, graph|
|cholesky|TF error: function not compilable|complex|cpu, gpu|compiled|
|cholesky|TF error: op not defined for dtype|complex|tpu|compiled|
|clamp|TF error: op not defined for dtype|int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|conv_general_dilated|TF error: jax2tf BUG: batch_group_count > 1 not yet converted||cpu, gpu, tpu|compiled, eager, graph|
|conv_general_dilated|TF error: XLA bug in the HLO -> LLVM IR lowering|complex|cpu, gpu|compiled, eager, graph|
|cosh|TF error: op not defined for dtype|float16|cpu, gpu|eager, graph|
|cummax|TF error: op not defined for dtype|complex128, uint64|cpu, gpu|compiled, eager, graph|
|cummax|TF error: op not defined for dtype|complex64, int8, uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|cummin|TF error: op not defined for dtype|complex128, uint64|cpu, gpu|compiled, eager, graph|
|cummin|TF error: op not defined for dtype|complex64, int8, uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|cumprod|TF error: op not defined for dtype|uint64|cpu, gpu|compiled, eager, graph|
|cumprod|TF error: op not defined for dtype|uint32|cpu, gpu, tpu|compiled, eager, graph|
|cumsum|TF error: op not defined for dtype|uint64|cpu, gpu|compiled, eager, graph|
|cumsum|TF error: op not defined for dtype|complex64|tpu|compiled|
|cumsum|TF error: op not defined for dtype|uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|digamma|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|div|TF error: op not defined for dtype|int16, int8, unsigned|cpu, gpu, tpu|compiled, eager, graph|
|div|TF error: TF integer division fails if divisor contains 0; JAX returns NaN|integer|cpu, gpu, tpu|compiled, eager, graph|
|dot_general|TF error: op not defined for dtype|bool, int16, int8, unsigned|cpu, gpu, tpu|compiled, eager, graph|
|dot_general|TF error: op not defined for dtype|int64|cpu, gpu|compiled|
|eig|TF error: function not compilable||cpu, gpu, tpu|compiled|
|eig|TF error: TF Conversion of eig is not implemented when both compute_left_eigenvectors and compute_right_eigenvectors are set to True||cpu, gpu, tpu|compiled, eager, graph|
|eigh|TF error: function not compilable|complex|cpu, gpu, tpu|compiled|
|erf|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|erf_inv|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu|eager, graph|
|erfc|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|fft|TF error: TF function not compileable|complex128, float64|cpu, gpu|compiled|
|ge|TF error: op not defined for dtype|bool|cpu, gpu, tpu|compiled, eager, graph|
|ge|TF error: op not defined for dtype|uint16, uint32|cpu, gpu|eager, graph|
|ge|TF error: op not defined for dtype|uint64|cpu, gpu|eager, graph|
|gt|TF error: op not defined for dtype|bool|cpu, gpu, tpu|compiled, eager, graph|
|gt|TF error: op not defined for dtype|uint16, uint32|cpu, gpu|eager, graph|
|gt|TF error: op not defined for dtype|uint64|cpu, gpu|eager, graph|
|integer_pow|TF error: op not defined for dtype|int16, int8, unsigned|cpu, gpu, tpu|compiled, eager, graph|
|le|TF error: op not defined for dtype|bool|cpu, gpu, tpu|compiled, eager, graph|
|le|TF error: op not defined for dtype|uint16, uint32|cpu, gpu|eager, graph|
|le|TF error: op not defined for dtype|uint64|cpu, gpu|eager, graph|
|lgamma|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|lt|TF error: op not defined for dtype|bool|cpu, gpu, tpu|compiled, eager, graph|
|lt|TF error: op not defined for dtype|uint16, uint32|cpu, gpu|eager, graph|
|lt|TF error: op not defined for dtype|uint64|cpu, gpu|eager, graph|
|lu|TF error: op not defined for dtype|complex64|tpu|compiled|
|max|TF error: op not defined for dtype|bool, complex64, int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|max|TF error: op not defined for dtype|complex128|cpu, gpu|compiled, eager, graph|
|min|TF error: op not defined for dtype|bool, complex64, int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|min|TF error: op not defined for dtype|complex128|cpu, gpu|compiled, eager, graph|
|mul|TF error: op not defined for dtype|uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|neg|TF error: op not defined for dtype|unsigned|cpu, gpu, tpu|compiled, eager, graph|
|nextafter|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu, tpu|compiled, eager, graph|
|population_count|TF error: op not defined for dtype|uint32, uint64|cpu, gpu|eager, graph|
|qr|TF error: op not defined for dtype|bfloat16|tpu|compiled|
|reduce_max|TF error: op not defined for dtype|complex64|cpu, gpu, tpu|compiled, eager, graph|
|reduce_max|TF error: op not defined for dtype|complex128|cpu, gpu, tpu|compiled, eager, graph|
|reduce_min|TF error: op not defined for dtype|complex64|cpu, gpu, tpu|compiled, eager, graph|
|reduce_min|TF error: op not defined for dtype|complex128|cpu, gpu, tpu|compiled, eager, graph|
|reduce_window_add|TF error: op not defined for dtype|uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|reduce_window_add|TF error: op not defined for dtype|complex64|tpu|compiled, eager, graph|
|reduce_window_add|TF error: op not defined for dtype|uint64|cpu, gpu|compiled, eager, graph|
|reduce_window_max|TF error: op not defined for dtype|bool, complex64, uint32|cpu, gpu, tpu|compiled, eager, graph|
|reduce_window_max|TF error: op not defined for dtype|complex128, uint64|cpu, gpu|compiled, eager, graph|
|reduce_window_max|TF error: TF kernel missing, except when the initial_value is the minimum for the dtype|int8, uint16|cpu, gpu, tpu|compiled, eager, graph|
|reduce_window_min|TF error: op not defined for dtype|bool, complex64, int8, uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|reduce_window_min|TF error: op not defined for dtype|complex128, uint64|cpu, gpu|compiled, eager, graph|
|reduce_window_mul|TF error: op not defined for dtype|uint32|cpu, gpu, tpu|compiled, eager, graph|
|reduce_window_mul|TF error: op not defined for dtype|uint64|cpu, gpu|compiled, eager, graph|
|regularized_incomplete_beta|TF error: op not defined for dtype|bfloat16, float16|cpu, gpu, tpu|compiled, eager, graph|
|rem|TF error: op not defined for dtype|int16, int8, unsigned|cpu, gpu, tpu|compiled, eager, graph|
|rem|TF error: TF integer division fails if divisor contains 0; JAX returns NaN|integer|cpu, gpu, tpu|compiled, eager, graph|
|rem|TF error: op not defined for dtype|float16|cpu, gpu, tpu|eager, graph|
|rev|TF error: op not defined for dtype|uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|round|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|rsqrt|TF error: op not defined for dtype|bfloat16|cpu, gpu|eager, graph|
|scatter_add|TF error: op not defined for dtype|bool, complex64, int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|scatter_max|TF error: op not defined for dtype|bool, complex, int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|scatter_min|TF error: op not defined for dtype|bool, complex, int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|scatter_mul|TF error: op not defined for dtype|bool, complex64, int8, uint16, uint32, uint64|cpu, gpu, tpu|compiled, eager, graph|
|select_and_gather_add|TF error: op not defined for dtype|float32|tpu|compiled|
|select_and_gather_add|TF error: jax2tf unimplemented|float64|cpu, gpu|compiled, eager, graph|
|select_and_scatter_add|TF error: op not defined for dtype|uint16, uint32|cpu, gpu, tpu|compiled, eager, graph|
|select_and_scatter_add|TF error: op not defined for dtype|uint64|cpu, gpu|compiled, eager, graph|
|sign|TF error: op not defined for dtype|int16, int8, unsigned|cpu, gpu, tpu|compiled, eager, graph|
|sinh|TF error: op not defined for dtype|float16|cpu, gpu|eager, graph|
|sort|TF error: op not defined for dtype|complex|cpu, gpu|eager, graph|
|sort|TF error: TODO: XlaSort does not support more than 2 arrays||cpu, gpu, tpu|compiled, eager, graph|
|sort|TF error: TODO: XlaSort does not support sorting axis||cpu, gpu, tpu|compiled, eager, graph|
|sub|TF error: op not defined for dtype|uint64|cpu, gpu, tpu|compiled, eager, graph|
|svd|TF error: function not compilable|complex|cpu, gpu|compiled|
|svd|TF error: op not defined for dtype|bfloat16|tpu|compiled|
|top_k|TF error: op not defined for dtype|int64, uint64|cpu, gpu|compiled|
|triangular_solve|TF error: op not defined for dtype|bfloat16|cpu, gpu, tpu|compiled, eager, graph|
|triangular_solve|TF error: op not defined for dtype|float16|cpu, gpu, tpu|eager, graph|

## Updating the documentation

To update this documentation, run the following command:

```
  JAX_ENABLE_X64=1 JAX_OUTPUT_LIMITATIONS_DOC=1 python jax/experimental/jax2tf/tests/primitives_test.py JaxPrimitiveTest.test_generate_limitations_doc
```
