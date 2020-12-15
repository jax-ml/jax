# Size of generated HLO IR by test name

| Test name | JAX | TF compiled | Notice |
| --- | --- | --- | --- |
| `test_add_jaxvals_dtypes_lhs=bfloat16[2]_rhs=bfloat16[2]` | 10 | 9 |  |
| `test_add_jaxvals_dtypes_lhs=complex128[2]_rhs=complex128[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=complex64[2]_rhs=complex64[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=float16[2]_rhs=float16[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=float32[2]_rhs=float32[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=float64[2]_rhs=float64[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=int16[2]_rhs=int16[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=int32[2]_rhs=int32[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=int64[2]_rhs=int64[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=int8[2]_rhs=int8[2]` | 7 | 6 |  |
| `test_add_jaxvals_dtypes_lhs=uint16[2]_rhs=uint16[2]` | 7 | 0 |  |
| `test_add_jaxvals_dtypes_lhs=uint32[2]_rhs=uint32[2]` | 7 | 0 |  |
| `test_add_jaxvals_dtypes_lhs=uint64[2]_rhs=uint64[2]` | 7 | 0 |  |
| `test_add_jaxvals_dtypes_lhs=uint8[2]_rhs=uint8[2]` | 7 | 6 |  |
| `test_add_mul_fun=add_bfloat16` | 10 | 9 |  |
| `test_add_mul_fun=add_bounds_bfloat16` | 10 | 9 |  |
| `test_add_mul_fun=add_bounds_complex128` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_complex64` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_float16` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_float32` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_float64` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_int16` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_int32` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_int64` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_int8` | 7 | 6 |  |
| `test_add_mul_fun=add_bounds_uint16` | 7 | 0 |  |
| `test_add_mul_fun=add_bounds_uint32` | 7 | 0 |  |
| `test_add_mul_fun=add_bounds_uint64` | 7 | 0 |  |
| `test_add_mul_fun=add_bounds_uint8` | 7 | 6 |  |
| `test_add_mul_fun=add_complex128` | 7 | 6 |  |
| `test_add_mul_fun=add_complex64` | 7 | 6 |  |
| `test_add_mul_fun=add_float16` | 7 | 6 |  |
| `test_add_mul_fun=add_float32` | 7 | 6 |  |
| `test_add_mul_fun=add_float64` | 7 | 6 |  |
| `test_add_mul_fun=add_int16` | 7 | 6 |  |
| `test_add_mul_fun=add_int32` | 7 | 6 |  |
| `test_add_mul_fun=add_int64` | 7 | 6 |  |
| `test_add_mul_fun=add_int8` | 7 | 6 |  |
| `test_add_mul_fun=add_uint16` | 7 | 0 |  |
| `test_add_mul_fun=add_uint32` | 7 | 0 |  |
| `test_add_mul_fun=add_uint64` | 7 | 0 |  |
| `test_add_mul_fun=add_uint8` | 7 | 6 |  |
| `test_add_mul_fun=mul_bfloat16` | 10 | 9 |  |
| `test_add_mul_fun=mul_bounds_bfloat16` | 10 | 9 |  |
| `test_add_mul_fun=mul_bounds_complex128` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_complex64` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_float16` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_float32` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_float64` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_int16` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_int32` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_int64` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_int8` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_uint16` | 7 | 6 |  |
| `test_add_mul_fun=mul_bounds_uint32` | 7 | 0 |  |
| `test_add_mul_fun=mul_bounds_uint64` | 7 | 0 |  |
| `test_add_mul_fun=mul_bounds_uint8` | 7 | 6 |  |
| `test_add_mul_fun=mul_complex128` | 7 | 6 |  |
| `test_add_mul_fun=mul_complex64` | 7 | 6 |  |
| `test_add_mul_fun=mul_float16` | 7 | 6 |  |
| `test_add_mul_fun=mul_float32` | 7 | 6 |  |
| `test_add_mul_fun=mul_float64` | 7 | 6 |  |
| `test_add_mul_fun=mul_int16` | 7 | 6 |  |
| `test_add_mul_fun=mul_int32` | 7 | 6 |  |
| `test_add_mul_fun=mul_int64` | 7 | 6 |  |
| `test_add_mul_fun=mul_int8` | 7 | 6 |  |
| `test_add_mul_fun=mul_uint16` | 7 | 6 |  |
| `test_add_mul_fun=mul_uint32` | 7 | 0 |  |
| `test_add_mul_fun=mul_uint64` | 7 | 0 |  |
| `test_add_mul_fun=mul_uint8` | 7 | 6 |  |
| `test_argminmax_axes_prim=argmax_shape=float32[18,12]_axes=(1,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_axes_prim=argmin_shape=float32[18,12]_axes=(1,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=bfloat16[15]_axes=(0,)_indexdtype=int32` | 50 | 48 |  |
| `test_argminmax_dtypes_prim=argmax_shape=bool[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=float16[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=float64[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=int16[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=int32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=int64[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=int8[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=uint16[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=uint32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=uint64[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmax_shape=uint8[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=bfloat16[15]_axes=(0,)_indexdtype=int32` | 50 | 48 |  |
| `test_argminmax_dtypes_prim=argmin_shape=bool[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=float16[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=float64[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=int16[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=int32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=int64[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=int8[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=uint16[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=uint32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=uint64[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_dtypes_prim=argmin_shape=uint8[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=int16` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=int64` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=int8` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=uint16` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=uint32` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=uint64` | 41 | 45 |  |
| `test_argminmax_index_dtype_prim=argmax_shape=float32[15]_axes=(0,)_indexdtype=uint8` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=int16` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=int32` | 41 | 39 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=int64` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=int8` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=uint16` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=uint32` | 41 | 40 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=uint64` | 41 | 45 |  |
| `test_argminmax_index_dtype_prim=argmin_shape=float32[15]_axes=(0,)_indexdtype=uint8` | 41 | 40 |  |
| `test_betainc_bfloat16` | 493 | 0 |  |
| `test_betainc_float16` | 493 | 0 |  |
| `test_betainc_float32` | 469 | 468 |  |
| `test_betainc_float64` | 471 | 470 |  |
| `test_binary_elementwise_add_bfloat16` | 10 | 9 |  |
| `test_binary_elementwise_add_float16` | 7 | 6 |  |
| `test_binary_elementwise_add_float32` | 7 | 6 |  |
| `test_binary_elementwise_add_float64` | 7 | 6 |  |
| `test_binary_elementwise_atan2_bfloat16` | 10 | 9 |  |
| `test_binary_elementwise_atan2_float16` | 7 | 6 |  |
| `test_binary_elementwise_atan2_float32` | 7 | 6 |  |
| `test_binary_elementwise_atan2_float64` | 7 | 6 |  |
| `test_binary_elementwise_div_bfloat16` | 10 | 9 |  |
| `test_binary_elementwise_div_float16` | 7 | 6 |  |
| `test_binary_elementwise_div_float32` | 7 | 6 |  |
| `test_binary_elementwise_div_float64` | 7 | 6 |  |
| `test_binary_elementwise_igamma_float32` | 911 | 910 |  |
| `test_binary_elementwise_igamma_float64` | 913 | 912 |  |
| `test_binary_elementwise_igammac_float32` | 889 | 888 |  |
| `test_binary_elementwise_igammac_float64` | 891 | 890 |  |
| `test_binary_elementwise_logical_bitwise_and_bool` | 7 | 16 | **LARGE DIFF** |
| `test_binary_elementwise_logical_bitwise_and_int16` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_int32` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_int64` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_int8` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_uint16` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_uint32` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_uint64` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_and_uint8` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_bool` | 7 | 16 | **LARGE DIFF** |
| `test_binary_elementwise_logical_bitwise_or_int16` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_int32` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_int64` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_int8` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_uint16` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_uint32` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_uint64` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_or_uint8` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_bool` | 7 | 16 | **LARGE DIFF** |
| `test_binary_elementwise_logical_bitwise_xor_int16` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_int32` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_int64` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_int8` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_uint16` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_uint32` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_uint64` | 7 | 6 |  |
| `test_binary_elementwise_logical_bitwise_xor_uint8` | 7 | 6 |  |
| `test_binary_elementwise_logical_shift_left_int16` | 7 | 19 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_int32` | 7 | 19 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_int64` | 7 | 19 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_int8` | 7 | 19 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_uint16` | 7 | 22 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_uint32` | 7 | 22 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_uint64` | 7 | 22 | **LARGE DIFF** |
| `test_binary_elementwise_logical_shift_left_uint8` | 7 | 22 | **LARGE DIFF** |
| `test_binary_elementwise_max_bfloat16` | 10 | 9 |  |
| `test_binary_elementwise_max_float16` | 7 | 6 |  |
| `test_binary_elementwise_max_float32` | 7 | 6 |  |
| `test_binary_elementwise_max_float64` | 7 | 6 |  |
| `test_binary_elementwise_min_bfloat16` | 10 | 9 |  |
| `test_binary_elementwise_min_float16` | 7 | 6 |  |
| `test_binary_elementwise_min_float32` | 7 | 6 |  |
| `test_binary_elementwise_min_float64` | 7 | 6 |  |
| `test_binary_elementwise_nextafter_bfloat16` | 50 | 0 |  |
| `test_binary_elementwise_nextafter_float16` | 46 | 0 |  |
| `test_binary_elementwise_nextafter_float32` | 46 | 45 |  |
| `test_binary_elementwise_nextafter_float64` | 46 | 45 |  |
| `test_binary_elementwise_rem_bfloat16` | 10 | 59 | **LARGE DIFF** |
| `test_binary_elementwise_rem_float16` | 7 | 31 | **LARGE DIFF** |
| `test_binary_elementwise_rem_float32` | 7 | 31 | **LARGE DIFF** |
| `test_binary_elementwise_rem_float64` | 7 | 31 | **LARGE DIFF** |
| `test_binary_elementwise_sub_bfloat16` | 10 | 9 |  |
| `test_binary_elementwise_sub_float16` | 7 | 6 |  |
| `test_binary_elementwise_sub_float32` | 7 | 6 |  |
| `test_binary_elementwise_sub_float64` | 7 | 6 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=bfloat16[2,3]_newdtype=bfloat16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=bfloat16[2,3]_newdtype=float16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=bfloat16[2,3]_newdtype=int16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=bfloat16[2,3]_newdtype=uint16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=bool[2,3]_newdtype=bool` | 6 | 0 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=complex128[2,3]_newdtype=complex128` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=complex64[2,3]_newdtype=complex64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float16[2,3]_newdtype=bfloat16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float16[2,3]_newdtype=float16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float16[2,3]_newdtype=int16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float16[2,3]_newdtype=uint16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float32[2,3]_newdtype=float32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float32[2,3]_newdtype=int32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float32[2,3]_newdtype=uint32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float64[2,3]_newdtype=float64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float64[2,3]_newdtype=int64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=float64[2,3]_newdtype=uint64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int16[2,3]_newdtype=bfloat16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int16[2,3]_newdtype=float16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int16[2,3]_newdtype=int16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int16[2,3]_newdtype=uint16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int32[2,3]_newdtype=float32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int32[2,3]_newdtype=int32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int32[2,3]_newdtype=uint32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int64[2,3]_newdtype=float64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int64[2,3]_newdtype=int64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int64[2,3]_newdtype=uint64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int8[2,3]_newdtype=int8` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=int8[2,3]_newdtype=uint8` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint16[2,3]_newdtype=bfloat16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint16[2,3]_newdtype=float16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint16[2,3]_newdtype=int16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint16[2,3]_newdtype=uint16` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint32[2,3]_newdtype=float32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint32[2,3]_newdtype=int32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint32[2,3]_newdtype=uint32` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint64[2,3]_newdtype=float64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint64[2,3]_newdtype=int64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint64[2,3]_newdtype=uint64` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint8[2,3]_newdtype=int8` | 6 | 5 |  |
| `test_bitcast_convert_type_dtypes_to_new_dtypes_shape=uint8[2,3]_newdtype=uint8` | 6 | 5 |  |
| `test_bitwise_not_bool` | 6 | 5 |  |
| `test_bitwise_not_int16` | 6 | 5 |  |
| `test_bitwise_not_int32` | 6 | 5 |  |
| `test_bitwise_not_int64` | 6 | 5 |  |
| `test_bitwise_not_int8` | 6 | 5 |  |
| `test_bitwise_not_uint16` | 6 | 5 |  |
| `test_bitwise_not_uint32` | 6 | 5 |  |
| `test_bitwise_not_uint64` | 6 | 5 |  |
| `test_bitwise_not_uint8` | 6 | 5 |  |
| `test_boolean_gather` | 13 | 16 | **LARGE DIFF** |
| `test_broadcast_dtypes_shape=bfloat16[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=bool[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=complex128[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=complex64[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=float16[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=float32[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=float64[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=int16[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=int32[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=int64[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=int8[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=uint16[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=uint32[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=uint64[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_dtypes_shape=uint8[2]_sizes=()` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=bfloat16[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=bool[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=complex128[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=complex64[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=float16[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=float32[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=float64[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=int16[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=int32[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=int64[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=int8[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=uint16[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=uint32[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=uint64[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_dtypes_shape=uint8[2]_outshape=(2,)_broadcastdimensions=(0,)` | 6 | 5 |  |
| `test_broadcast_in_dim_parameter_combinations_shape=float32[1,2]_outshape=(4, 3, 2)_broadcastdimensio`<br>`ns=(0, 2)` | 11 | 10 |  |
| `test_broadcast_in_dim_parameter_combinations_shape=float32[2]_outshape=(2, 3)_broadcastdimensions=(0`<br>`,)` | 6 | 5 |  |
| `test_broadcast_in_dim_parameter_combinations_shape=float32[2]_outshape=(3, 2)_broadcastdimensions=(1`<br>`,)` | 6 | 5 |  |
| `test_broadcast_in_dim_parameter_combinations_shape=float32[]_outshape=(2, 3)_broadcastdimensions=()` | 6 | 5 |  |
| `test_broadcast_sizes_shape=float32[2]_sizes=(1, 2, 3)` | 6 | 5 |  |
| `test_broadcast_sizes_shape=float32[2]_sizes=(2,)` | 6 | 5 |  |
| `test_cholesky_shape=complex128[1,1]` | 21 | 0 |  |
| `test_cholesky_shape=complex128[1000,0,0]` | 7 | 0 |  |
| `test_cholesky_shape=complex128[2,5,5]` | 25 | 0 |  |
| `test_cholesky_shape=complex128[200,200]` | 32 | 0 |  |
| `test_cholesky_shape=complex128[4,4]` | 23 | 0 |  |
| `test_cholesky_shape=complex64[1,1]` | 21 | 0 |  |
| `test_cholesky_shape=complex64[1000,0,0]` | 7 | 0 |  |
| `test_cholesky_shape=complex64[2,5,5]` | 25 | 0 |  |
| `test_cholesky_shape=complex64[200,200]` | 23 | 0 |  |
| `test_cholesky_shape=complex64[4,4]` | 23 | 0 |  |
| `test_cholesky_shape=float32[1,1]` | 21 | 12 | **LARGE DIFF** |
| `test_cholesky_shape=float32[1000,0,0]` | 7 | 5 | **LARGE DIFF** |
| `test_cholesky_shape=float32[2,5,5]` | 25 | 91 | **LARGE DIFF** |
| `test_cholesky_shape=float32[200,200]` | 23 | 303 | **LARGE DIFF** |
| `test_cholesky_shape=float32[4,4]` | 23 | 89 | **LARGE DIFF** |
| `test_cholesky_shape=float64[1,1]` | 21 | 12 | **LARGE DIFF** |
| `test_cholesky_shape=float64[1000,0,0]` | 7 | 5 | **LARGE DIFF** |
| `test_cholesky_shape=float64[2,5,5]` | 25 | 91 | **LARGE DIFF** |
| `test_cholesky_shape=float64[200,200]` | 23 | 303 | **LARGE DIFF** |
| `test_cholesky_shape=float64[4,4]` | 23 | 89 | **LARGE DIFF** |
| `test_clamp_broadcasting_min=float32[2,3]_operand=float32[2,3]_max=float32[2,3]` | 8 | 15 | **LARGE DIFF** |
| `test_clamp_broadcasting_min=float32[2,3]_operand=float32[2,3]_max=float32[]` | 15 | 16 |  |
| `test_clamp_broadcasting_min=float32[]_operand=float32[2,3]_max=float32[2,3]` | 15 | 16 |  |
| `test_clamp_dtypes_min=bfloat16[]_operand=bfloat16[2,3]_max=bfloat16[]` | 24 | 30 | **LARGE DIFF** |
| `test_clamp_dtypes_min=float16[]_operand=float16[2,3]_max=float16[]` | 16 | 17 |  |
| `test_clamp_dtypes_min=float32[]_operand=float32[2,3]_max=float32[]` | 16 | 17 |  |
| `test_clamp_dtypes_min=float64[]_operand=float64[2,3]_max=float64[]` | 16 | 17 |  |
| `test_clamp_dtypes_min=int16[]_operand=int16[2,3]_max=int16[]` | 16 | 17 |  |
| `test_clamp_dtypes_min=int32[]_operand=int32[2,3]_max=int32[]` | 16 | 17 |  |
| `test_clamp_dtypes_min=int64[]_operand=int64[2,3]_max=int64[]` | 16 | 17 |  |
| `test_clamp_dtypes_min=int8[]_operand=int8[2,3]_max=int8[]` | 16 | 0 |  |
| `test_clamp_dtypes_min=uint16[]_operand=uint16[2,3]_max=uint16[]` | 16 | 0 |  |
| `test_clamp_dtypes_min=uint32[]_operand=uint32[2,3]_max=uint32[]` | 16 | 0 |  |
| `test_clamp_dtypes_min=uint64[]_operand=uint64[2,3]_max=uint64[]` | 16 | 0 |  |
| `test_clamp_dtypes_min=uint8[]_operand=uint8[2,3]_max=uint8[]` | 16 | 17 |  |
| `test_clamp_order=False_min=float32[]_operand=float32[2,3]_max=float32[]` | 16 | 17 |  |
| `test_clamp_order=True_min=float32[]_operand=float32[2,3]_max=float32[]` | 16 | 17 |  |
| `test_comparators_broadcasting_op=eq_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_comparators_broadcasting_op=eq_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_comparators_broadcasting_op=ge_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_comparators_broadcasting_op=ge_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_comparators_broadcasting_op=gt_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_comparators_broadcasting_op=gt_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_comparators_broadcasting_op=le_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_comparators_broadcasting_op=le_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_comparators_broadcasting_op=lt_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_comparators_broadcasting_op=lt_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_comparators_broadcasting_op=ne_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_comparators_broadcasting_op=ne_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_comparators_dtypes_op=eq_lhs=bfloat16[]_rhs=bfloat16[]` | 9 | 8 |  |
| `test_comparators_dtypes_op=eq_lhs=bool[]_rhs=bool[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=complex128[]_rhs=complex128[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=complex64[]_rhs=complex64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=float16[]_rhs=float16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=float32[]_rhs=float32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=float64[]_rhs=float64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=int16[]_rhs=int16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=int32[]_rhs=int32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=int64[]_rhs=int64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=int8[]_rhs=int8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=uint16[]_rhs=uint16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=uint32[]_rhs=uint32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=uint64[]_rhs=uint64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=eq_lhs=uint8[]_rhs=uint8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=bfloat16[]_rhs=bfloat16[]` | 9 | 8 |  |
| `test_comparators_dtypes_op=ge_lhs=bool[]_rhs=bool[]` | 7 | 0 |  |
| `test_comparators_dtypes_op=ge_lhs=float16[]_rhs=float16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=float32[]_rhs=float32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=float64[]_rhs=float64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=int16[]_rhs=int16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=int32[]_rhs=int32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=int64[]_rhs=int64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=int8[]_rhs=int8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=uint16[]_rhs=uint16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=uint32[]_rhs=uint32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=uint64[]_rhs=uint64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ge_lhs=uint8[]_rhs=uint8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=bfloat16[]_rhs=bfloat16[]` | 9 | 8 |  |
| `test_comparators_dtypes_op=gt_lhs=bool[]_rhs=bool[]` | 7 | 0 |  |
| `test_comparators_dtypes_op=gt_lhs=float16[]_rhs=float16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=float32[]_rhs=float32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=float64[]_rhs=float64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=int16[]_rhs=int16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=int32[]_rhs=int32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=int64[]_rhs=int64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=int8[]_rhs=int8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=uint16[]_rhs=uint16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=uint32[]_rhs=uint32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=uint64[]_rhs=uint64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=gt_lhs=uint8[]_rhs=uint8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=bfloat16[]_rhs=bfloat16[]` | 9 | 8 |  |
| `test_comparators_dtypes_op=le_lhs=bool[]_rhs=bool[]` | 7 | 0 |  |
| `test_comparators_dtypes_op=le_lhs=float16[]_rhs=float16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=float32[]_rhs=float32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=float64[]_rhs=float64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=int16[]_rhs=int16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=int32[]_rhs=int32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=int64[]_rhs=int64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=int8[]_rhs=int8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=uint16[]_rhs=uint16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=uint32[]_rhs=uint32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=uint64[]_rhs=uint64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=le_lhs=uint8[]_rhs=uint8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=bfloat16[]_rhs=bfloat16[]` | 9 | 8 |  |
| `test_comparators_dtypes_op=lt_lhs=bool[]_rhs=bool[]` | 7 | 0 |  |
| `test_comparators_dtypes_op=lt_lhs=float16[]_rhs=float16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=float32[]_rhs=float32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=float64[]_rhs=float64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=int16[]_rhs=int16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=int32[]_rhs=int32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=int64[]_rhs=int64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=int8[]_rhs=int8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=uint16[]_rhs=uint16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=uint32[]_rhs=uint32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=uint64[]_rhs=uint64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=lt_lhs=uint8[]_rhs=uint8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=bfloat16[]_rhs=bfloat16[]` | 9 | 8 |  |
| `test_comparators_dtypes_op=ne_lhs=bool[]_rhs=bool[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=complex128[]_rhs=complex128[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=complex64[]_rhs=complex64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=float16[]_rhs=float16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=float32[]_rhs=float32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=float64[]_rhs=float64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=int16[]_rhs=int16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=int32[]_rhs=int32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=int64[]_rhs=int64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=int8[]_rhs=int8[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=uint16[]_rhs=uint16[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=uint32[]_rhs=uint32[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=uint64[]_rhs=uint64[]` | 7 | 6 |  |
| `test_comparators_dtypes_op=ne_lhs=uint8[]_rhs=uint8[]` | 7 | 6 |  |
| `test_complex_broadcast_lhs=float32[3,1]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_complex_broadcast_lhs=float32[3,2]_rhs=float32[3,1]` | 14 | 13 |  |
| `test_complex_dtypes_lhs=float32[3,4]_rhs=float32[3,4]` | 7 | 6 |  |
| `test_complex_dtypes_lhs=float64[3,4]_rhs=float64[3,4]` | 7 | 6 |  |
| `test_concat` | 16 | 15 |  |
| `test_concatenate_dimension_shapes=float32[2,3]_float32[2,3]_dimension=1` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=bfloat16[2,3]_bfloat16[2,3]_dimension=0` | 10 | 9 |  |
| `test_concatenate_dtypes_shapes=bool[2,3]_bool[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=complex128[2,3]_complex128[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=complex64[2,3]_complex64[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=float16[2,3]_float16[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=float32[2,3]_float32[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=float64[2,3]_float64[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=int16[2,3]_int16[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=int32[2,3]_int32[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=int64[2,3]_int64[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=int8[2,3]_int8[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=uint16[2,3]_uint16[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=uint32[2,3]_uint32[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=uint64[2,3]_uint64[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_dtypes_shapes=uint8[2,3]_uint8[2,3]_dimension=0` | 7 | 6 |  |
| `test_concatenate_nb_operands_shapes=float32[2,3,4]_float32[3,3,4]_float32[4,3,4]_dimension=0` | 8 | 7 |  |
| `test_conj_dtypes_operand=complex128[3,4]_kwargs={}` | 13 | 12 |  |
| `test_conj_dtypes_operand=complex64[3,4]_kwargs={}` | 13 | 12 |  |
| `test_conj_dtypes_operand=float32[3,4]_kwargs={}` | 13 | 5 | **LARGE DIFF** |
| `test_conj_dtypes_operand=float64[3,4]_kwargs={}` | 13 | 5 | **LARGE DIFF** |
| `test_conj_kwargs_operand=float32[3,4]_kwargs={'_input_dtype':<class'numpy.float32'>}` | 13 | 5 | **LARGE DIFF** |
| `test_conv_general_dilated_dilations_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(1,1)_p`<br>`adding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(2,3)_dimensionnumbers=('NCHW','OIHW','NCHW')_fea`<br>`turegroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 8 | 7 |  |
| `test_conv_general_dilated_dilations_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(1,1)_p`<br>`adding=((0,0),(0,0))_lhsdilation=(2,2)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW')_fea`<br>`turegroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dilations_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(1,1)_p`<br>`adding=((0,0),(0,0))_lhsdilation=(2,3)_rhsdilation=(3,2)_dimensionnumbers=('NCHW','OIHW','NCHW')_fea`<br>`turegroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dimension_numbers_lhs=float32[2,3,9,10]_rhs=float32[4,5,3,3]_windowstrides`<br>`=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','HWIO','NH`<br>`WC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 8 | 7 |  |
| `test_conv_general_dilated_dimension_numbers_lhs=float32[2,9,10,3]_rhs=float32[4,5,3,3]_windowstrides`<br>`=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NHWC','HWIO','NH`<br>`WC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_dtype_precision_lhs=bfloat16[2,3,9,10]_rhs=bfloat16[3,3,4,5]_windowstrides`<br>`=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NC`<br>`HW')_featuregroupcount=1_batchgroupcount=1_precision=DEFAULT_enablexla=True` | 22 | 25 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=bfloat16[2,3,9,10]_rhs=bfloat16[3,3,4,5]_windowstrides`<br>`=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NC`<br>`HW')_featuregroupcount=1_batchgroupcount=1_precision=HIGHEST_enablexla=True` | 22 | 25 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=bfloat16[2,3,9,10]_rhs=bfloat16[3,3,4,5]_windowstrides`<br>`=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NC`<br>`HW')_featuregroupcount=1_batchgroupcount=1_precision=HIGH_enablexla=True` | 22 | 25 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=bfloat16[2,3,9,10]_rhs=bfloat16[3,3,4,5]_windowstrides`<br>`=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NC`<br>`HW')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 22 | 25 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=complex128[2,3,9,10]_rhs=complex128[3,3,4,5]_windowstr`<br>`ides=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW'`<br>`,'NCHW')_featuregroupcount=1_batchgroupcount=1_precision=DEFAULT_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex128[2,3,9,10]_rhs=complex128[3,3,4,5]_windowstr`<br>`ides=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW'`<br>`,'NCHW')_featuregroupcount=1_batchgroupcount=1_precision=HIGHEST_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex128[2,3,9,10]_rhs=complex128[3,3,4,5]_windowstr`<br>`ides=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW'`<br>`,'NCHW')_featuregroupcount=1_batchgroupcount=1_precision=HIGH_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex128[2,3,9,10]_rhs=complex128[3,3,4,5]_windowstr`<br>`ides=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW'`<br>`,'NCHW')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex64[2,3,9,10]_rhs=complex64[3,3,4,5]_windowstrid`<br>`es=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','`<br>`NCHW')_featuregroupcount=1_batchgroupcount=1_precision=DEFAULT_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex64[2,3,9,10]_rhs=complex64[3,3,4,5]_windowstrid`<br>`es=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','`<br>`NCHW')_featuregroupcount=1_batchgroupcount=1_precision=HIGHEST_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex64[2,3,9,10]_rhs=complex64[3,3,4,5]_windowstrid`<br>`es=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','`<br>`NCHW')_featuregroupcount=1_batchgroupcount=1_precision=HIGH_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=complex64[2,3,9,10]_rhs=complex64[3,3,4,5]_windowstrid`<br>`es=(1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','`<br>`NCHW')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 62 | 0 |  |
| `test_conv_general_dilated_dtype_precision_lhs=float16[2,3,9,10]_rhs=float16[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=DEFAULT_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float16[2,3,9,10]_rhs=float16[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=HIGHEST_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float16[2,3,9,10]_rhs=float16[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=HIGH_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float16[2,3,9,10]_rhs=float16[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=DEFAULT_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=HIGHEST_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=HIGH_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float64[2,3,9,10]_rhs=float64[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=DEFAULT_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float64[2,3,9,10]_rhs=float64[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=HIGHEST_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float64[2,3,9,10]_rhs=float64[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=HIGH_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_dtype_precision_lhs=float64[2,3,9,10]_rhs=float64[3,3,4,5]_windowstrides=(`<br>`1,1)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_group_counts_lhs=float32[2,6,9,10]_rhs=float32[6,3,4,5]_windowstrides=(1,1`<br>`)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW')_`<br>`featuregroupcount=2_batchgroupcount=1_precision=None_enablexla=True` | 26 | 25 |  |
| `test_conv_general_dilated_group_counts_lhs=float32[4,3,9,10]_rhs=float32[6,3,4,5]_windowstrides=(1,1`<br>`)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW')_`<br>`featuregroupcount=1_batchgroupcount=2_precision=None_enablexla=True` | 26 | 0 |  |
| `test_conv_general_dilated_padding_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(1,1)_pad`<br>`ding=((1,2),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW')_featu`<br>`regroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_padding_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(1,1)_pad`<br>`ding=((1,2),(2,1))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW')_featu`<br>`regroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=SAME_lhsdilation=(1,)_rhsdilation=(1,)_dimensionnumbers=('NWC','WIO','NWC')_featuregr`<br>`oupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 9 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=SAME_lhsdilation=(1,)_rhsdilation=(1,)_dimensionnumbers=('NWC','WIO','NWC')_featuregr`<br>`oupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=SAME_lhsdilation=(1,)_rhsdilation=(2,)_dimensionnumbers=('NWC','WIO','NWC')_featuregr`<br>`oupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 21 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=SAME_lhsdilation=(1,)_rhsdilation=(2,)_dimensionnumbers=('NWC','WIO','NWC')_featuregr`<br>`oupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=VALID_lhsdilation=(1,)_rhsdilation=(1,)_dimensionnumbers=('NWC','WIO','NWC')_featureg`<br>`roupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 9 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=VALID_lhsdilation=(1,)_rhsdilation=(1,)_dimensionnumbers=('NWC','WIO','NWC')_featureg`<br>`roupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=VALID_lhsdilation=(1,)_rhsdilation=(2,)_dimensionnumbers=('NWC','WIO','NWC')_featureg`<br>`roupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 19 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_1d_lhs=float32[1,28,1]_rhs=float32[3,1,16]_windowstride`<br>`s=(1,)_padding=VALID_lhsdilation=(1,)_rhsdilation=(2,)_dimensionnumbers=('NWC','WIO','NWC')_featureg`<br>`roupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=SAME_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NHWC','HWIO','NHWC'`<br>`)_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=SAME_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NHWC','HWIO','NHWC'`<br>`)_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=SAME_lhsdilation=(1,1)_rhsdilation=(1,2)_dimensionnumbers=('NHWC','HWIO','NHWC'`<br>`)_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 22 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=SAME_lhsdilation=(1,1)_rhsdilation=(1,2)_dimensionnumbers=('NHWC','HWIO','NHWC'`<br>`)_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=VALID_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NHWC','HWIO','NHWC`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=VALID_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NHWC','HWIO','NHWC`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=VALID_lhsdilation=(1,1)_rhsdilation=(1,2)_dimensionnumbers=('NHWC','HWIO','NHWC`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 7 | 20 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_2d_lhs=float32[1,28,28,1]_rhs=float32[3,3,1,16]_windows`<br>`trides=(1,1)_padding=VALID_lhsdilation=(1,1)_rhsdilation=(1,2)_dimensionnumbers=('NHWC','HWIO','NHWC`<br>`')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 7 | 6 |  |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=SAME_lhsdilation=(1,1,1)_rhsdilation=(1,1,1)_dimensionnumbers=('NDHWC','D`<br>`HWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 12 | 11 |  |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=SAME_lhsdilation=(1,1,1)_rhsdilation=(1,1,1)_dimensionnumbers=('NDHWC','D`<br>`HWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 12 | 11 |  |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=SAME_lhsdilation=(1,1,1)_rhsdilation=(1,1,2)_dimensionnumbers=('NDHWC','D`<br>`HWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 12 | 27 | **LARGE DIFF** |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=SAME_lhsdilation=(1,1,1)_rhsdilation=(1,1,2)_dimensionnumbers=('NDHWC','D`<br>`HWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 12 | 11 |  |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=VALID_lhsdilation=(1,1,1)_rhsdilation=(1,1,1)_dimensionnumbers=('NDHWC','`<br>`DHWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=False` | 12 | 11 |  |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=VALID_lhsdilation=(1,1,1)_rhsdilation=(1,1,1)_dimensionnumbers=('NDHWC','`<br>`DHWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 12 | 11 |  |
| `test_conv_general_dilated_tf_conversion_path_3d_lhs=float32[1,4,28,28,1]_rhs=float32[2,3,3,1,16]_win`<br>`dowstrides=(1,1,1)_padding=VALID_lhsdilation=(1,1,1)_rhsdilation=(1,1,2)_dimensionnumbers=('NDHWC','`<br>`DHWIO','NDHWC')_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 12 | 11 |  |
| `test_conv_general_dilated_window_strides_lhs=float32[2,3,9,10]_rhs=float32[3,3,4,5]_windowstrides=(2`<br>`,3)_padding=((0,0),(0,0))_lhsdilation=(1,1)_rhsdilation=(1,1)_dimensionnumbers=('NCHW','OIHW','NCHW'`<br>`)_featuregroupcount=1_batchgroupcount=1_precision=None_enablexla=True` | 11 | 14 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=bfloat`<br>`16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=bool` | 6 | 19 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=comple`<br>`x128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=comple`<br>`x64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=float1`<br>`6` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=float3`<br>`2` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=float6`<br>`4` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=int16` | 6 | 27 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=int32` | 6 | 27 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=int64` | 6 | 27 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=bfloat16[100,100]_olddtype=bfloat16_newdtype=int8` | 6 | 27 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=bool` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=bool[100,100]_olddtype=bool_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=bf`<br>`loat16` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=bo`<br>`ol` | 11 | 12 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=co`<br>`mplex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=co`<br>`mplex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=fl`<br>`oat16` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=fl`<br>`oat32` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=fl`<br>`oat64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=in`<br>`t16` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=in`<br>`t32` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=in`<br>`t64` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex128[100,100]_olddtype=complex128_newdtype=in`<br>`t8` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=bflo`<br>`at16` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=bool` | 11 | 12 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=comp`<br>`lex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=comp`<br>`lex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=floa`<br>`t16` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=floa`<br>`t32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=floa`<br>`t64` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=int1`<br>`6` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=int3`<br>`2` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=int6`<br>`4` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=complex64[100,100]_olddtype=complex64_newdtype=int8` | 11 | 10 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=complex1`<br>`28` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=complex6`<br>`4` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=int16` | 6 | 17 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=int32` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=int64` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float16[100,100]_olddtype=float16_newdtype=int8` | 6 | 17 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=complex1`<br>`28` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=complex6`<br>`4` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=int16` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=int32` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=int64` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float32[100,100]_olddtype=float32_newdtype=int8` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=complex1`<br>`28` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=complex6`<br>`4` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=int16` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=int32` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=int64` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=float64[100,100]_olddtype=float64_newdtype=int8` | 6 | 13 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int16[100,100]_olddtype=int16_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int32[100,100]_olddtype=int32_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int64[100,100]_olddtype=int64_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=int8[100,100]_olddtype=int8_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint16[100,100]_olddtype=uint16_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint32[100,100]_olddtype=uint32_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint64[100,100]_olddtype=uint64_newdtype=uint8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=bfloat16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=bool` | 6 | 11 | **LARGE DIFF** |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=complex128` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=complex64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=float16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=float32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=float64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=int16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=int32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=int64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=int8` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=uint16` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=uint32` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=uint64` | 6 | 5 |  |
| `test_convert_element_type_dtypes_to_dtypes_shape=uint8[100,100]_olddtype=uint8_newdtype=uint8` | 6 | 5 |  |
| `test_cumreduce_axis_by_fun_f=cummax_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cummax_shape=float32[8,9]_axis=1_reverse=False` | 73 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cummin_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cummin_shape=float32[8,9]_axis=1_reverse=False` | 73 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cumprod_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cumprod_shape=float32[8,9]_axis=1_reverse=False` | 73 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cumsum_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_axis_by_fun_f=cumsum_shape=float32[8,9]_axis=1_reverse=False` | 73 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=bfloat16[8,9]_axis=0_reverse=False` | 172 | 14 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=complex128[8,9]_axis=0_reverse=False` | 138 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=complex64[8,9]_axis=0_reverse=False` | 138 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=float16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=float64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=int16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=int32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=int64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=int8[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=uint16[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=uint32[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=uint64[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummax_shape=uint8[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=bfloat16[8,9]_axis=0_reverse=False` | 172 | 14 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=complex128[8,9]_axis=0_reverse=False` | 138 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=complex64[8,9]_axis=0_reverse=False` | 138 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=float16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=float64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=int16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=int32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=int64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=int8[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=uint16[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=uint32[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=uint64[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cummin_shape=uint8[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=bfloat16[8,9]_axis=0_reverse=False` | 172 | 14 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=complex128[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=complex64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=float16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=float64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=int16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=int32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=int64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=int8[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=uint16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=uint32[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=uint64[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cumprod_shape=uint8[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=bfloat16[8,9]_axis=0_reverse=False` | 172 | 14 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=complex128[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=complex64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=float16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=float32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=float64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=int16[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=int32[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=int64[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=int8[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=uint16[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=uint32[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=uint64[8,9]_axis=0_reverse=False` | 74 | 0 |  |
| `test_cumreduce_dtype_by_fun_f=cumsum_shape=uint8[8,9]_axis=0_reverse=False` | 74 | 11 | **LARGE DIFF** |
| `test_cumreduce_reverse_f=cummax_shape=float32[8,9]_axis=0_reverse=True` | 82 | 11 | **LARGE DIFF** |
| `test_cumreduce_reverse_f=cummin_shape=float32[8,9]_axis=0_reverse=True` | 82 | 11 | **LARGE DIFF** |
| `test_cumreduce_reverse_f=cumprod_shape=float32[8,9]_axis=0_reverse=True` | 82 | 11 | **LARGE DIFF** |
| `test_cumreduce_reverse_f=cumsum_shape=float32[8,9]_axis=0_reverse=True` | 82 | 11 | **LARGE DIFF** |
| `test_device_put_shape=bfloat16[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=bool[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=complex128[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=complex64[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=float16[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=float32[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=float32[3,4]_device=cpu` | 6 | 5 |  |
| `test_device_put_shape=float64[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=int16[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=int32[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=int64[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=int8[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=uint16[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=uint32[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=uint64[3,4]_device=None` | 6 | 5 |  |
| `test_device_put_shape=uint8[3,4]_device=None` | 6 | 5 |  |
| `test_disable_xla_pad` | 7 | 0 |  |
| `test_div_broadcast_prim=div_lhs=float32[2,1,3]_rhs=float32[2,4,3]` | 14 | 13 |  |
| `test_div_broadcast_prim=div_lhs=float32[2,4,3]_rhs=float32[2,1,3]` | 14 | 13 |  |
| `test_div_broadcast_prim=rem_lhs=float32[2,1,3]_rhs=float32[2,4,3]` | 14 | 28 | **LARGE DIFF** |
| `test_div_broadcast_prim=rem_lhs=float32[2,4,3]_rhs=float32[2,1,3]` | 14 | 28 | **LARGE DIFF** |
| `test_div_dtypes_prim=div_lhs=bfloat16[2]_rhs=bfloat16[2]` | 10 | 9 |  |
| `test_div_dtypes_prim=div_lhs=complex128[2]_rhs=complex128[2]` | 7 | 6 |  |
| `test_div_dtypes_prim=div_lhs=complex64[2]_rhs=complex64[2]` | 7 | 6 |  |
| `test_div_dtypes_prim=div_lhs=float16[2]_rhs=float16[2]` | 7 | 6 |  |
| `test_div_dtypes_prim=div_lhs=float32[2]_rhs=float32[2]` | 7 | 6 |  |
| `test_div_dtypes_prim=div_lhs=float64[2]_rhs=float64[2]` | 7 | 6 |  |
| `test_div_dtypes_prim=div_lhs=int16[2]_rhs=int16[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=div_lhs=int32[2]_rhs=int32[2]` | 7 | 42 | **LARGE DIFF** |
| `test_div_dtypes_prim=div_lhs=int64[2]_rhs=int64[2]` | 7 | 42 | **LARGE DIFF** |
| `test_div_dtypes_prim=div_lhs=int8[2]_rhs=int8[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=div_lhs=uint16[2]_rhs=uint16[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=div_lhs=uint32[2]_rhs=uint32[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=div_lhs=uint64[2]_rhs=uint64[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=div_lhs=uint8[2]_rhs=uint8[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=rem_lhs=bfloat16[2]_rhs=bfloat16[2]` | 10 | 59 | **LARGE DIFF** |
| `test_div_dtypes_prim=rem_lhs=float16[2]_rhs=float16[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_dtypes_prim=rem_lhs=float32[2]_rhs=float32[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_dtypes_prim=rem_lhs=float64[2]_rhs=float64[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_dtypes_prim=rem_lhs=int16[2]_rhs=int16[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=rem_lhs=int32[2]_rhs=int32[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_dtypes_prim=rem_lhs=int64[2]_rhs=int64[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_dtypes_prim=rem_lhs=int8[2]_rhs=int8[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=rem_lhs=uint16[2]_rhs=uint16[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=rem_lhs=uint32[2]_rhs=uint32[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=rem_lhs=uint64[2]_rhs=uint64[2]` | 7 | 0 |  |
| `test_div_dtypes_prim=rem_lhs=uint8[2]_rhs=uint8[2]` | 7 | 0 |  |
| `test_div_singularity_0_by_0_prim=div_lhs=float32[2]_rhs=float32[2]` | 7 | 6 |  |
| `test_div_singularity_0_by_0_prim=rem_lhs=float32[2]_rhs=float32[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_singularity_inf_by_inf_prim=div_lhs=float32[1]_rhs=float32[1]` | 7 | 6 |  |
| `test_div_singularity_inf_by_inf_prim=rem_lhs=float32[1]_rhs=float32[1]` | 7 | 30 | **LARGE DIFF** |
| `test_div_singularity_negative_by_0_prim=div_lhs=float32[2]_rhs=float32[2]` | 7 | 6 |  |
| `test_div_singularity_negative_by_0_prim=rem_lhs=float32[2]_rhs=float32[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_singularity_positive_by_0_int32_prim=rem_lhs=int32[2]_rhs=int32[2]` | 7 | 31 | **LARGE DIFF** |
| `test_div_singularity_positive_by_0_prim=div_lhs=float32[2]_rhs=float32[2]` | 7 | 6 |  |
| `test_div_singularity_positive_by_0_prim=rem_lhs=float32[2]_rhs=float32[2]` | 7 | 31 | **LARGE DIFF** |
| `test_dot_general_batch_dimensions_lhs=float32[4,4,3,3,4]_rhs=float32[4,4,3,4,2]_dimensionnumbers=(((`<br>`4,),(3,)),((0,1,2),(0,1,2)))_precision=None` | 7 | 6 |  |
| `test_dot_general_batch_dimensions_lhs=float32[8,4,3,3,4]_rhs=float32[4,8,3,4,2]_dimensionnumbers=(((`<br>`4,3),(3,2)),((0,1),(1,0)))_precision=None` | 19 | 13 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[1,3,4]_rhs=bfloat16[1,4,3]_dimensionnumbers=(((2,`<br>`1),(1,2)),((0,),(0,)))_precision=DEFAULT` | 27 | 26 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[1,3,4]_rhs=bfloat16[1,4,3]_dimensionnumbers=(((2,`<br>`1),(1,2)),((0,),(0,)))_precision=HIGHEST` | 27 | 26 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[1,3,4]_rhs=bfloat16[1,4,3]_dimensionnumbers=(((2,`<br>`1),(1,2)),((0,),(0,)))_precision=HIGH` | 27 | 26 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[1,3,4]_rhs=bfloat16[1,4,3]_dimensionnumbers=(((2,`<br>`1),(1,2)),((0,),(0,)))_precision=None` | 27 | 26 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[3,4]_rhs=bfloat16[4,2]_dimensionnumbers=(((1,),(0`<br>`,)),((),()))_precision=DEFAULT` | 10 | 9 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[3,4]_rhs=bfloat16[4,2]_dimensionnumbers=(((1,),(0`<br>`,)),((),()))_precision=HIGHEST` | 10 | 9 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[3,4]_rhs=bfloat16[4,2]_dimensionnumbers=(((1,),(0`<br>`,)),((),()))_precision=HIGH` | 10 | 9 |  |
| `test_dot_general_dtypes_and_precision_lhs=bfloat16[3,4]_rhs=bfloat16[4,2]_dimensionnumbers=(((1,),(0`<br>`,)),((),()))_precision=None` | 10 | 9 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[1,3,4]_rhs=bool[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[1,3,4]_rhs=bool[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[1,3,4]_rhs=bool[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[1,3,4]_rhs=bool[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[3,4]_rhs=bool[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[3,4]_rhs=bool[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[3,4]_rhs=bool[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=bool[3,4]_rhs=bool[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[1,3,4]_rhs=complex128[1,4,3]_dimensionnumbers=(`<br>`((2,1),(1,2)),((0,),(0,)))_precision=DEFAULT` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[1,3,4]_rhs=complex128[1,4,3]_dimensionnumbers=(`<br>`((2,1),(1,2)),((0,),(0,)))_precision=HIGHEST` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[1,3,4]_rhs=complex128[1,4,3]_dimensionnumbers=(`<br>`((2,1),(1,2)),((0,),(0,)))_precision=HIGH` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[1,3,4]_rhs=complex128[1,4,3]_dimensionnumbers=(`<br>`((2,1),(1,2)),((0,),(0,)))_precision=None` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[3,4]_rhs=complex128[4,2]_dimensionnumbers=(((1,`<br>`),(0,)),((),()))_precision=DEFAULT` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[3,4]_rhs=complex128[4,2]_dimensionnumbers=(((1,`<br>`),(0,)),((),()))_precision=HIGHEST` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[3,4]_rhs=complex128[4,2]_dimensionnumbers=(((1,`<br>`),(0,)),((),()))_precision=HIGH` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex128[3,4]_rhs=complex128[4,2]_dimensionnumbers=(((1,`<br>`),(0,)),((),()))_precision=None` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[1,3,4]_rhs=complex64[1,4,3]_dimensionnumbers=(((`<br>`2,1),(1,2)),((0,),(0,)))_precision=DEFAULT` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[1,3,4]_rhs=complex64[1,4,3]_dimensionnumbers=(((`<br>`2,1),(1,2)),((0,),(0,)))_precision=HIGHEST` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[1,3,4]_rhs=complex64[1,4,3]_dimensionnumbers=(((`<br>`2,1),(1,2)),((0,),(0,)))_precision=HIGH` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[1,3,4]_rhs=complex64[1,4,3]_dimensionnumbers=(((`<br>`2,1),(1,2)),((0,),(0,)))_precision=None` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[3,4]_rhs=complex64[4,2]_dimensionnumbers=(((1,),`<br>`(0,)),((),()))_precision=DEFAULT` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[3,4]_rhs=complex64[4,2]_dimensionnumbers=(((1,),`<br>`(0,)),((),()))_precision=HIGHEST` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[3,4]_rhs=complex64[4,2]_dimensionnumbers=(((1,),`<br>`(0,)),((),()))_precision=HIGH` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=complex64[3,4]_rhs=complex64[4,2]_dimensionnumbers=(((1,),`<br>`(0,)),((),()))_precision=None` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[1,3,4]_rhs=float16[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=DEFAULT` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[1,3,4]_rhs=float16[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=HIGHEST` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[1,3,4]_rhs=float16[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=HIGH` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[1,3,4]_rhs=float16[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=None` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[3,4]_rhs=float16[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=DEFAULT` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[3,4]_rhs=float16[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=HIGHEST` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[3,4]_rhs=float16[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=HIGH` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float16[3,4]_rhs=float16[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=None` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[1,3,4]_rhs=float32[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=DEFAULT` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[1,3,4]_rhs=float32[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=HIGHEST` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[1,3,4]_rhs=float32[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=HIGH` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[1,3,4]_rhs=float32[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=None` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[3,4]_rhs=float32[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=DEFAULT` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[3,4]_rhs=float32[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=HIGHEST` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[3,4]_rhs=float32[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=HIGH` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float32[3,4]_rhs=float32[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=None` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[1,3,4]_rhs=float64[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=DEFAULT` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[1,3,4]_rhs=float64[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=HIGHEST` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[1,3,4]_rhs=float64[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=HIGH` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[1,3,4]_rhs=float64[1,4,3]_dimensionnumbers=(((2,1)`<br>`,(1,2)),((0,),(0,)))_precision=None` | 16 | 15 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[3,4]_rhs=float64[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=DEFAULT` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[3,4]_rhs=float64[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=HIGHEST` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[3,4]_rhs=float64[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=HIGH` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=float64[3,4]_rhs=float64[4,2]_dimensionnumbers=(((1,),(0,)`<br>`),((),()))_precision=None` | 7 | 6 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[1,3,4]_rhs=int16[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[1,3,4]_rhs=int16[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[1,3,4]_rhs=int16[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[1,3,4]_rhs=int16[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[3,4]_rhs=int16[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[3,4]_rhs=int16[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[3,4]_rhs=int16[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int16[3,4]_rhs=int16[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int32[1,3,4]_rhs=int32[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=DEFAULT` | 20 | 15 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[1,3,4]_rhs=int32[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGHEST` | 20 | 15 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[1,3,4]_rhs=int32[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGH` | 20 | 15 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[1,3,4]_rhs=int32[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=None` | 20 | 15 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[3,4]_rhs=int32[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=DEFAULT` | 22 | 6 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[3,4]_rhs=int32[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGHEST` | 22 | 6 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[3,4]_rhs=int32[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGH` | 22 | 6 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int32[3,4]_rhs=int32[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=None` | 22 | 6 | **LARGE DIFF** |
| `test_dot_general_dtypes_and_precision_lhs=int64[1,3,4]_rhs=int64[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[1,3,4]_rhs=int64[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[1,3,4]_rhs=int64[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[1,3,4]_rhs=int64[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[3,4]_rhs=int64[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[3,4]_rhs=int64[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[3,4]_rhs=int64[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int64[3,4]_rhs=int64[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[1,3,4]_rhs=int8[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[1,3,4]_rhs=int8[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[1,3,4]_rhs=int8[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[1,3,4]_rhs=int8[1,4,3]_dimensionnumbers=(((2,1),(1,2)`<br>`),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[3,4]_rhs=int8[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[3,4]_rhs=int8[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[3,4]_rhs=int8[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=int8[3,4]_rhs=int8[4,2]_dimensionnumbers=(((1,),(0,)),((),`<br>`()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[1,3,4]_rhs=uint16[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[1,3,4]_rhs=uint16[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[1,3,4]_rhs=uint16[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[1,3,4]_rhs=uint16[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[3,4]_rhs=uint16[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[3,4]_rhs=uint16[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[3,4]_rhs=uint16[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint16[3,4]_rhs=uint16[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[1,3,4]_rhs=uint32[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[1,3,4]_rhs=uint32[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[1,3,4]_rhs=uint32[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[1,3,4]_rhs=uint32[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[3,4]_rhs=uint32[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[3,4]_rhs=uint32[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[3,4]_rhs=uint32[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint32[3,4]_rhs=uint32[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[1,3,4]_rhs=uint64[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[1,3,4]_rhs=uint64[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[1,3,4]_rhs=uint64[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[1,3,4]_rhs=uint64[1,4,3]_dimensionnumbers=(((2,1),(`<br>`1,2)),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[3,4]_rhs=uint64[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[3,4]_rhs=uint64[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[3,4]_rhs=uint64[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint64[3,4]_rhs=uint64[4,2]_dimensionnumbers=(((1,),(0,)),`<br>`((),()))_precision=None` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[1,3,4]_rhs=uint8[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=DEFAULT` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[1,3,4]_rhs=uint8[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGHEST` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[1,3,4]_rhs=uint8[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=HIGH` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[1,3,4]_rhs=uint8[1,4,3]_dimensionnumbers=(((2,1),(1,`<br>`2)),((0,),(0,)))_precision=None` | 20 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[3,4]_rhs=uint8[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=DEFAULT` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[3,4]_rhs=uint8[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGHEST` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[3,4]_rhs=uint8[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=HIGH` | 22 | 0 |  |
| `test_dot_general_dtypes_and_precision_lhs=uint8[3,4]_rhs=uint8[4,2]_dimensionnumbers=(((1,),(0,)),((`<br>`),()))_precision=None` | 22 | 0 |  |
| `test_dot_general_squeeze_lhs=float32[4,4]_rhs=float32[4]_dimensionnumbers=(((1,),(0,)),((),()))_prec`<br>`ision=None` | 7 | 6 |  |
| `test_dot_general_squeeze_lhs=float32[4]_rhs=float32[4,4]_dimensionnumbers=(((0,),(0,)),((),()))_prec`<br>`ision=None` | 7 | 6 |  |
| `test_dot_general_squeeze_lhs=float32[4]_rhs=float32[4]_dimensionnumbers=(((0,),(0,)),((),()))_precis`<br>`ion=None` | 7 | 6 |  |
| `test_dynamic_slice_shape=(3,)_start_indices=(1,)_limit_indices=(2,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(2, 1)_strides=(1, 1)` | 8 | 7 |  |
| `test_dynamic_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(3, 1)_strides=None` | 8 | 7 |  |
| `test_dynamic_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(3, 2)_strides=None` | 22 | 21 |  |
| `test_dynamic_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(5, 3)_strides=(2, 1)` | 22 | 21 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-1,)_limit_indices=(0,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-1,)_limit_indices=(1,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-10,)_limit_indices=(-9,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-100,)_limit_indices=(-99,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-4,)_limit_indices=(-2,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-5,)_limit_indices=(-2,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(-6,)_limit_indices=(-5,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(1,)_limit_indices=(5,)_strides=(2,)` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(10,)_limit_indices=(11,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(3,)_limit_indices=(6,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(5,)_start_indices=(5,)_limit_indices=(6,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(7, 5, 3)_start_indices=(4, 0, 1)_limit_indices=(7, 1, 3)_strides=None` | 24 | 23 |  |
| `test_dynamic_slice_shape=(7,)_start_indices=(4,)_limit_indices=(7,)_strides=None` | 18 | 17 |  |
| `test_dynamic_slice_shape=(8,)_start_indices=(1,)_limit_indices=(6,)_strides=(2,)` | 18 | 17 |  |
| `test_dynamic_update_slice_operand=float32[3]_update=float32[1]_start_indices=(-1,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float32[3]_update=float32[1]_start_indices=(1,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float32[3]_update=float32[1]_start_indices=(10,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float32[3]_update=float32[2]_start_indices=(10,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float32[5,3]_update=float32[3,1]_start_indices=(1, 1)` | 25 | 24 |  |
| `test_dynamic_update_slice_operand=float32[7,5,3]_update=float32[2,0,1]_start_indices=(4, 1, 0)` | 8 | 6 | **LARGE DIFF** |
| `test_dynamic_update_slice_operand=float64[3]_update=float64[1]_start_indices=(-1,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float64[3]_update=float64[1]_start_indices=(1,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float64[3]_update=float64[1]_start_indices=(10,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float64[3]_update=float64[2]_start_indices=(10,)` | 21 | 20 |  |
| `test_dynamic_update_slice_operand=float64[5,3]_update=float64[3,1]_start_indices=(1, 1)` | 25 | 24 |  |
| `test_dynamic_update_slice_operand=float64[7,5,3]_update=float64[2,0,1]_start_indices=(4, 1, 0)` | 8 | 6 | **LARGE DIFF** |
| `test_eig_shape=complex128[0,0]_computelefteigenvectors=False_computerighteigenvectors=False` | 7 | 0 |  |
| `test_eig_shape=complex128[0,0]_computelefteigenvectors=False_computerighteigenvectors=True` | 9 | 0 |  |
| `test_eig_shape=complex128[0,0]_computelefteigenvectors=True_computerighteigenvectors=False` | 9 | 0 |  |
| `test_eig_shape=complex128[0,0]_computelefteigenvectors=True_computerighteigenvectors=True` | 10 | 0 |  |
| `test_eig_shape=complex128[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=False` | 24 | 0 |  |
| `test_eig_shape=complex128[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=True` | 39 | 0 |  |
| `test_eig_shape=complex128[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=False` | 39 | 0 |  |
| `test_eig_shape=complex128[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=True` | 52 | 0 |  |
| `test_eig_shape=complex128[5,5]_computelefteigenvectors=False_computerighteigenvectors=False` | 23 | 0 |  |
| `test_eig_shape=complex128[5,5]_computelefteigenvectors=False_computerighteigenvectors=True` | 37 | 0 |  |
| `test_eig_shape=complex128[5,5]_computelefteigenvectors=True_computerighteigenvectors=False` | 37 | 0 |  |
| `test_eig_shape=complex128[5,5]_computelefteigenvectors=True_computerighteigenvectors=True` | 49 | 0 |  |
| `test_eig_shape=complex64[0,0]_computelefteigenvectors=False_computerighteigenvectors=False` | 7 | 0 |  |
| `test_eig_shape=complex64[0,0]_computelefteigenvectors=False_computerighteigenvectors=True` | 9 | 0 |  |
| `test_eig_shape=complex64[0,0]_computelefteigenvectors=True_computerighteigenvectors=False` | 9 | 0 |  |
| `test_eig_shape=complex64[0,0]_computelefteigenvectors=True_computerighteigenvectors=True` | 10 | 0 |  |
| `test_eig_shape=complex64[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=False` | 24 | 0 |  |
| `test_eig_shape=complex64[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=True` | 39 | 0 |  |
| `test_eig_shape=complex64[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=False` | 39 | 0 |  |
| `test_eig_shape=complex64[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=True` | 52 | 0 |  |
| `test_eig_shape=complex64[5,5]_computelefteigenvectors=False_computerighteigenvectors=False` | 23 | 0 |  |
| `test_eig_shape=complex64[5,5]_computelefteigenvectors=False_computerighteigenvectors=True` | 37 | 0 |  |
| `test_eig_shape=complex64[5,5]_computelefteigenvectors=True_computerighteigenvectors=False` | 37 | 0 |  |
| `test_eig_shape=complex64[5,5]_computelefteigenvectors=True_computerighteigenvectors=True` | 49 | 0 |  |
| `test_eig_shape=float32[0,0]_computelefteigenvectors=False_computerighteigenvectors=False` | 7 | 0 |  |
| `test_eig_shape=float32[0,0]_computelefteigenvectors=False_computerighteigenvectors=True` | 9 | 0 |  |
| `test_eig_shape=float32[0,0]_computelefteigenvectors=True_computerighteigenvectors=False` | 9 | 0 |  |
| `test_eig_shape=float32[0,0]_computelefteigenvectors=True_computerighteigenvectors=True` | 10 | 0 |  |
| `test_eig_shape=float32[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=False` | 27 | 0 |  |
| `test_eig_shape=float32[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=True` | 42 | 0 |  |
| `test_eig_shape=float32[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=False` | 42 | 0 |  |
| `test_eig_shape=float32[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=True` | 55 | 0 |  |
| `test_eig_shape=float32[5,5]_computelefteigenvectors=False_computerighteigenvectors=False` | 26 | 0 |  |
| `test_eig_shape=float32[5,5]_computelefteigenvectors=False_computerighteigenvectors=True` | 40 | 0 |  |
| `test_eig_shape=float32[5,5]_computelefteigenvectors=True_computerighteigenvectors=False` | 40 | 0 |  |
| `test_eig_shape=float32[5,5]_computelefteigenvectors=True_computerighteigenvectors=True` | 52 | 0 |  |
| `test_eig_shape=float64[0,0]_computelefteigenvectors=False_computerighteigenvectors=False` | 7 | 0 |  |
| `test_eig_shape=float64[0,0]_computelefteigenvectors=False_computerighteigenvectors=True` | 9 | 0 |  |
| `test_eig_shape=float64[0,0]_computelefteigenvectors=True_computerighteigenvectors=False` | 9 | 0 |  |
| `test_eig_shape=float64[0,0]_computelefteigenvectors=True_computerighteigenvectors=True` | 10 | 0 |  |
| `test_eig_shape=float64[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=False` | 27 | 0 |  |
| `test_eig_shape=float64[2,6,6]_computelefteigenvectors=False_computerighteigenvectors=True` | 42 | 0 |  |
| `test_eig_shape=float64[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=False` | 42 | 0 |  |
| `test_eig_shape=float64[2,6,6]_computelefteigenvectors=True_computerighteigenvectors=True` | 55 | 0 |  |
| `test_eig_shape=float64[5,5]_computelefteigenvectors=False_computerighteigenvectors=False` | 26 | 0 |  |
| `test_eig_shape=float64[5,5]_computelefteigenvectors=False_computerighteigenvectors=True` | 40 | 0 |  |
| `test_eig_shape=float64[5,5]_computelefteigenvectors=True_computerighteigenvectors=False` | 40 | 0 |  |
| `test_eig_shape=float64[5,5]_computelefteigenvectors=True_computerighteigenvectors=True` | 52 | 0 |  |
| `test_eigh_shape=bfloat16[0,0]_lower=False` | 9 | 8 |  |
| `test_eigh_shape=bfloat16[0,0]_lower=True` | 9 | 8 |  |
| `test_eigh_shape=bfloat16[2,20,20]_lower=False` | 38 | 450 | **LARGE DIFF** |
| `test_eigh_shape=bfloat16[2,20,20]_lower=True` | 38 | 449 | **LARGE DIFF** |
| `test_eigh_shape=bfloat16[50,50]_lower=False` | 36 | 430 | **LARGE DIFF** |
| `test_eigh_shape=bfloat16[50,50]_lower=True` | 35 | 429 | **LARGE DIFF** |
| `test_eigh_shape=complex128[0,0]_lower=False` | 9 | 8 |  |
| `test_eigh_shape=complex128[0,0]_lower=True` | 9 | 8 |  |
| `test_eigh_shape=complex128[2,20,20]_lower=False` | 38 | 0 |  |
| `test_eigh_shape=complex128[2,20,20]_lower=True` | 38 | 0 |  |
| `test_eigh_shape=complex128[50,50]_lower=False` | 36 | 0 |  |
| `test_eigh_shape=complex128[50,50]_lower=True` | 35 | 0 |  |
| `test_eigh_shape=complex64[0,0]_lower=False` | 9 | 8 |  |
| `test_eigh_shape=complex64[0,0]_lower=True` | 9 | 8 |  |
| `test_eigh_shape=complex64[2,20,20]_lower=False` | 38 | 0 |  |
| `test_eigh_shape=complex64[2,20,20]_lower=True` | 38 | 0 |  |
| `test_eigh_shape=complex64[50,50]_lower=False` | 36 | 0 |  |
| `test_eigh_shape=complex64[50,50]_lower=True` | 35 | 0 |  |
| `test_eigh_shape=float32[0,0]_lower=False` | 9 | 8 |  |
| `test_eigh_shape=float32[0,0]_lower=True` | 9 | 8 |  |
| `test_eigh_shape=float32[2,20,20]_lower=False` | 38 | 450 | **LARGE DIFF** |
| `test_eigh_shape=float32[2,20,20]_lower=True` | 38 | 449 | **LARGE DIFF** |
| `test_eigh_shape=float32[50,50]_lower=False` | 36 | 430 | **LARGE DIFF** |
| `test_eigh_shape=float32[50,50]_lower=True` | 35 | 429 | **LARGE DIFF** |
| `test_eigh_shape=float64[0,0]_lower=False` | 9 | 8 |  |
| `test_eigh_shape=float64[0,0]_lower=True` | 9 | 8 |  |
| `test_eigh_shape=float64[2,20,20]_lower=False` | 38 | 450 | **LARGE DIFF** |
| `test_eigh_shape=float64[2,20,20]_lower=True` | 38 | 449 | **LARGE DIFF** |
| `test_eigh_shape=float64[50,50]_lower=False` | 36 | 430 | **LARGE DIFF** |
| `test_eigh_shape=float64[50,50]_lower=True` | 35 | 429 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.FFT_fftlengths=(15, 16, 17)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.FFT_fftlengths=(16, 17)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.FFT_fftlengths=(17,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.IFFT_fftlengths=(15, 16, 17)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.IFFT_fftlengths=(16, 17)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.IFFT_fftlengths=(17,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.IRFFT_fftlengths=(15, 16, 32)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.IRFFT_fftlengths=(16, 32)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=complex64[14,15,16,17]_ffttype=FftType.IRFFT_fftlengths=(32,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=float32[14,15,16,17]_ffttype=FftType.RFFT_fftlengths=(15, 16, 17)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=float32[14,15,16,17]_ffttype=FftType.RFFT_fftlengths=(16, 17)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dims_shape=float32[14,15,16,17]_ffttype=FftType.RFFT_fftlengths=(17,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dtypes_shape=complex128[14,15,16,17]_ffttype=FftType.FFT_fftlengths=(17,)` | 7 | 0 |  |
| `test_fft_dtypes_shape=complex128[14,15,16,17]_ffttype=FftType.IFFT_fftlengths=(17,)` | 7 | 0 |  |
| `test_fft_dtypes_shape=complex128[14,15,16,17]_ffttype=FftType.IRFFT_fftlengths=(32,)` | 7 | 0 |  |
| `test_fft_dtypes_shape=complex64[14,15,16,17]_ffttype=FftType.FFT_fftlengths=(17,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dtypes_shape=complex64[14,15,16,17]_ffttype=FftType.IFFT_fftlengths=(17,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dtypes_shape=complex64[14,15,16,17]_ffttype=FftType.IRFFT_fftlengths=(32,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dtypes_shape=float32[14,15,16,17]_ffttype=FftType.RFFT_fftlengths=(17,)` | 7 | 5 | **LARGE DIFF** |
| `test_fft_dtypes_shape=float64[14,15,16,17]_ffttype=FftType.RFFT_fftlengths=(17,)` | 7 | 0 |  |
| `test_gather_from_take_indices_shape=()_axis=0` | 14 | 13 |  |
| `test_gather_from_take_indices_shape=()_axis=1` | 14 | 13 |  |
| `test_gather_from_take_indices_shape=()_axis=2` | 14 | 13 |  |
| `test_gather_from_take_indices_shape=(1,)_axis=0` | 14 | 13 |  |
| `test_gather_from_take_indices_shape=(1,)_axis=1` | 14 | 13 |  |
| `test_gather_from_take_indices_shape=(1,)_axis=2` | 14 | 13 |  |
| `test_gather_from_take_indices_shape=(2, 2)_axis=0` | 14 | 12 | **LARGE DIFF** |
| `test_gather_from_take_indices_shape=(2, 2)_axis=1` | 14 | 12 | **LARGE DIFF** |
| `test_gather_from_take_indices_shape=(2, 2)_axis=2` | 14 | 12 | **LARGE DIFF** |
| `test_gather_from_take_indices_shape=(2,)_axis=0` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(2,)_axis=1` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(2,)_axis=2` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(3,)_axis=0` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(3,)_axis=1` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(3,)_axis=2` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(4,)_axis=0` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(4,)_axis=1` | 13 | 12 |  |
| `test_gather_from_take_indices_shape=(4,)_axis=2` | 13 | 12 |  |
| `test_gather_rank_change` | 21 | 20 |  |
| `test_gather_shape=(10, 5)_idxs_shape=(2, 2)_dnums=GatherDimensionNumbers(offset_dims=(1,), collapsed`<br>`_slice_dims=(0,), start_index_map=(0, 1))_slice_sizes=(1, 3)` | 7 | 6 |  |
| `test_gather_shape=(10, 5)_idxs_shape=(3, 1)_dnums=GatherDimensionNumbers(offset_dims=(1,), collapsed`<br>`_slice_dims=(0,), start_index_map=(0,))_slice_sizes=(1, 3)` | 7 | 6 |  |
| `test_gather_shape=(10,)_idxs_shape=(3, 1)_dnums=GatherDimensionNumbers(offset_dims=(1,), collapsed_s`<br>`lice_dims=(), start_index_map=(0,))_slice_sizes=(2,)` | 7 | 6 |  |
| `test_gather_shape=(5,)_idxs_shape=(2, 1)_dnums=GatherDimensionNumbers(offset_dims=(), collapsed_slic`<br>`e_dims=(0,), start_index_map=(0,))_slice_sizes=(1,)` | 7 | 6 |  |
| `test_integer_pow_dtypes_shape=bfloat16[20,30]_y=3` | 17 | 16 |  |
| `test_integer_pow_dtypes_shape=complex128[20,30]_y=3` | 11 | 11 |  |
| `test_integer_pow_dtypes_shape=complex64[20,30]_y=3` | 11 | 11 |  |
| `test_integer_pow_dtypes_shape=float16[20,30]_y=3` | 11 | 10 |  |
| `test_integer_pow_dtypes_shape=float32[20,30]_y=3` | 11 | 10 |  |
| `test_integer_pow_dtypes_shape=float64[20,30]_y=3` | 11 | 10 |  |
| `test_integer_pow_dtypes_shape=int16[20,30]_y=3` | 11 | 0 |  |
| `test_integer_pow_dtypes_shape=int32[20,30]_y=3` | 11 | 10 |  |
| `test_integer_pow_dtypes_shape=int64[20,30]_y=3` | 11 | 10 |  |
| `test_integer_pow_dtypes_shape=int8[20,30]_y=3` | 11 | 0 |  |
| `test_integer_pow_dtypes_shape=uint16[20,30]_y=3` | 11 | 0 |  |
| `test_integer_pow_dtypes_shape=uint32[20,30]_y=3` | 11 | 0 |  |
| `test_integer_pow_dtypes_shape=uint64[20,30]_y=3` | 11 | 0 |  |
| `test_integer_pow_dtypes_shape=uint8[20,30]_y=3` | 11 | 0 |  |
| `test_integer_pow_negative_exp_shape=bfloat16[20,30]_y=-1000` | 73 | 16 | **LARGE DIFF** |
| `test_integer_pow_negative_exp_shape=complex128[20,30]_y=-1000` | 26 | 11 | **LARGE DIFF** |
| `test_integer_pow_negative_exp_shape=complex64[20,30]_y=-1000` | 26 | 11 | **LARGE DIFF** |
| `test_integer_pow_negative_exp_shape=float16[20,30]_y=-1000` | 26 | 11 | **LARGE DIFF** |
| `test_integer_pow_negative_exp_shape=float32[20,30]_y=-1000` | 26 | 11 | **LARGE DIFF** |
| `test_integer_pow_negative_exp_shape=float64[20,30]_y=-1000` | 26 | 11 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=bfloat16[20,30]_y=1000` | 65 | 16 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=complex128[20,30]_y=1000` | 23 | 11 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=complex64[20,30]_y=1000` | 23 | 11 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=float16[20,30]_y=1000` | 23 | 11 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=float32[20,30]_y=1000` | 23 | 11 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=float64[20,30]_y=1000` | 23 | 11 | **LARGE DIFF** |
| `test_integer_pow_overflow_shape=int16[20,30]_y=1000` | 23 | 0 |  |
| `test_integer_pow_overflow_shape=int8[20,30]_y=1000` | 23 | 0 |  |
| `test_integer_pow_overflow_shape=uint16[20,30]_y=1000` | 23 | 0 |  |
| `test_integer_pow_overflow_shape=uint32[20,30]_y=1000` | 23 | 0 |  |
| `test_integer_pow_overflow_shape=uint64[20,30]_y=1000` | 23 | 0 |  |
| `test_integer_pow_overflow_shape=uint8[20,30]_y=1000` | 23 | 0 |  |
| `test_iota_broadcasting_shape=float32[4,8,1,1]_dimension=1` | 5 | 5 |  |
| `test_iota_broadcasting_shape=float32[4,8,1,1]_dimension=2` | 6 | 5 |  |
| `test_iota_dtypes_shape=bfloat16[2,3]_dimension=0` | 6 | 5 |  |
| `test_iota_dtypes_shape=complex128[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=complex64[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=float16[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=float32[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=float64[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=int16[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=int32[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=int64[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=int8[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=uint16[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=uint32[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=uint64[2,3]_dimension=0` | 5 | 5 |  |
| `test_iota_dtypes_shape=uint8[2,3]_dimension=0` | 5 | 5 |  |
| `test_linear_solve_dtypes_a=float32[4,4]_b=float32[4]_solve=explicit_jacobian_solve_transposesolve=ex`<br>`plicit_jacobian_solve_symmetric=False` | 134 | 749 | **LARGE DIFF** |
| `test_linear_solve_dtypes_a=float64[4,4]_b=float64[4]_solve=explicit_jacobian_solve_transposesolve=ex`<br>`plicit_jacobian_solve_symmetric=False` | 134 | 749 | **LARGE DIFF** |
| `test_linear_solve_symmetric_a=float32[4,4]_b=float32[4]_solve=explicit_jacobian_solve_transposesolve`<br>`=explicit_jacobian_solve_symmetric=True` | 134 | 749 | **LARGE DIFF** |
| `test_linear_solve_transpose_solve_a=float32[4,4]_b=float32[4]_solve=explicit_jacobian_solve_transpos`<br>`esolve=None_symmetric=False` | 134 | 749 | **LARGE DIFF** |
| `test_lu_shape=complex128[3,5,5]` | 209 | 546 | **LARGE DIFF** |
| `test_lu_shape=complex128[3,5]` | 107 | 617 | **LARGE DIFF** |
| `test_lu_shape=complex128[5,5]` | 106 | 446 | **LARGE DIFF** |
| `test_lu_shape=complex64[3,5,5]` | 209 | 546 | **LARGE DIFF** |
| `test_lu_shape=complex64[3,5]` | 107 | 617 | **LARGE DIFF** |
| `test_lu_shape=complex64[5,5]` | 106 | 446 | **LARGE DIFF** |
| `test_lu_shape=float32[3,5,5]` | 209 | 542 | **LARGE DIFF** |
| `test_lu_shape=float32[3,5]` | 107 | 613 | **LARGE DIFF** |
| `test_lu_shape=float32[5,5]` | 106 | 442 | **LARGE DIFF** |
| `test_lu_shape=float64[3,5,5]` | 209 | 542 | **LARGE DIFF** |
| `test_lu_shape=float64[3,5]` | 107 | 613 | **LARGE DIFF** |
| `test_lu_shape=float64[5,5]` | 106 | 442 | **LARGE DIFF** |
| `test_min_max_fun=max_bfloat16` | 10 | 9 |  |
| `test_min_max_fun=max_bool` | 7 | 0 |  |
| `test_min_max_fun=max_complex128` | 20 | 0 |  |
| `test_min_max_fun=max_complex64` | 20 | 0 |  |
| `test_min_max_fun=max_float16` | 7 | 6 |  |
| `test_min_max_fun=max_float32` | 7 | 6 |  |
| `test_min_max_fun=max_float64` | 7 | 6 |  |
| `test_min_max_fun=max_inf_nan_bfloat16_-inf_nan` | 10 | 9 |  |
| `test_min_max_fun=max_inf_nan_bfloat16_inf_nan` | 10 | 9 |  |
| `test_min_max_fun=max_inf_nan_complex128_(-inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=max_inf_nan_complex128_(inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=max_inf_nan_complex64_(-inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=max_inf_nan_complex64_(inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=max_inf_nan_float16_-inf_nan` | 7 | 6 |  |
| `test_min_max_fun=max_inf_nan_float16_inf_nan` | 7 | 6 |  |
| `test_min_max_fun=max_inf_nan_float32_-inf_nan` | 7 | 6 |  |
| `test_min_max_fun=max_inf_nan_float32_inf_nan` | 7 | 6 |  |
| `test_min_max_fun=max_inf_nan_float64_-inf_nan` | 7 | 6 |  |
| `test_min_max_fun=max_inf_nan_float64_inf_nan` | 7 | 6 |  |
| `test_min_max_fun=max_int16` | 7 | 6 |  |
| `test_min_max_fun=max_int32` | 7 | 6 |  |
| `test_min_max_fun=max_int64` | 7 | 6 |  |
| `test_min_max_fun=max_int8` | 7 | 0 |  |
| `test_min_max_fun=max_uint16` | 7 | 0 |  |
| `test_min_max_fun=max_uint32` | 7 | 0 |  |
| `test_min_max_fun=max_uint64` | 7 | 0 |  |
| `test_min_max_fun=max_uint8` | 7 | 6 |  |
| `test_min_max_fun=min_bfloat16` | 10 | 9 |  |
| `test_min_max_fun=min_bool` | 7 | 0 |  |
| `test_min_max_fun=min_complex128` | 20 | 0 |  |
| `test_min_max_fun=min_complex64` | 20 | 0 |  |
| `test_min_max_fun=min_float16` | 7 | 6 |  |
| `test_min_max_fun=min_float32` | 7 | 6 |  |
| `test_min_max_fun=min_float64` | 7 | 6 |  |
| `test_min_max_fun=min_inf_nan_bfloat16_-inf_nan` | 10 | 9 |  |
| `test_min_max_fun=min_inf_nan_bfloat16_inf_nan` | 10 | 9 |  |
| `test_min_max_fun=min_inf_nan_complex128_(-inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=min_inf_nan_complex128_(inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=min_inf_nan_complex64_(-inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=min_inf_nan_complex64_(inf+0j)_(nan+0j)` | 20 | 0 |  |
| `test_min_max_fun=min_inf_nan_float16_-inf_nan` | 7 | 6 |  |
| `test_min_max_fun=min_inf_nan_float16_inf_nan` | 7 | 6 |  |
| `test_min_max_fun=min_inf_nan_float32_-inf_nan` | 7 | 6 |  |
| `test_min_max_fun=min_inf_nan_float32_inf_nan` | 7 | 6 |  |
| `test_min_max_fun=min_inf_nan_float64_-inf_nan` | 7 | 6 |  |
| `test_min_max_fun=min_inf_nan_float64_inf_nan` | 7 | 6 |  |
| `test_min_max_fun=min_int16` | 7 | 6 |  |
| `test_min_max_fun=min_int32` | 7 | 6 |  |
| `test_min_max_fun=min_int64` | 7 | 6 |  |
| `test_min_max_fun=min_int8` | 7 | 0 |  |
| `test_min_max_fun=min_uint16` | 7 | 0 |  |
| `test_min_max_fun=min_uint32` | 7 | 0 |  |
| `test_min_max_fun=min_uint64` | 7 | 0 |  |
| `test_min_max_fun=min_uint8` | 7 | 6 |  |
| `test_pad_inshape=bfloat16[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 9 | 8 |  |
| `test_pad_inshape=bfloat16[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 18 | 17 |  |
| `test_pad_inshape=bfloat16[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=bfloat16[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=bfloat16[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 10 | 9 |  |
| `test_pad_inshape=bfloat16[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 10 | 9 |  |
| `test_pad_inshape=bool[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=bool[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=bool[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=bool[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=bool[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=bool[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex128[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex128[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=complex128[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=complex128[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex128[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex128[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex64[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex64[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=complex64[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=complex64[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex64[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=complex64[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float16[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float16[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=float16[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=float16[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float16[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float16[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float32[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float32[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=float32[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=float32[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float32[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float32[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float64[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float64[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=float64[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=float64[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float64[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=float64[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int16[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int16[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=int16[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=int16[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int16[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int16[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int32[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int32[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=int32[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=int32[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int32[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int32[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int64[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int64[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=int64[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=int64[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int64[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int64[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int8[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int8[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=int8[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=int8[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int8[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=int8[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint16[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint16[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=uint16[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=uint16[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint16[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint16[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint32[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint32[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=uint32[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=uint32[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint32[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint32[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint64[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint64[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=uint64[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=uint64[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint64[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint64[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint8[2,3]_pads=[(0, 0, 0), (-1, -1, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint8[2,3]_pads=[(0, 0, 0), (-2, -2, 4)]` | 13 | 12 |  |
| `test_pad_inshape=uint8[2,3]_pads=[(0, 0, 0), (-2, -3, 1)]` | 8 | 7 |  |
| `test_pad_inshape=uint8[2,3]_pads=[(0, 0, 0), (0, 0, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint8[2,3]_pads=[(1, 1, 0), (2, 2, 0)]` | 7 | 6 |  |
| `test_pad_inshape=uint8[2,3]_pads=[(1, 2, 1), (0, 1, 0)]` | 7 | 6 |  |
| `test_population_count_int16` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_int32` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_int64` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_int8` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_uint16` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_uint32` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_uint64` | 6 | 11 | **LARGE DIFF** |
| `test_population_count_uint8` | 6 | 5 |  |
| `test_pow_broadcast_lhs=float32[4,1,6]_rhs=float32[4,5,6]` | 14 | 13 |  |
| `test_pow_broadcast_lhs=float32[4,5,6]_rhs=float32[4,1,6]` | 14 | 13 |  |
| `test_pow_broadcast_lhs=float32[4,5,6]_rhs=float32[]` | 13 | 12 |  |
| `test_pow_broadcast_lhs=float32[]_rhs=float32[4,5,6]` | 13 | 12 |  |
| `test_pow_dtypes_lhs=bfloat16[20,30]_rhs=bfloat16[20,30]` | 10 | 9 |  |
| `test_pow_dtypes_lhs=complex128[20,30]_rhs=complex128[20,30]` | 7 | 6 |  |
| `test_pow_dtypes_lhs=complex64[20,30]_rhs=complex64[20,30]` | 7 | 6 |  |
| `test_pow_dtypes_lhs=float16[20,30]_rhs=float16[20,30]` | 7 | 6 |  |
| `test_pow_dtypes_lhs=float32[20,30]_rhs=float32[20,30]` | 7 | 6 |  |
| `test_pow_dtypes_lhs=float64[20,30]_rhs=float64[20,30]` | 7 | 6 |  |
| `test_qr_multi_array_shape=complex128[1,1]_fullmatrices=False` | 48 | 76 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[1,1]_fullmatrices=True` | 48 | 76 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[2,10,5]_fullmatrices=False` | 60 | 483 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[2,10,5]_fullmatrices=True` | 68 | 481 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[2,200,100]_fullmatrices=False` | 70 | 519 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[2,200,100]_fullmatrices=True` | 88 | 521 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[3,3]_fullmatrices=False` | 55 | 430 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[3,3]_fullmatrices=True` | 55 | 430 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[3,4]_fullmatrices=False` | 64 | 447 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex128[3,4]_fullmatrices=True` | 64 | 447 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[1,1]_fullmatrices=False` | 48 | 76 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[1,1]_fullmatrices=True` | 48 | 76 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[2,10,5]_fullmatrices=False` | 60 | 483 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[2,10,5]_fullmatrices=True` | 68 | 481 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[2,200,100]_fullmatrices=False` | 60 | 487 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[2,200,100]_fullmatrices=True` | 78 | 489 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[3,3]_fullmatrices=False` | 55 | 430 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[3,3]_fullmatrices=True` | 55 | 430 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[3,4]_fullmatrices=False` | 64 | 447 | **LARGE DIFF** |
| `test_qr_multi_array_shape=complex64[3,4]_fullmatrices=True` | 64 | 447 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[1,1]_fullmatrices=False` | 48 | 8 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[1,1]_fullmatrices=True` | 48 | 8 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[2,10,5]_fullmatrices=False` | 60 | 361 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[2,10,5]_fullmatrices=True` | 68 | 359 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[2,200,100]_fullmatrices=False` | 60 | 362 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[2,200,100]_fullmatrices=True` | 68 | 360 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[3,3]_fullmatrices=False` | 55 | 322 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[3,3]_fullmatrices=True` | 55 | 322 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[3,4]_fullmatrices=False` | 64 | 339 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float32[3,4]_fullmatrices=True` | 64 | 339 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[1,1]_fullmatrices=False` | 48 | 8 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[1,1]_fullmatrices=True` | 48 | 8 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[2,10,5]_fullmatrices=False` | 60 | 361 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[2,10,5]_fullmatrices=True` | 68 | 359 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[2,200,100]_fullmatrices=False` | 60 | 365 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[2,200,100]_fullmatrices=True` | 78 | 367 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[3,3]_fullmatrices=False` | 55 | 322 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[3,3]_fullmatrices=True` | 55 | 322 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[3,4]_fullmatrices=False` | 64 | 339 | **LARGE DIFF** |
| `test_qr_multi_array_shape=float64[3,4]_fullmatrices=True` | 64 | 339 | **LARGE DIFF** |
| `test_random_gamma_shape=float32[3]` | 1945 | 2787 | **LARGE DIFF** |
| `test_random_gamma_shape=float32[]` | 1865 | 2468 | **LARGE DIFF** |
| `test_random_gamma_shape=float64[3]` | 1944 | 2783 | **LARGE DIFF** |
| `test_random_gamma_shape=float64[]` | 1862 | 2464 | **LARGE DIFF** |
| `test_random_split_i=0` | 195 | 221 | **LARGE DIFF** |
| `test_random_split_i=1` | 195 | 221 | **LARGE DIFF** |
| `test_random_split_i=2` | 195 | 221 | **LARGE DIFF** |
| `test_random_split_i=3` | 195 | 221 | **LARGE DIFF** |
| `test_random_split_i=4` | 195 | 221 | **LARGE DIFF** |
| `test_real_imag_dtypes_prim=imag_shape=complex128[2,3]` | 6 | 5 |  |
| `test_real_imag_dtypes_prim=imag_shape=complex64[2,3]` | 6 | 5 |  |
| `test_real_imag_dtypes_prim=real_shape=complex128[2,3]` | 6 | 5 |  |
| `test_real_imag_dtypes_prim=real_shape=complex64[2,3]` | 6 | 5 |  |
| `test_reduce_ops_with_boolean_input_all` | 12 | 11 |  |
| `test_reduce_ops_with_boolean_input_amax` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_ops_with_boolean_input_amin` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_ops_with_boolean_input_any` | 12 | 11 |  |
| `test_reduce_ops_with_boolean_input_prod` | 17 | 16 |  |
| `test_reduce_ops_with_boolean_input_sum` | 17 | 16 |  |
| `test_reduce_ops_with_numerical_input_all` | 17 | 18 |  |
| `test_reduce_ops_with_numerical_input_amax` | 12 | 11 |  |
| `test_reduce_ops_with_numerical_input_amin` | 12 | 11 |  |
| `test_reduce_ops_with_numerical_input_any` | 17 | 18 |  |
| `test_reduce_ops_with_numerical_input_prod` | 12 | 11 |  |
| `test_reduce_ops_with_numerical_input_sum` | 12 | 11 |  |
| `test_reduce_window_base_dilation_shape=float32[4,6]_initvalue=0_computation=add_windowdimensions=(2,`<br>`2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,2)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=bfloat16[4,6]_initvalue=-inf_computation=max_windowdimensions=(2,2)_`<br>`windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 15 | 27 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=bfloat16[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_win`<br>`dowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 15 | 14 |  |
| `test_reduce_window_dtypes_shape=bfloat16[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_win`<br>`dowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 15 | 14 |  |
| `test_reduce_window_dtypes_shape=bfloat16[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_win`<br>`dowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 15 | 14 |  |
| `test_reduce_window_dtypes_shape=bfloat16[4,6]_initvalue=inf_computation=min_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 15 | 14 |  |
| `test_reduce_window_dtypes_shape=bool[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_windows`<br>`trides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=bool[4,6]_initvalue=False_computation=max_windowdimensions=(2,2)_win`<br>`dowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=bool[4,6]_initvalue=True_computation=min_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=complex128[4,6]_initvalue=(-inf+0j)_computation=max_windowdimensions`<br>`=(2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 25 | 0 |  |
| `test_reduce_window_dtypes_shape=complex128[4,6]_initvalue=(inf+0j)_computation=min_windowdimensions=`<br>`(2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 25 | 0 |  |
| `test_reduce_window_dtypes_shape=complex128[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=complex128[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 25 | 0 |  |
| `test_reduce_window_dtypes_shape=complex128[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=complex64[4,6]_initvalue=(-inf+0j)_computation=max_windowdimensions=`<br>`(2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 25 | 0 |  |
| `test_reduce_window_dtypes_shape=complex64[4,6]_initvalue=(inf+0j)_computation=min_windowdimensions=(`<br>`2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 25 | 0 |  |
| `test_reduce_window_dtypes_shape=complex64[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=complex64[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 25 | 0 |  |
| `test_reduce_window_dtypes_shape=complex64[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float16[4,6]_initvalue=-inf_computation=max_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=float16[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 30 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=float16[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float16[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float16[4,6]_initvalue=inf_computation=min_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float32[4,6]_initvalue=-inf_computation=max_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=float32[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=float32[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float32[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float32[4,6]_initvalue=inf_computation=min_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float64[4,6]_initvalue=-inf_computation=max_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=float64[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=float64[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float64[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=float64[4,6]_initvalue=inf_computation=min_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int16[4,6]_initvalue=-32768_computation=max_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=int16[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int16[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int16[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int16[4,6]_initvalue=32767_computation=min_windowdimensions=(2,2)_wi`<br>`ndowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int32[4,6]_initvalue=-2147483648_computation=max_windowdimensions=(2`<br>`,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=int32[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int32[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int32[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int32[4,6]_initvalue=2147483647_computation=min_windowdimensions=(2,`<br>`2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int64[4,6]_initvalue=-9223372036854775808_computation=max_windowdime`<br>`nsions=(2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=int64[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int64[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int64[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int64[4,6]_initvalue=9223372036854775807_computation=min_windowdimen`<br>`sions=(2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int8[4,6]_initvalue=-128_computation=max_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=int8[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_windows`<br>`trides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=int8[4,6]_initvalue=127_computation=min_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=int8[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_windows`<br>`trides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=int8[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_windows`<br>`trides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=uint16[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint16[4,6]_initvalue=0_computation=max_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=uint16[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint16[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=uint16[4,6]_initvalue=65535_computation=min_windowdimensions=(2,2)_w`<br>`indowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint32[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint32[4,6]_initvalue=0_computation=max_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint32[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint32[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint32[4,6]_initvalue=4294967295_computation=min_windowdimensions=(2`<br>`,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint64[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint64[4,6]_initvalue=0_computation=max_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint64[4,6]_initvalue=18446744073709551615_computation=min_windowdim`<br>`ensions=(2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint64[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint64[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_windo`<br>`wstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 0 |  |
| `test_reduce_window_dtypes_shape=uint8[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=uint8[4,6]_initvalue=0_computation=max_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_dtypes_shape=uint8[4,6]_initvalue=1_computation=max_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=uint8[4,6]_initvalue=1_computation=mul_windowdimensions=(2,2)_window`<br>`strides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_dtypes_shape=uint8[4,6]_initvalue=255_computation=min_windowdimensions=(2,2)_wind`<br>`owstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_padding_shape=float32[4,6]_initvalue=0_computation=add_windowdimensions=(2,2)_win`<br>`dowstrides=(1,1)_padding=((1,2),(0,3))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_squeeze_dim_shape=float32[1,2,1]_initvalue=-inf_computation=max_windowdimensions=`<br>`(1,2,1)_windowstrides=(1,1,1)_padding=((0,0),(0,0),(0,0))_basedilation=(1,1,1)_windowdilation=(1,1,1`<br>`)` | 13 | 12 |  |
| `test_reduce_window_squeeze_dim_shape=float32[1,2]_initvalue=-inf_computation=max_windowdimensions=(1`<br>`,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 17 | 13 | **LARGE DIFF** |
| `test_reduce_window_squeeze_dim_shape=float32[1,4,3,2,1]_initvalue=-inf_computation=max_windowdimensi`<br>`ons=(1,2,2,2,1)_windowstrides=(1,1,1,1,1)_padding=((0,0),(0,0),(0,0),(0,0),(0,0))_basedilation=(1,1,`<br>`1,1,1)_windowdilation=(1,1,1,1,1)` | 12 | 11 |  |
| `test_reduce_window_squeeze_dim_shape=float32[2,1]_initvalue=-inf_computation=max_windowdimensions=(2`<br>`,1)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 13 | 13 |  |
| `test_reduce_window_squeeze_dim_shape=float32[2,4,3]_initvalue=-inf_computation=max_windowdimensions=`<br>`(2,2,2)_windowstrides=(1,1,1)_padding=((0,0),(0,0),(0,0))_basedilation=(1,1,1)_windowdilation=(1,1,1`<br>`)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_squeeze_dim_shape=float32[2,4]_initvalue=-inf_computation=max_windowdimensions=(2`<br>`,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reduce_window_squeeze_dim_shape=float32[2]_initvalue=-inf_computation=max_windowdimensions=(2,)`<br>`_windowstrides=(1,)_padding=((0,0),)_basedilation=(1,)_windowdilation=(1,)` | 17 | 13 | **LARGE DIFF** |
| `test_reduce_window_window_dilation_shape=float32[4,6]_initvalue=0_computation=add_windowdimensions=(`<br>`2,2)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,2)` | 12 | 11 |  |
| `test_reduce_window_window_dimensions_shape=float32[4,6]_initvalue=0_computation=add_windowdimensions`<br>`=(1,1)_windowstrides=(1,1)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 11 |  |
| `test_reduce_window_window_strides_shape=float32[4,6]_initvalue=0_computation=add_windowdimensions=(2`<br>`,2)_windowstrides=(1,2)_padding=((0,0),(0,0))_basedilation=(1,1)_windowdilation=(1,1)` | 12 | 18 | **LARGE DIFF** |
| `test_reducers_dtypes_prime=reduce_and_shape=bool[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=bfloat16[2,3]` | 15 | 14 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=bool[2,3]` | 12 | 19 | **LARGE DIFF** |
| `test_reducers_dtypes_prime=reduce_max_shape=complex128[2,3]` | 25 | 0 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=complex64[2,3]` | 25 | 0 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=float16[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=float32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=float64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=int16[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=int32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=int64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=int8[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=uint16[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=uint32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=uint64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_max_shape=uint8[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=bfloat16[2,3]` | 15 | 14 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=bool[2,3]` | 12 | 19 | **LARGE DIFF** |
| `test_reducers_dtypes_prime=reduce_min_shape=complex128[2,3]` | 25 | 0 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=complex64[2,3]` | 25 | 0 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=float16[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=float32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=float64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=int16[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=int32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=int64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=int8[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=uint16[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=uint32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=uint64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_min_shape=uint8[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_or_shape=bool[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=bfloat16[2,3]` | 15 | 13 | **LARGE DIFF** |
| `test_reducers_dtypes_prime=reduce_prod_shape=complex128[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=complex64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=float16[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=float32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=float64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=int16[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=int32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=int64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=int8[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=uint16[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=uint32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=uint64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_prod_shape=uint8[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=bfloat16[2,3]` | 15 | 13 | **LARGE DIFF** |
| `test_reducers_dtypes_prime=reduce_sum_shape=complex128[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=complex64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=float16[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=float32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=float64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=int16[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=int32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=int64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=int8[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=uint16[2,3]` | 12 | 13 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=uint32[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=uint64[2,3]` | 12 | 11 |  |
| `test_reducers_dtypes_prime=reduce_sum_shape=uint8[2,3]` | 12 | 13 |  |
| `test_reshape_dimensions_shape=float32[3,4,5]_newsizes=(3, 20)_dimensions=(2, 0, 1)` | 11 | 10 |  |
| `test_reshape_dimensions_shape=float32[3,4,5]_newsizes=(3, 20)_dimensions=(2, 1, 0)` | 11 | 10 |  |
| `test_reshape_dtypes_shape=bfloat16[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 8 | 7 |  |
| `test_reshape_dtypes_shape=bool[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=complex128[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=complex64[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=float16[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=float32[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=float64[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=int16[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=int32[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=int64[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=int8[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=uint16[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=uint32[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=uint64[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_dtypes_shape=uint8[2,3]_newsizes=(3, 2)_dimensions=(0, 1)` | 7 | 6 |  |
| `test_reshape_new_sizes_shape=float32[3,4,5]_newsizes=(3, 20)_dimensions=(0, 1, 2)` | 7 | 6 |  |
| `test_reshape_new_sizes_shape=float32[3,4,5]_newsizes=(4, 15)_dimensions=(0, 1, 2)` | 7 | 6 |  |
| `test_rev_dimensions_shape=float32[3,4,5]_dimensions=()` | 6 | 5 |  |
| `test_rev_dimensions_shape=float32[3,4,5]_dimensions=(0, 1, 2)` | 6 | 5 |  |
| `test_rev_dimensions_shape=float32[3,4,5]_dimensions=(0, 2)` | 6 | 5 |  |
| `test_rev_dimensions_shape=float32[3,4,5]_dimensions=(2, 0, 1)` | 6 | 5 |  |
| `test_rev_dtypes_shape=bfloat16[4,5]_dimensions=(0,)` | 8 | 7 |  |
| `test_rev_dtypes_shape=bool[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=complex128[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=complex64[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=float16[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=float32[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=float64[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=int16[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=int32[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=int64[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=int8[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=uint16[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_rev_dtypes_shape=uint32[4,5]_dimensions=(0,)` | 6 | 0 |  |
| `test_rev_dtypes_shape=uint64[4,5]_dimensions=(0,)` | 6 | 0 |  |
| `test_rev_dtypes_shape=uint8[4,5]_dimensions=(0,)` | 6 | 5 |  |
| `test_round_dtypes_shape=bfloat16[100,100]_roundingmethod=0` | 8 | 94 | **LARGE DIFF** |
| `test_round_dtypes_shape=float16[100,100]_roundingmethod=0` | 6 | 39 | **LARGE DIFF** |
| `test_round_dtypes_shape=float32[100,100]_roundingmethod=0` | 6 | 39 | **LARGE DIFF** |
| `test_round_dtypes_shape=float32[2,3]_roundingmethod=0` | 6 | 35 | **LARGE DIFF** |
| `test_round_dtypes_shape=float32[2,3]_roundingmethod=1` | 28 | 27 |  |
| `test_round_dtypes_shape=float64[100,100]_roundingmethod=0` | 6 | 39 | **LARGE DIFF** |
| `test_round_edge_case_round_away_from_0_shape=float32[2,3]_roundingmethod=0` | 6 | 35 | **LARGE DIFF** |
| `test_scatter_dtypes_fun=scatter_min_shape=bfloat16[5]_scatterindices=[[0],[2]]_updateshape=(2,)_upda`<br>`tewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniquei`<br>`ndices=False` | 61 | 60 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=bool[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updatewi`<br>`ndowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueindic`<br>`es=False` | 47 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=complex128[5]_scatterindices=[[0],[2]]_updateshape=(2,)_up`<br>`datewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqu`<br>`eindices=False` | 65 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=complex64[5]_scatterindices=[[0],[2]]_updateshape=(2,)_upd`<br>`atewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_unique`<br>`indices=False` | 65 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=float16[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updat`<br>`ewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniquein`<br>`dices=False` | 47 | 46 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=float32[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updat`<br>`ewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniquein`<br>`dices=False` | 47 | 46 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=float64[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updat`<br>`ewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniquein`<br>`dices=False` | 47 | 46 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=int16[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updatew`<br>`indowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueindi`<br>`ces=False` | 47 | 46 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=int32[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updatew`<br>`indowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueindi`<br>`ces=False` | 47 | 46 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=int64[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updatew`<br>`indowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueindi`<br>`ces=False` | 47 | 46 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=int8[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updatewi`<br>`ndowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueindic`<br>`es=False` | 47 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=uint16[5]_scatterindices=[[0],[2]]_updateshape=(2,)_update`<br>`windowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueind`<br>`ices=False` | 47 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=uint32[5]_scatterindices=[[0],[2]]_updateshape=(2,)_update`<br>`windowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueind`<br>`ices=False` | 47 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=uint64[5]_scatterindices=[[0],[2]]_updateshape=(2,)_update`<br>`windowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueind`<br>`ices=False` | 47 | 0 |  |
| `test_scatter_dtypes_fun=scatter_min_shape=uint8[5]_scatterindices=[[0],[2]]_updateshape=(2,)_updatew`<br>`indowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_uniqueindi`<br>`ces=False` | 47 | 46 |  |
| `test_scatter_indices_are_sorted_fun=scatter_min_shape=float32[5]_scatterindices=[[0],[2]]_updateshap`<br>`e=(2,)_updatewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=Tr`<br>`ue_uniqueindices=False` | 47 | 46 |  |
| `test_scatter_shapes_and_dimension_numbers_fun=scatter_min_shape=float32[10,5]_scatterindices=[[0],[2`<br>`],[1]]_updateshape=(3,3)_updatewindowdims=(1,)_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)`<br>`_indicesaresorted=False_uniqueindices=False` | 71 | 70 |  |
| `test_scatter_shapes_and_dimension_numbers_fun=scatter_min_shape=float32[10]_scatterindices=[[0],[0],`<br>`[0]]_updateshape=(3,2)_updatewindowdims=(1,)_insertedwindowdims=()_scatterdimstooperanddims=(0,)_ind`<br>`icesaresorted=False_uniqueindices=False` | 54 | 53 |  |
| `test_scatter_static_index_add` | 79 | 78 |  |
| `test_scatter_static_index_max` | 79 | 78 |  |
| `test_scatter_static_index_min` | 79 | 78 |  |
| `test_scatter_static_index_mul` | 79 | 78 |  |
| `test_scatter_static_index_update` | 78 | 77 |  |
| `test_scatter_unique_indices_fun=scatter_min_shape=float32[5]_scatterindices=[[0],[2]]_updateshape=(2`<br>`,)_updatewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False_`<br>`uniqueindices=False` | 47 | 46 |  |
| `test_scatter_update_function_fun=scatter_add_shape=float32[5]_scatterindices=[[0],[2]]_updateshape=(`<br>`2,)_updatewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False`<br>`_uniqueindices=False` | 47 | 46 |  |
| `test_scatter_update_function_fun=scatter_max_shape=float32[5]_scatterindices=[[0],[2]]_updateshape=(`<br>`2,)_updatewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False`<br>`_uniqueindices=False` | 47 | 46 |  |
| `test_scatter_update_function_fun=scatter_mul_shape=float32[5]_scatterindices=[[0],[2]]_updateshape=(`<br>`2,)_updatewindowdims=()_insertedwindowdims=(0,)_scatterdimstooperanddims=(0,)_indicesaresorted=False`<br>`_uniqueindices=False` | 47 | 46 |  |
| `test_select_and_gather_add_dilations_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 2)_window`<br>`strides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(2, 3)` | 45 | 44 |  |
| `test_select_and_gather_add_dilations_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 2)_window`<br>`strides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(2, 3)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_gather_add_dilations_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 2)_window`<br>`strides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(2, 3)_windowdilation=(3, 2)` | 45 | 44 |  |
| `test_select_and_gather_add_dtypes_shape=bfloat16[4,6]_selectprim=le_windowdimensions=(2, 2)_windowst`<br>`rides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 47 | 46 |  |
| `test_select_and_gather_add_dtypes_shape=float16[4,6]_selectprim=le_windowdimensions=(2, 2)_windowstr`<br>`ides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_gather_add_dtypes_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 2)_windowstr`<br>`ides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_gather_add_dtypes_shape=float64[4,6]_selectprim=le_windowdimensions=(2, 2)_windowstr`<br>`ides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 0 |  |
| `test_select_and_gather_add_padding_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 2)_windowst`<br>`rides=(1, 1)_padding=((0, 1), (0, 1))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_gather_add_select_prim_shape=float32[4,6]_selectprim=ge_windowdimensions=(2, 2)_wind`<br>`owstrides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_gather_add_window_dimensions_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 3`<br>`)_windowstrides=(1, 1)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_gather_add_window_strides_shape=float32[4,6]_selectprim=le_windowdimensions=(2, 2)_w`<br>`indowstrides=(2, 3)_padding=((0, 0), (0, 0))_basedilation=(1, 1)_windowdilation=(1, 1)` | 45 | 44 |  |
| `test_select_and_scatter_add_dtypes_shape=bfloat16[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_wi`<br>`ndowstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 23 | 22 |  |
| `test_select_and_scatter_add_dtypes_shape=bool[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_window`<br>`strides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 26 | **LARGE DIFF** |
| `test_select_and_scatter_add_dtypes_shape=float16[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_win`<br>`dowstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=float32[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_win`<br>`dowstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=float64[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_win`<br>`dowstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=int16[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_windo`<br>`wstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=int32[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_windo`<br>`wstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=int64[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_windo`<br>`wstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=int8[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_window`<br>`strides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_dtypes_shape=uint16[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_wind`<br>`owstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 0 |  |
| `test_select_and_scatter_add_dtypes_shape=uint32[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_wind`<br>`owstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 0 |  |
| `test_select_and_scatter_add_dtypes_shape=uint64[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_wind`<br>`owstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 0 |  |
| `test_select_and_scatter_add_dtypes_shape=uint8[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_windo`<br>`wstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_padding_shape=float32[2,4,6]_selectprim=ge_windowdimensions=(2, 2, 2)_wi`<br>`ndowstrides=(1, 1, 1)_padding=((1, 1), (1, 1), (1, 1))` | 18 | 17 |  |
| `test_select_and_scatter_add_select_prim_shape=float32[2,4,6]_selectprim=le_windowdimensions=(2, 2, 2`<br>`)_windowstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=bfloat16[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1`<br>`)_windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 23 | 22 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=float16[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)`<br>`_windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=float32[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)`<br>`_windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=float64[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)`<br>`_windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=int16[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)_w`<br>`indowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=int32[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)_w`<br>`indowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=int64[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)_w`<br>`indowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=uint16[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)_`<br>`windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 0 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=uint32[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)_`<br>`windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 0 |  |
| `test_select_and_scatter_add_tpu_dtypes_shape=uint64[2,4,6]_selectprim=ge_windowdimensions=(1, 3, 1)_`<br>`windowstrides=(1, 2, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 0 |  |
| `test_select_and_scatter_add_window_dimensions_shape=float32[2,4,6]_selectprim=ge_windowdimensions=(1`<br>`, 2, 3)_windowstrides=(1, 1, 1)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_and_scatter_add_window_strides_shape=float32[2,4,6]_selectprim=ge_windowdimensions=(2, 2`<br>`, 2)_windowstrides=(1, 2, 3)_padding=((0, 0), (0, 0), (0, 0))` | 18 | 17 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=bfloat16[2,3]` | 11 | 10 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=bool[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=complex128[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=complex64[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=float16[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=float32[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=float64[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=int16[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=int32[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=int64[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=int8[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=uint16[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=uint32[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=uint64[2,3]` | 8 | 7 |  |
| `test_select_dtypes_shapepred=bool[2,3]_shapeargs=uint8[2,3]` | 8 | 7 |  |
| `test_select_shapes_shapepred=bool[]_shapeargs=float32[18]` | 15 | 14 |  |
| `test_shift_left_dtype=int16_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int16_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int16_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=int16_shift_amount=16` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int16_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=int16_shift_amount=32` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int16_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=int16_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int16_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=int16_shift_amount=8` | 12 | 11 |  |
| `test_shift_left_dtype=int32_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int32_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int32_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=int32_shift_amount=16` | 12 | 11 |  |
| `test_shift_left_dtype=int32_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=int32_shift_amount=32` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int32_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=int32_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int32_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=int32_shift_amount=8` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int64_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int64_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=16` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=32` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int64_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=int64_shift_amount=8` | 12 | 11 |  |
| `test_shift_left_dtype=int8_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int8_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int8_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=int8_shift_amount=16` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int8_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=int8_shift_amount=32` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int8_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=int8_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=int8_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=int8_shift_amount=8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint16_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint16_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint16_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=uint16_shift_amount=16` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint16_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=uint16_shift_amount=32` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint16_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=uint16_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint16_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=uint16_shift_amount=8` | 12 | 11 |  |
| `test_shift_left_dtype=uint32_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint32_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint32_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=uint32_shift_amount=16` | 12 | 11 |  |
| `test_shift_left_dtype=uint32_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=uint32_shift_amount=32` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint32_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=uint32_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint32_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=uint32_shift_amount=8` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint64_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint64_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=16` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=32` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint64_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=uint64_shift_amount=8` | 12 | 11 |  |
| `test_shift_left_dtype=uint8_shift_amount=-1` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint8_shift_amount=-8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint8_shift_amount=0` | 12 | 11 |  |
| `test_shift_left_dtype=uint8_shift_amount=16` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint8_shift_amount=1` | 12 | 11 |  |
| `test_shift_left_dtype=uint8_shift_amount=32` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint8_shift_amount=3` | 12 | 11 |  |
| `test_shift_left_dtype=uint8_shift_amount=64` | 12 | 6 | **LARGE DIFF** |
| `test_shift_left_dtype=uint8_shift_amount=7` | 12 | 11 |  |
| `test_shift_left_dtype=uint8_shift_amount=8` | 12 | 6 | **LARGE DIFF** |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=-1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=-8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=0` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=16` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=32` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=3` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=64` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=7` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int16_shift_amount=8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=-1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=-8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=0` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=16` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=32` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=3` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=64` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=7` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int32_shift_amount=8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=-1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=-8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=0` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=16` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=32` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=3` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=64` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=7` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int64_shift_amount=8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=-1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=-8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=0` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=16` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=1` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=32` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=3` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=64` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=7` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=int8_shift_amount=8` | 12 | 11 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=-1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=-8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=0` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=16` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=32` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=3` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=64` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=7` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint16_shift_amount=8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=-1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=-8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=0` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=16` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=32` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=3` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=64` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=7` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint32_shift_amount=8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=-1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=-8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=0` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=16` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=32` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=3` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=64` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=7` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint64_shift_amount=8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=-1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=-8` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=0` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=16` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=1` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=32` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=3` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=64` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=7` | 12 | 13 |  |
| `test_shift_right_arithmetic_dtype=uint8_shift_amount=8` | 12 | 13 |  |
| `test_shift_right_logical_dtype=int16_shift_amount=-1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=-8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=0` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=16` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=32` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=3` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=64` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=7` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int16_shift_amount=8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=-1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=-8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=0` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=16` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=32` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=3` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=64` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=7` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int32_shift_amount=8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=-1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=-8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=0` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=16` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=32` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=3` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=64` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=7` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int64_shift_amount=8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=-1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=-8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=0` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=16` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=1` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=32` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=3` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=64` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=7` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=int8_shift_amount=8` | 13 | 23 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=-1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=-8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=0` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=16` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=32` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=3` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=64` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=7` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint16_shift_amount=8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=-1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=-8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=0` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=16` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=32` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=3` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=64` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=7` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint32_shift_amount=8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=-1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=-8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=0` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=16` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=32` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=3` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=64` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=7` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint64_shift_amount=8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=-1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=-8` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=0` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=16` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=1` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=32` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=3` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=64` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=7` | 13 | 22 | **LARGE DIFF** |
| `test_shift_right_logical_dtype=uint8_shift_amount=8` | 13 | 22 | **LARGE DIFF** |
| `test_slice_shape=(3,)_start_indices=(1,)_limit_indices=(2,)_strides=None` | 6 | 5 |  |
| `test_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(2, 1)_strides=(1, 1)` | 7 | 6 |  |
| `test_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(3, 1)_strides=None` | 7 | 6 |  |
| `test_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(3, 2)_strides=None` | 6 | 5 |  |
| `test_slice_shape=(5, 3)_start_indices=(1, 1)_limit_indices=(5, 3)_strides=(2, 1)` | 6 | 5 |  |
| `test_slice_shape=(5,)_start_indices=(1,)_limit_indices=(5,)_strides=(2,)` | 6 | 5 |  |
| `test_slice_shape=(7, 5, 3)_start_indices=(4, 0, 1)_limit_indices=(7, 1, 3)_strides=None` | 6 | 5 |  |
| `test_slice_shape=(7,)_start_indices=(4,)_limit_indices=(7,)_strides=None` | 6 | 5 |  |
| `test_slice_shape=(8,)_start_indices=(1,)_limit_indices=(6,)_strides=(2,)` | 6 | 5 |  |
| `test_sort_dimensions_nbarrays=1_shape=float32[5,7]_axis=1_isstable=False` | 30 | 29 |  |
| `test_sort_dtypes_nbarrays=1_shape=bfloat16[5,7]_axis=0_isstable=False` | 13 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=bool[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=complex128[5,7]_axis=0_isstable=False` | 50 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=complex64[5,7]_axis=0_isstable=False` | 50 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=float16[5,7]_axis=0_isstable=False` | 30 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=float32[5,7]_axis=0_isstable=False` | 30 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=float64[5,7]_axis=0_isstable=False` | 30 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=int16[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=int32[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=int64[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=int8[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=uint16[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=uint32[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=uint64[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_dtypes_nbarrays=1_shape=uint8[5,7]_axis=0_isstable=False` | 11 | 0 |  |
| `test_sort_edge_cases_nbarrays=1_shape=float32[5]_axis=-1_isstable=False` | 30 | 0 |  |
| `test_sort_is_stable_nbarrays=1_shape=float32[5,7]_axis=0_isstable=True` | 30 | 0 |  |
| `test_sort_multiple_arrays_nbarrays=2_shape=bool[5,7]_axis=0_isstable=False` | 13 | 0 |  |
| `test_sort_multiple_arrays_nbarrays=2_shape=float32[5,7]_axis=0_isstable=False` | 32 | 0 |  |
| `test_sort_multiple_arrays_nbarrays=3_shape=float32[5,7]_axis=0_isstable=False` | 35 | 0 |  |
| `test_squeeze_inshape=float32[1]_dimensions=(-1,)` | 7 | 6 |  |
| `test_squeeze_inshape=float32[1]_dimensions=(0,)` | 7 | 6 |  |
| `test_squeeze_inshape=float32[2,1,3,1]_dimensions=(1, -1)` | 7 | 6 |  |
| `test_squeeze_inshape=float32[2,1,3,1]_dimensions=(1, 3)` | 7 | 6 |  |
| `test_squeeze_inshape=float32[2,1,3,1]_dimensions=(1,)` | 7 | 10 | **LARGE DIFF** |
| `test_squeeze_inshape=float32[2,1,3,1]_dimensions=(3,)` | 7 | 10 | **LARGE DIFF** |
| `test_squeeze_inshape=float32[2,1,4]_dimensions=(-2,)` | 7 | 6 |  |
| `test_squeeze_inshape=float32[2,1,4]_dimensions=(1,)` | 7 | 6 |  |
| `test_stop_gradient_bfloat16[20,20]` | 6 | 5 |  |
| `test_stop_gradient_bool[20,20]` | 6 | 5 |  |
| `test_stop_gradient_complex128[20,20]` | 6 | 5 |  |
| `test_stop_gradient_complex64[20,20]` | 6 | 5 |  |
| `test_stop_gradient_float16[20,20]` | 6 | 5 |  |
| `test_stop_gradient_float32[20,20]` | 6 | 5 |  |
| `test_stop_gradient_float64[20,20]` | 6 | 5 |  |
| `test_stop_gradient_int16[20,20]` | 6 | 5 |  |
| `test_stop_gradient_int32[20,20]` | 6 | 5 |  |
| `test_stop_gradient_int64[20,20]` | 6 | 5 |  |
| `test_stop_gradient_int8[20,20]` | 6 | 5 |  |
| `test_stop_gradient_uint16[20,20]` | 6 | 5 |  |
| `test_stop_gradient_uint32[20,20]` | 6 | 5 |  |
| `test_stop_gradient_uint64[20,20]` | 6 | 5 |  |
| `test_stop_gradient_uint8[20,20]` | 6 | 5 |  |
| `test_sub_broadcasting_lhs=float32[1,2]_rhs=float32[3,2]` | 14 | 13 |  |
| `test_sub_broadcasting_lhs=float32[]_rhs=float32[2,3]` | 13 | 12 |  |
| `test_sub_dtypes_lhs=bfloat16[2,3]_rhs=bfloat16[2,3]` | 10 | 9 |  |
| `test_sub_dtypes_lhs=complex128[2,3]_rhs=complex128[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=complex64[2,3]_rhs=complex64[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=float16[2,3]_rhs=float16[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=float32[2,3]_rhs=float32[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=float64[2,3]_rhs=float64[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=int16[2,3]_rhs=int16[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=int32[2,3]_rhs=int32[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=int64[2,3]_rhs=int64[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=int8[2,3]_rhs=int8[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=uint16[2,3]_rhs=uint16[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=uint32[2,3]_rhs=uint32[2,3]` | 7 | 6 |  |
| `test_sub_dtypes_lhs=uint64[2,3]_rhs=uint64[2,3]` | 7 | 0 |  |
| `test_sub_dtypes_lhs=uint8[2,3]_rhs=uint8[2,3]` | 7 | 6 |  |
| `test_svd_shape=complex128[2,2]_fullmatrices=False_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex128[2,2]_fullmatrices=False_computeuv=True` | 50 | 0 |  |
| `test_svd_shape=complex128[2,2]_fullmatrices=True_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex128[2,2]_fullmatrices=True_computeuv=True` | 49 | 0 |  |
| `test_svd_shape=complex128[2,3,29,7]_fullmatrices=False_computeuv=False` | 26 | 0 |  |
| `test_svd_shape=complex128[2,3,29,7]_fullmatrices=False_computeuv=True` | 55 | 0 |  |
| `test_svd_shape=complex128[2,3,29,7]_fullmatrices=True_computeuv=False` | 27 | 0 |  |
| `test_svd_shape=complex128[2,3,29,7]_fullmatrices=True_computeuv=True` | 54 | 0 |  |
| `test_svd_shape=complex128[2,3,53]_fullmatrices=False_computeuv=False` | 26 | 0 |  |
| `test_svd_shape=complex128[2,3,53]_fullmatrices=False_computeuv=True` | 55 | 0 |  |
| `test_svd_shape=complex128[2,3,53]_fullmatrices=True_computeuv=False` | 27 | 0 |  |
| `test_svd_shape=complex128[2,3,53]_fullmatrices=True_computeuv=True` | 54 | 0 |  |
| `test_svd_shape=complex128[2,7]_fullmatrices=False_computeuv=False` | 25 | 0 |  |
| `test_svd_shape=complex128[2,7]_fullmatrices=False_computeuv=True` | 51 | 0 |  |
| `test_svd_shape=complex128[2,7]_fullmatrices=True_computeuv=False` | 25 | 0 |  |
| `test_svd_shape=complex128[2,7]_fullmatrices=True_computeuv=True` | 50 | 0 |  |
| `test_svd_shape=complex128[29,29]_fullmatrices=False_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex128[29,29]_fullmatrices=False_computeuv=True` | 50 | 0 |  |
| `test_svd_shape=complex128[29,29]_fullmatrices=True_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex128[29,29]_fullmatrices=True_computeuv=True` | 49 | 0 |  |
| `test_svd_shape=complex64[2,2]_fullmatrices=False_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex64[2,2]_fullmatrices=False_computeuv=True` | 50 | 0 |  |
| `test_svd_shape=complex64[2,2]_fullmatrices=True_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex64[2,2]_fullmatrices=True_computeuv=True` | 49 | 0 |  |
| `test_svd_shape=complex64[2,3,29,7]_fullmatrices=False_computeuv=False` | 26 | 0 |  |
| `test_svd_shape=complex64[2,3,29,7]_fullmatrices=False_computeuv=True` | 55 | 0 |  |
| `test_svd_shape=complex64[2,3,29,7]_fullmatrices=True_computeuv=False` | 27 | 0 |  |
| `test_svd_shape=complex64[2,3,29,7]_fullmatrices=True_computeuv=True` | 54 | 0 |  |
| `test_svd_shape=complex64[2,3,53]_fullmatrices=False_computeuv=False` | 26 | 0 |  |
| `test_svd_shape=complex64[2,3,53]_fullmatrices=False_computeuv=True` | 55 | 0 |  |
| `test_svd_shape=complex64[2,3,53]_fullmatrices=True_computeuv=False` | 27 | 0 |  |
| `test_svd_shape=complex64[2,3,53]_fullmatrices=True_computeuv=True` | 54 | 0 |  |
| `test_svd_shape=complex64[2,7]_fullmatrices=False_computeuv=False` | 25 | 0 |  |
| `test_svd_shape=complex64[2,7]_fullmatrices=False_computeuv=True` | 51 | 0 |  |
| `test_svd_shape=complex64[2,7]_fullmatrices=True_computeuv=False` | 25 | 0 |  |
| `test_svd_shape=complex64[2,7]_fullmatrices=True_computeuv=True` | 50 | 0 |  |
| `test_svd_shape=complex64[29,29]_fullmatrices=False_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex64[29,29]_fullmatrices=False_computeuv=True` | 50 | 0 |  |
| `test_svd_shape=complex64[29,29]_fullmatrices=True_computeuv=False` | 24 | 0 |  |
| `test_svd_shape=complex64[29,29]_fullmatrices=True_computeuv=True` | 49 | 0 |  |
| `test_svd_shape=float32[2,2]_fullmatrices=False_computeuv=False` | 24 | 875 | **LARGE DIFF** |
| `test_svd_shape=float32[2,2]_fullmatrices=False_computeuv=True` | 50 | 957 | **LARGE DIFF** |
| `test_svd_shape=float32[2,2]_fullmatrices=True_computeuv=False` | 24 | 875 | **LARGE DIFF** |
| `test_svd_shape=float32[2,2]_fullmatrices=True_computeuv=True` | 49 | 957 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,29,7]_fullmatrices=False_computeuv=False` | 26 | 1433 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,29,7]_fullmatrices=False_computeuv=True` | 55 | 1541 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,29,7]_fullmatrices=True_computeuv=False` | 27 | 1433 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,29,7]_fullmatrices=True_computeuv=True` | 54 | 1540 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,53]_fullmatrices=False_computeuv=False` | 26 | 1416 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,53]_fullmatrices=False_computeuv=True` | 55 | 1525 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,53]_fullmatrices=True_computeuv=False` | 27 | 1416 | **LARGE DIFF** |
| `test_svd_shape=float32[2,3,53]_fullmatrices=True_computeuv=True` | 54 | 1524 | **LARGE DIFF** |
| `test_svd_shape=float32[2,7]_fullmatrices=False_computeuv=False` | 25 | 885 | **LARGE DIFF** |
| `test_svd_shape=float32[2,7]_fullmatrices=False_computeuv=True` | 51 | 993 | **LARGE DIFF** |
| `test_svd_shape=float32[2,7]_fullmatrices=True_computeuv=False` | 25 | 885 | **LARGE DIFF** |
| `test_svd_shape=float32[2,7]_fullmatrices=True_computeuv=True` | 50 | 992 | **LARGE DIFF** |
| `test_svd_shape=float32[29,29]_fullmatrices=False_computeuv=False` | 24 | 1342 | **LARGE DIFF** |
| `test_svd_shape=float32[29,29]_fullmatrices=False_computeuv=True` | 50 | 1424 | **LARGE DIFF** |
| `test_svd_shape=float32[29,29]_fullmatrices=True_computeuv=False` | 24 | 1342 | **LARGE DIFF** |
| `test_svd_shape=float32[29,29]_fullmatrices=True_computeuv=True` | 49 | 1424 | **LARGE DIFF** |
| `test_svd_shape=float64[2,2]_fullmatrices=False_computeuv=False` | 24 | 875 | **LARGE DIFF** |
| `test_svd_shape=float64[2,2]_fullmatrices=False_computeuv=True` | 50 | 957 | **LARGE DIFF** |
| `test_svd_shape=float64[2,2]_fullmatrices=True_computeuv=False` | 24 | 875 | **LARGE DIFF** |
| `test_svd_shape=float64[2,2]_fullmatrices=True_computeuv=True` | 49 | 957 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,29,7]_fullmatrices=False_computeuv=False` | 26 | 1433 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,29,7]_fullmatrices=False_computeuv=True` | 55 | 1541 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,29,7]_fullmatrices=True_computeuv=False` | 27 | 1433 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,29,7]_fullmatrices=True_computeuv=True` | 54 | 1540 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,53]_fullmatrices=False_computeuv=False` | 26 | 1403 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,53]_fullmatrices=False_computeuv=True` | 55 | 1512 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,53]_fullmatrices=True_computeuv=False` | 27 | 1403 | **LARGE DIFF** |
| `test_svd_shape=float64[2,3,53]_fullmatrices=True_computeuv=True` | 54 | 1511 | **LARGE DIFF** |
| `test_svd_shape=float64[2,7]_fullmatrices=False_computeuv=False` | 25 | 885 | **LARGE DIFF** |
| `test_svd_shape=float64[2,7]_fullmatrices=False_computeuv=True` | 51 | 993 | **LARGE DIFF** |
| `test_svd_shape=float64[2,7]_fullmatrices=True_computeuv=False` | 25 | 885 | **LARGE DIFF** |
| `test_svd_shape=float64[2,7]_fullmatrices=True_computeuv=True` | 50 | 992 | **LARGE DIFF** |
| `test_svd_shape=float64[29,29]_fullmatrices=False_computeuv=False` | 24 | 1342 | **LARGE DIFF** |
| `test_svd_shape=float64[29,29]_fullmatrices=False_computeuv=True` | 50 | 1424 | **LARGE DIFF** |
| `test_svd_shape=float64[29,29]_fullmatrices=True_computeuv=False` | 24 | 1342 | **LARGE DIFF** |
| `test_svd_shape=float64[29,29]_fullmatrices=True_computeuv=True` | 49 | 1424 | **LARGE DIFF** |
| `test_top_k_dtypes_inshape=bfloat16[5,3]_k=2` | 41 | 41 |  |
| `test_top_k_dtypes_inshape=bool[5,3]_k=2` | 18 | 26 | **LARGE DIFF** |
| `test_top_k_dtypes_inshape=float16[5,3]_k=2` | 37 | 10 | **LARGE DIFF** |
| `test_top_k_dtypes_inshape=float32[5,3]_k=2` | 5 | 5 |  |
| `test_top_k_dtypes_inshape=float64[5,3]_k=2` | 37 | 37 |  |
| `test_top_k_dtypes_inshape=int16[5,3]_k=2` | 18 | 24 | **LARGE DIFF** |
| `test_top_k_dtypes_inshape=int32[5,3]_k=2` | 18 | 18 |  |
| `test_top_k_dtypes_inshape=int64[5,3]_k=2` | 18 | 0 |  |
| `test_top_k_dtypes_inshape=int8[5,3]_k=2` | 18 | 24 | **LARGE DIFF** |
| `test_top_k_dtypes_inshape=uint16[5,3]_k=2` | 18 | 24 | **LARGE DIFF** |
| `test_top_k_dtypes_inshape=uint32[5,3]_k=2` | 18 | 18 |  |
| `test_top_k_dtypes_inshape=uint64[5,3]_k=2` | 18 | 0 |  |
| `test_top_k_dtypes_inshape=uint8[5,3]_k=2` | 18 | 24 | **LARGE DIFF** |
| `test_top_k_sort_inf_nan_inshape=float32[5]_k=5` | 32 | 32 |  |
| `test_top_k_stability_inshape=int32[6]_k=3` | 18 | 18 |  |
| `test_transpose_dtypes_shape=bfloat16[2,3]_permutation=(1,0)` | 11 | 14 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=bool[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=complex128[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=complex64[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=float16[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=float32[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=float64[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=int16[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=int32[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=int64[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=int8[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=uint16[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=uint32[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=uint64[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_dtypes_shape=uint8[2,3]_permutation=(1,0)` | 7 | 10 | **LARGE DIFF** |
| `test_transpose_permutations_shape=float32[2,3,4]_permutation=(0,1,2)` | 6 | 5 |  |
| `test_transpose_permutations_shape=float32[2,3,4]_permutation=(1,2,0)` | 7 | 10 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`False_transposea=False_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`False_transposea=False_conjugatea=True_unitdiagonal=False` | 22 | 154 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`False_transposea=True_conjugatea=False_unitdiagonal=False` | 14 | 158 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`False_transposea=True_conjugatea=True_unitdiagonal=False` | 15 | 150 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`True_transposea=False_conjugatea=False_unitdiagonal=False` | 14 | 130 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`True_transposea=False_conjugatea=True_unitdiagonal=False` | 22 | 138 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`True_transposea=True_conjugatea=False_unitdiagonal=False` | 14 | 142 | **LARGE DIFF** |
| `test_triangular_solve_complex_transformations_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=`<br>`True_transposea=True_conjugatea=True_unitdiagonal=False` | 15 | 134 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=bfloat16[4,4]_b=bfloat16[4,1]_leftside=True_lower=False_transposea=Fa`<br>`lse_conjugatea=False_unitdiagonal=False` | 239 | 0 |  |
| `test_triangular_solve_dtypes_a=bfloat16[4,4]_b=bfloat16[4,1]_leftside=True_lower=False_transposea=Fa`<br>`lse_conjugatea=False_unitdiagonal=True` | 251 | 0 |  |
| `test_triangular_solve_dtypes_a=complex128[4,4]_b=complex128[4,1]_leftside=True_lower=False_transpose`<br>`a=False_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=complex128[4,4]_b=complex128[4,1]_leftside=True_lower=False_transpose`<br>`a=False_conjugatea=False_unitdiagonal=True` | 14 | 157 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=False_transposea=`<br>`False_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=complex64[4,4]_b=complex64[4,1]_leftside=True_lower=False_transposea=`<br>`False_conjugatea=False_unitdiagonal=True` | 14 | 157 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=float16[4,4]_b=float16[4,1]_leftside=True_lower=False_transposea=Fals`<br>`e_conjugatea=False_unitdiagonal=False` | 147 | 146 |  |
| `test_triangular_solve_dtypes_a=float16[4,4]_b=float16[4,1]_leftside=True_lower=False_transposea=Fals`<br>`e_conjugatea=False_unitdiagonal=True` | 153 | 157 |  |
| `test_triangular_solve_dtypes_a=float32[4,4]_b=float32[4,1]_leftside=True_lower=False_transposea=Fals`<br>`e_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=float32[4,4]_b=float32[4,1]_leftside=True_lower=False_transposea=Fals`<br>`e_conjugatea=False_unitdiagonal=True` | 14 | 157 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=float64[4,4]_b=float64[4,1]_leftside=True_lower=False_transposea=Fals`<br>`e_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_dtypes_a=float64[4,4]_b=float64[4,1]_leftside=True_lower=False_transposea=Fals`<br>`e_conjugatea=False_unitdiagonal=True` | 14 | 157 | **LARGE DIFF** |
| `test_triangular_solve_real_transformations_a=float32[4,4]_b=float32[4,1]_leftside=True_lower=False_t`<br>`ransposea=False_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_real_transformations_a=float32[4,4]_b=float32[4,1]_leftside=True_lower=False_t`<br>`ransposea=True_conjugatea=False_unitdiagonal=False` | 14 | 146 | **LARGE DIFF** |
| `test_triangular_solve_real_transformations_a=float32[4,4]_b=float32[4,1]_leftside=True_lower=True_tr`<br>`ansposea=False_conjugatea=False_unitdiagonal=False` | 14 | 130 | **LARGE DIFF** |
| `test_triangular_solve_real_transformations_a=float32[4,4]_b=float32[4,1]_leftside=True_lower=True_tr`<br>`ansposea=True_conjugatea=False_unitdiagonal=False` | 14 | 130 | **LARGE DIFF** |
| `test_triangular_solve_shapes_right_a=float32[2,8,8]_b=float32[2,10,8]_leftside=False_lower=False_tra`<br>`nsposea=False_conjugatea=False_unitdiagonal=False` | 141 | 132 |  |
| `test_triangular_solve_shapes_right_a=float32[4,4]_b=float32[1,4]_leftside=False_lower=False_transpos`<br>`ea=False_conjugatea=False_unitdiagonal=False` | 14 | 132 | **LARGE DIFF** |
| `test_type_promotion_add` | 7 | 6 |  |
| `test_type_promotion_equal` | 7 | 6 |  |
| `test_type_promotion_greater_equal` | 7 | 6 |  |
| `test_type_promotion_greater` | 7 | 6 |  |
| `test_type_promotion_less_equal` | 7 | 6 |  |
| `test_type_promotion_less` | 7 | 6 |  |
| `test_type_promotion_maximum` | 7 | 6 |  |
| `test_type_promotion_minimum` | 7 | 6 |  |
| `test_type_promotion_multiply` | 7 | 6 |  |
| `test_type_promotion_not_equal` | 7 | 6 |  |
| `test_type_promotion_subtract` | 7 | 6 |  |
| `test_type_promotion_true_divide` | 7 | 6 |  |
| `test_unary_elementwise_abs_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_abs_float16` | 6 | 5 |  |
| `test_unary_elementwise_abs_float32` | 6 | 5 |  |
| `test_unary_elementwise_abs_float64` | 6 | 5 |  |
| `test_unary_elementwise_acosh_bfloat16` | 72 | 71 |  |
| `test_unary_elementwise_acosh_float16` | 31 | 30 |  |
| `test_unary_elementwise_acosh_float32` | 31 | 30 |  |
| `test_unary_elementwise_acosh_float64` | 31 | 30 |  |
| `test_unary_elementwise_asinh_bfloat16` | 35 | 34 |  |
| `test_unary_elementwise_asinh_float16` | 35 | 34 |  |
| `test_unary_elementwise_asinh_float32` | 33 | 32 |  |
| `test_unary_elementwise_asinh_float64` | 33 | 32 |  |
| `test_unary_elementwise_atanh_bfloat16` | 25 | 24 |  |
| `test_unary_elementwise_atanh_float16` | 23 | 22 |  |
| `test_unary_elementwise_atanh_float32` | 23 | 22 |  |
| `test_unary_elementwise_atanh_float64` | 23 | 22 |  |
| `test_unary_elementwise_bessel_i0e_bfloat16` | 155 | 154 |  |
| `test_unary_elementwise_bessel_i0e_float16` | 155 | 154 |  |
| `test_unary_elementwise_bessel_i0e_float32` | 153 | 152 |  |
| `test_unary_elementwise_bessel_i0e_float64` | 303 | 302 |  |
| `test_unary_elementwise_bessel_i1e_bfloat16` | 153 | 152 |  |
| `test_unary_elementwise_bessel_i1e_float16` | 153 | 152 |  |
| `test_unary_elementwise_bessel_i1e_float32` | 151 | 150 |  |
| `test_unary_elementwise_bessel_i1e_float64` | 301 | 300 |  |
| `test_unary_elementwise_ceil_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_ceil_float16` | 6 | 5 |  |
| `test_unary_elementwise_ceil_float32` | 6 | 5 |  |
| `test_unary_elementwise_ceil_float64` | 6 | 5 |  |
| `test_unary_elementwise_cos_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_cos_float16` | 6 | 5 |  |
| `test_unary_elementwise_cos_float32` | 6 | 5 |  |
| `test_unary_elementwise_cos_float64` | 6 | 5 |  |
| `test_unary_elementwise_cosh_bfloat16` | 19 | 18 |  |
| `test_unary_elementwise_cosh_float16` | 19 | 18 |  |
| `test_unary_elementwise_cosh_float32` | 17 | 16 |  |
| `test_unary_elementwise_cosh_float64` | 17 | 16 |  |
| `test_unary_elementwise_digamma_bfloat16` | 135 | 134 |  |
| `test_unary_elementwise_digamma_float16` | 135 | 134 |  |
| `test_unary_elementwise_digamma_float32` | 133 | 132 |  |
| `test_unary_elementwise_digamma_float64` | 135 | 134 |  |
| `test_unary_elementwise_erf_bfloat16` | 68 | 67 |  |
| `test_unary_elementwise_erf_float16` | 68 | 67 |  |
| `test_unary_elementwise_erf_float32` | 66 | 65 |  |
| `test_unary_elementwise_erf_float64` | 196 | 195 |  |
| `test_unary_elementwise_erf_inv_bfloat16` | 96 | 95 |  |
| `test_unary_elementwise_erf_inv_float16` | 96 | 95 |  |
| `test_unary_elementwise_erf_inv_float32` | 94 | 93 |  |
| `test_unary_elementwise_erf_inv_float64` | 243 | 242 |  |
| `test_unary_elementwise_erfc_bfloat16` | 134 | 133 |  |
| `test_unary_elementwise_erfc_float16` | 134 | 133 |  |
| `test_unary_elementwise_erfc_float32` | 132 | 131 |  |
| `test_unary_elementwise_erfc_float64` | 196 | 195 |  |
| `test_unary_elementwise_exp_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_exp_float16` | 6 | 5 |  |
| `test_unary_elementwise_exp_float32` | 6 | 5 |  |
| `test_unary_elementwise_exp_float64` | 6 | 5 |  |
| `test_unary_elementwise_expm1_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_expm1_float16` | 6 | 5 |  |
| `test_unary_elementwise_expm1_float32` | 6 | 5 |  |
| `test_unary_elementwise_expm1_float64` | 6 | 5 |  |
| `test_unary_elementwise_floor_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_floor_float16` | 6 | 5 |  |
| `test_unary_elementwise_floor_float32` | 6 | 5 |  |
| `test_unary_elementwise_floor_float64` | 6 | 5 |  |
| `test_unary_elementwise_is_finite_bfloat16` | 7 | 6 |  |
| `test_unary_elementwise_is_finite_float16` | 6 | 5 |  |
| `test_unary_elementwise_is_finite_float32` | 6 | 5 |  |
| `test_unary_elementwise_is_finite_float64` | 6 | 5 |  |
| `test_unary_elementwise_lgamma_bfloat16` | 117 | 116 |  |
| `test_unary_elementwise_lgamma_float16` | 117 | 116 |  |
| `test_unary_elementwise_lgamma_float32` | 115 | 114 |  |
| `test_unary_elementwise_lgamma_float64` | 117 | 116 |  |
| `test_unary_elementwise_log1p_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_log1p_float16` | 6 | 5 |  |
| `test_unary_elementwise_log1p_float32` | 6 | 5 |  |
| `test_unary_elementwise_log1p_float64` | 6 | 5 |  |
| `test_unary_elementwise_log_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_log_float16` | 6 | 5 |  |
| `test_unary_elementwise_log_float32` | 6 | 5 |  |
| `test_unary_elementwise_log_float64` | 6 | 5 |  |
| `test_unary_elementwise_neg_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_neg_float16` | 6 | 5 |  |
| `test_unary_elementwise_neg_float32` | 6 | 5 |  |
| `test_unary_elementwise_neg_float64` | 6 | 5 |  |
| `test_unary_elementwise_round_bfloat16` | 8 | 90 | **LARGE DIFF** |
| `test_unary_elementwise_round_float16` | 6 | 35 | **LARGE DIFF** |
| `test_unary_elementwise_round_float32` | 6 | 35 | **LARGE DIFF** |
| `test_unary_elementwise_round_float64` | 6 | 35 | **LARGE DIFF** |
| `test_unary_elementwise_rsqrt_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_rsqrt_float16` | 6 | 5 |  |
| `test_unary_elementwise_rsqrt_float32` | 6 | 5 |  |
| `test_unary_elementwise_rsqrt_float64` | 6 | 5 |  |
| `test_unary_elementwise_sign_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_sign_float16` | 6 | 5 |  |
| `test_unary_elementwise_sign_float32` | 6 | 5 |  |
| `test_unary_elementwise_sign_float64` | 6 | 5 |  |
| `test_unary_elementwise_sin_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_sin_float16` | 6 | 5 |  |
| `test_unary_elementwise_sin_float32` | 6 | 5 |  |
| `test_unary_elementwise_sin_float64` | 6 | 5 |  |
| `test_unary_elementwise_sinh_bfloat16` | 31 | 30 |  |
| `test_unary_elementwise_sinh_float16` | 31 | 30 |  |
| `test_unary_elementwise_sinh_float32` | 29 | 28 |  |
| `test_unary_elementwise_sinh_float64` | 29 | 28 |  |
| `test_unary_elementwise_sqrt_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_sqrt_float16` | 6 | 5 |  |
| `test_unary_elementwise_sqrt_float32` | 6 | 5 |  |
| `test_unary_elementwise_sqrt_float64` | 6 | 5 |  |
| `test_unary_elementwise_tan_bfloat16` | 14 | 18 | **LARGE DIFF** |
| `test_unary_elementwise_tan_float16` | 14 | 13 |  |
| `test_unary_elementwise_tan_float32` | 12 | 11 |  |
| `test_unary_elementwise_tan_float64` | 12 | 11 |  |
| `test_unary_elementwise_tanh_bfloat16` | 8 | 7 |  |
| `test_unary_elementwise_tanh_float16` | 6 | 5 |  |
| `test_unary_elementwise_tanh_float32` | 6 | 5 |  |
| `test_unary_elementwise_tanh_float64` | 6 | 5 |  |
| `test_zeros_like_shape=bfloat16[3,4,5]` | 9 | 8 |  |
| `test_zeros_like_shape=bool[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=complex128[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=complex64[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=float16[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=float32[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=float64[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=int16[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=int32[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=int64[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=int8[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=uint16[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=uint32[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=uint64[3,4,5]` | 7 | 6 |  |
| `test_zeros_like_shape=uint8[3,4,5]` | 7 | 6 |  |
