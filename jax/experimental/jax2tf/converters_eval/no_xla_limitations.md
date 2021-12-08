# jax2tf Limitations for `enable_xla=False`

*Note: the list below is only for running jax2tf with `enable_xla=False`. For general jax2tf known issues please see [here](https://github.com/google/jax/tree/main/jax/experimental/jax2tf#known-issues)*

For most JAX primitives there is a natural TF op that fits the needed semantics
(e.g., `jax.lax.abs` is equivalent to `tf.abs`). However, there are a number of
JAX primitives for which there is no single TF op with matching semantics
(e.g., `jax.lax.conv_general_dilated` does not have a matching `tf` op). For
these cases, the `jax2tf` emitter uses a set of special TF ops that are thin
wrappers over HLO ops.

However, these ops are only be executable by a consumer that has XLA linked in,
and this is not the case for the TF.js and TFLite converter. Therefore we
provide limited support for these ops by implementing them in terms of ops that
are supported in [impl_no_xla.py](../impl_no_xla.py).

## Summary Table

The table below shows for each XLA ops by which JAX primitives it is used, and
whether the ops is fully, partially, or not supported by the `jax2tf` emitted.
In the next section we provide more details on the ops for which we provide
partial support.

| XLA ops | JAX primitive(s) | Supported |
| ------- | ---------------- | ------- |
| [XlaDot](https://www.tensorflow.org/xla/operation_semantics#dot)  | `lax.dot_general` | Full |
| [XlaDynamicSlice](https://www.tensorflow.org/xla/operation_semantics#dynamicslice) | `lax.dynamic_slice` | Full |
| [XlaDynamicUpdateSlice](https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice) | `lax.dynamic_update_slice` | Full |
| [XlaPad](https://www.tensorflow.org/xla/operation_semantics#pad)  | `lax.pad` | Full |
| [XlaConv](https://www.tensorflow.org/xla/operation_semantics#conv_convolution) | `lax.conv_general_dilated` | Partial |
| [XlaGather](https://www.tensorflow.org/xla/operation_semantics#gather) | `lax.gather` | Partial |
| [XlaReduceWindow](https://www.tensorflow.org/xla/operation_semantics#reducewindow) | `lax.reduce_window_sum_p`, `lax.reduce_window_min_p`, `lax.reduce_window_max_p`, and `lax.reduce_window_p` | Partial |
| [XlaScatter](https://www.tensorflow.org/xla/operation_semantics#scatter) | `lax.scatter_p`, `lax.scatter_min_p`, `lax.scatter_max_p`, `lax.scatter_mul_p`, `lax.scatter_add_p` | Unsupported |
| [XlaSelectAndScatter](https://www.tensorflow.org/xla/operation_semantics#selectandscatter) | `lax.select_and_scatter_add_p` | Unsupported |
| [XlaReduce](https://www.tensorflow.org/xla/operation_semantics#reduce) | `lax.reduce`, `lax.argmin`, `lax.argmax` | Unsupported |
| [XlaSort](https://www.tensorflow.org/xla/operation_semantics#sort) | `lax.sort` | Unsupported |


## Partially Supported JAX Primitives

Below we describe for all partially supported JAX primitives which cases we
support and which not.

### XlaConv

JAX convolutions are done using
[`lax.conv_general_dilated`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html).

```
lax.conv_general_dilated(
    lhs, rhs, window_strides, padding, lhs_dilation,
    rhs_dilation, dimension_numbers, feature_group_count,
    batch_group_count, precision, preferred_element_type
)
```

We provide support for the following cases:

* Any stride size (`window_strides`).
* Any padding type (`padding`).
* Atrous convolutions (`rhs_dilation != (1, 1, ...)`).
* Transposed convolutions (`lhs_dilation != (1, 1, ...)`).
* Depthwise convolutions (`in_channels == feature_group_count and feature_group_count > 1`).
* Provide input in any order (specified using `dimension_numbers`).

The implementation currently has the following limitations:

* Only 2D convolutions, i.e. `lhs.ndim == 4`.
* No batch groups, i.e. `batch_group_count == 1`.
* No feature groups, except for depth-wise convolutions (See above).
* If using transposed convolutions, then `padding == 'VALID'`.
* Only one of depthwise, atrous and tranposed convolutions.

### XlaGather

`TODO(marcvanzee): Write this`

### XlaReduceWindow

`TODO(marcvanzee): Write this`







