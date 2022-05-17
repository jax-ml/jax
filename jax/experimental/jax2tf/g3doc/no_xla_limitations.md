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
whether the ops is fully, partially, or not supported by the `jax2tf` emitter.
In the next section we provide more details on the ops for which we provide
partial support.

For a detailed description of these XLA ops, please see the
[XLA Operation Semantics documentation](https://www.tensorflow.org/xla/operation_semantics).

| XLA ops ([documentation](https://www.tensorflow.org/xla/operation_semantics)) | JAX primitive(s) ([documentation](https://jax.readthedocs.io/en/latest/jax.lax.html)) | Supported |
| ------- | ---------------- | ------- |
| XlaDot  | `lax.dot_general` | Full |
| XlaDynamicSlice | `lax.dynamic_slice` | Full |
| XlaDynamicUpdateSlice | `lax.dynamic_update_slice` | Full |
| XlaPad  | `lax.pad` | Full |
| XlaConv | `lax.conv_general_dilated` | [Partial](#xlaconv) |
| XlaGather | `lax.gather` | [Partial](#xlagather) |
| XlaReduceWindow | `lax.reduce_window_sum_p`, `lax.reduce_window_min_p`, `lax.reduce_window_max_p`, and `lax.reduce_window_p` | [Partial](#xlareducewindow) |
| XlaScatter | `lax.scatter_p`, `lax.scatter_min_p`, `lax.scatter_max_p`, `lax.scatter_mul_p`, `lax.scatter_add_p` | [Partial](#xlascatter) |
| XlaSelectAndScatter | `lax.select_and_scatter_add_p` | Unsupported |
| XlaReduce | `lax.reduce`, `lax.argmin`, `lax.argmax` | Unsupported |
| XlaVariadicSort | `lax.sort` | Unsupported |


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

We provide support for convolutions as follows:

* Only 2D convolutions, i.e. `lhs.ndim == 4`.
* Regular convolutions and atrous convolutions
  (i.e., `rhs_dilation != (1, 1, ...)`) are supported through the TF op
  [`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d).
* Transposed convolutions (i.e., `lhs_dilation != (1, 1, ...)`) are supported
  through
  [`tf.nn.conv2d_transpose`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose).
  If using transposed convolutions, then `padding == 'VALID'`.
* Depthwise convolutions (i.e.
  `in_channels == feature_group_count and feature_group_count > 1`) are
  supported through
  [`tf.nn.depthwise_conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d).
* No support for batch groups, i.e. `batch_group_count == 1`.
* No support for feature groups, except for depth-wise convolutions.
* Input may be provided in any order (specified using `dimension_numbers`).
* Only one of depthwise, atrous and tranposed convolutions may be used at the
  same time.

### XlaGather

XLA's gather op is complex and covers may use cases. It is called from JAX using
`lax.gather`, but many other primitives and operations use it as well, for
instance, parallelization primitives `vmap` and `pmap` use gather to specify a
batch dimension, and it is used for slices or multidimensional indexing as well,
e.g. `x[0, 1]`, `x[:, :1]`, or `x[[0], [1]]`.

The signature of [`lax.gather`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.gather.html#jax.lax.gather)
is as follows:

```
jax.lax.gather(
    operand, start_indices, dimension_numbers, slice_sizes,
    unique_indices=False, indices_are_sorted=False, mode=None,
    fill_value=None
)
```

We provide support for the following cases:

* *Scalar indexing*. This means we are indexing into a single dimension,
  retrieving either a partial slice or a single value from that dimension. For
  all other dimensions we retrieve the full slice. Examples include `op[2]`,
  `op[:, :5, :]`, and `jnp.take(op, 0, axis=0)`. This means that
  `len(start_indices.shape) == 1`. We provide support for this path through the
  TF op
  [`tf.strided_slice`](https://www.tensorflow.org/api_docs/python/tf/strided_slice).

* *Multi-dimensional indexing*. This means we index into multiple dimensions,
  e.g., `jnp.take(op, [[0], [1]], axis=0)` or `op[[0], [4], [1]]`. We currently
  only support multi-dimensional indexing if the last dimension is 1, which
  means we can only retrieve a single value per dimension, and we can't retrieve
  slices. We provide support for this path through the TF op
  [`tf.gather`](https://www.tensorflow.org/api_docs/python/tf/gather).

* *Gather with a batch dimension*. E.g., when doing
  `jax.vmap(lax.dynamic_slice)`, which will result in a call to `lax.gather`
  where the first dimension of the input is the batch dimension. This means that
  `len(batch_dims) == 1`. We currently only support a single batch dimension
  (i.e., `vmap(vmap))` does not work). We provide support for this path through
  the TF op [`tf.slice`](https://www.tensorflow.org/api_docs/python/tf/slice).

All other cases of `lax.gather` are currently not supported.


### XlaReduceWindow

This op is called by `lax.reduce_window_sum_p`, `lax.reduce_window_max_p`,
`lax.reduce_window_min_p` and `lax.reduce_window`.

Of these ops, we currently only support `lax.reduce_window_sum_p` and
`lax.reduce_window_max_p` through respectively the TF ops
[`tf.nn.avg_pool`](https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool) and
[`tf.nn.max_pool`](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool).

Both functions have the following signature:

```
lax.reduce_window_{sum,max}(
    operand, window_dimensions, window_strides,
    padding, base_dilation, window_dilation)
```

We support these ops with the following limitations:

* For `reduce_window_sum_p`, dtypes `jnp.bool`, `jnp.uint32`, `jnp.uint64`,
  `jnp.complex64`, and `jnp.complex128` are not supported. 
* For `reduce_window_max_p`, dtype `jnp.float16`, `jnp.float32`, and
  `jnp.float64` are not supported.
* We support at most 3 spatial dimension.
* `base_dilation = (1, 1, ...)`.
* `window_dilation == (1, 1, ...)`.
* `padding` should either be `VALID` or `SAME`.

`lax.reduce_window_min_p` and `lax.reduce_window` are currently not supported.

### XlaScatter

This op is called by `lax.scatter`, `lax.scatter_min`, `lax.scatter_max`, 
`lax.scatter_mul` and `lax.scatter_add`. 

We support all these ops for unique indices. For non-unique indices we 
support (min,max,mul,add) for single depth scatters.

There are a few more limitations:
* the GatherScatterMode must be PROMISE_IN_BOUNDS.
* dtypes `np.bool` and `jnp.complex*` are not supported.