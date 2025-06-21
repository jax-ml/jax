# jax2tf Limitations for `enable_xla=False`

*Note: the list below is only for running jax2tf with `enable_xla=False`. For general jax2tf known issues please see [here](https://github.com/jax-ml/jax/tree/main/jax/experimental/jax2tf#known-issues)*

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

| XLA ops ([documentation](https://www.tensorflow.org/xla/operation_semantics)) | JAX primitive(s) ([documentation](https://docs.jax.dev/en/latest/jax.lax.html)) | Supported |
| ------- | ---------------- | ------- |
| XlaDot  | `lax.dot_general` | Full |
| XlaDynamicSlice | `lax.dynamic_slice` | Full |
| XlaDynamicUpdateSlice | `lax.dynamic_update_slice` | Full |
| XlaPad  | `lax.pad` | Full |
| XlaConv | `lax.conv_general_dilated` | [Partial](#xlaconv) |
| XlaGather | `lax.gather` | [Partial](#xlagather) |
| XlaReduceWindow | `lax.reduce_window` | [Partial](#xlareducewindow) |
| XlaScatter | `lax.scatter`, `lax.scatter_min`, `lax.scatter_max`, `lax.scatter_mul`, `lax.scatter_add` | [Partial](#xlascatter) |
| XlaSelectAndScatter | `lax._select_and_scatter_add` | Unsupported |
| XlaReduce | `lax.reduce`, `lax.argmin`, `lax.argmax` | Unsupported |
| XlaVariadicSort | `lax.sort` | Unsupported |


## Partially Supported JAX Primitives

Below we describe for all partially supported JAX primitives which cases we
support and which not.

### XlaConv

JAX convolutions are done using
[`lax.conv_general_dilated`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html).

```
lax.conv_general_dilated(
    lhs, rhs, window_strides, padding, lhs_dilation,
    rhs_dilation, dimension_numbers, feature_group_count,
    batch_group_count, precision, preferred_element_type
)
```

We provide support for convolutions as follows:

* Only 1D and 2D convolutions, i.e. `lhs.ndim == 3 or 4`.
* Regular convolutions and atrous (aka, dilated) convolutions
  (i.e., `rhs_dilation != (1, 1, ...)`) are supported through the TF op
  [`tf.nn.conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d).
* Transposed convolutions (i.e., `lhs_dilation != (1, 1, ...)`) are supported
  through
  [`tf.nn.conv2d_transpose`](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose),
  with either 'SAME' or 'VALID' padding.
* Depthwise convolutions (i.e.
  `in_channels == feature_group_count and feature_group_count > 1`) are
  supported through
  [`tf.nn.depthwise_conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d).
  Note that atrous depthwise convolutions are supported.
* No support for batch groups, i.e. `batch_group_count == 1`.
* No support for feature groups, except for depth-wise convolutions.
* Input may be provided in any order (specified using `dimension_numbers`).
* Only one of depthwise, atrous and transposed convolutions may be used at the
  same time, though depthwise atrous convolutions are supported.
* Convolutions are known to have a somewhat higher numeric inaccuracy, so if you 
  are using many large convolutions, this may lead to large deviations.

### XlaGather

XLA's gather op is complex and covers may use cases. It is called from JAX using
`lax.gather`, but many other primitives and operations use it as well, for
instance, parallelization primitives `vmap` and `pmap` use gather to specify a
batch dimension, and it is used for slices or multidimensional indexing as well,
e.g. `x[0, 1]`, `x[:, :1]`, or `x[[0], [1]]`.

The signature of [`lax.gather`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html#jax.lax.gather)
is as follows:

```
lax.gather(
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

The signature of [`lax.reduce_window`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html)
is as follows:

```
lax.reduce_window(operand, init_value, computation: Callable,
                  window_dimensions: core.Shape, window_strides: Sequence[int],
                  padding: Union[str, Sequence[Tuple[int, int]]],
                  base_dilation: Optional[Sequence[int]] = None,
                  window_dilation: Optional[Sequence[int]] = None)
)
```

This function with either call a monoid reducer `lax.reduce_window_min_p`,
`lax.reduce_window_max_p`, or `lax.reduce_window_sum_p`, or the full reducer
function `lax.reduce_window_p` with the following conditions:

* If `computation` is one of `lax.min`, `lax.max`, or `lax.add` and `init_value`
  is the identity element for `computation` (for instance: 0 for `lax.add`),
  then it will call one of the monoid reducers.

* Otherwise, it will call the full reduction function `lax.reduce_window_p`.

We provide partial support for all these ops, with the following limitations:

*   `computation` should be one of `lax.min`, `lax.max`, or `lax.add`.
*   For `lax.min` and `lax.max`, dtypes `np.bool`, `np.uint32`, `np.uint64`,
    `np.complex64`, and `np.complex128` are not supported.
*   Additionally, for `lax.min`, dtypes `np.uint8` and `np.uint16` are not
    supported.
*   For `lax.add`, only dtypes `np.float16`, `np.float32`, and `np.float64` are
    supported.
*   We support at most 2 spatial dimension.
*   Base dilations other than `(1,) * len(operand)` are not supported.
*   `padding` should either be `VALID` or `SAME`.
*   We compute `lax.reduce_window_sum_p` by calling `tf.nn.avg_pool` (through
    `tf.nn.pool`), and then multiplying the result by
    `np.prod(window_dimensions)`. If you are using an NN library that implements
    `avg_pool` using `lax.reduce_window` (such as Flax's
    [pooling.py](https://github.com/google/flax/blob/main/flax/linen/pooling.py)),
    this is usually implemented by dividing the result with
    `np.prod(window_dimensions)`. So when converting this function, the
    resulting computation for `avg_pool` is `(tf.nn.avg_pool(xs) *
    np.prod(window)) / np.prod(window)`. This is redundant and can be optimized.
*   Using `lax.add` on TPU may give very large deviations. This is due to the
    way the conversion is implemented (first take the average over the window
    and then multiply by window size). This gives large deviations on TPU due to
    the fact that it uses `bfloat16` for computations.

We implement all reductions using the Tensorflow function
[tf.nn.pool](https://www.tensorflow.org/api_docs/python/tf/nn/pool).

### XlaScatter

This op is called by `lax.scatter`, `lax.scatter_min`, `lax.scatter_max`,
`lax.scatter_mul` and `lax.scatter_add`.

We support all these ops for unique indices. For non-unique indices we
support (min,max,mul,add) for single depth scatters.

We implement support for this op through
[tf.tensor_scatter_nd_update](https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update).

There are a few more limitations:

* Dtypes `np.bool` and `jnp.complex*` are not supported.
* We disallow scatter mode `lax.GatherScatterMode.CLIP` because it may lead to
  incorrect behavior for out-of-bounds indices (see next point).
* The behavior for out-of-bounds scatter indices is as follows:
  - When running in eager or graph mode, it throws an error. This is because
    `tf.scatter` throws an error as well. If this is problematic for your use
    case, please let us know and we can add more support for this.
  - When running in compile mode, the out-of-bounds indices are dropped, which
    is the behavior of both `lax.GatherScatterMode.FILL_OR_DROP` and
    `lax.GatherScatterMode.PROMISE_IN_BOUNDS`. This is why `CLIP` is not
    allowed.
