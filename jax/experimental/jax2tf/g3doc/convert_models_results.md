# JAX Converters Evaluation Results

*Last generated on: 2023-01-03* (YYYY-MM-DD)

This file contains the evaluation results for all converters in table format.

See [models_test_main.py](../tests/models_test_main.py) for instructions on how to
regenerate this table.

See [Description of Converters](#description-of-converters) below for more
details on the different converters.

## Summary Table

| Example | jax2tf_xla | jax2tf_noxla | jax2tfjs | jax2tflite | jax2tflite+flex |
| --- | --- | --- | --- | --- | --- |
| flax/actor_critic | YES | YES | YES | YES | YES |
| flax/bilstm | YES | YES | YES | [NO](#example-flaxbilstm--converter-jax2tflite) |  [NO](#example-flaxbilstm--converter-jax2tfliteflex) | 
| flax/cnn | YES | YES | YES | YES | YES |
| flax/gnn | YES |  [NO](#example-flaxgnn--converter-jax2tf_noxla) |  [NO](#example-flaxgnn--converter-jax2tfjs) |  [NO](#example-flaxgnn--converter-jax2tflite) |  [NO](#example-flaxgnn--converter-jax2tfliteflex) | 
| flax/gnn_conv | YES |  [NO](#example-flaxgnn_conv--converter-jax2tf_noxla) |  [NO](#example-flaxgnn_conv--converter-jax2tfjs) |  [NO](#example-flaxgnn_conv--converter-jax2tflite) |  [NO](#example-flaxgnn_conv--converter-jax2tfliteflex) | 
| flax/resnet50 | YES | YES | YES | YES | YES |
| flax/seq2seq_lstm | YES | YES | [NO](#example-flaxseq2seq_lstm--converter-jax2tfjs) |  [NO](#example-flaxseq2seq_lstm--converter-jax2tflite) |  YES |
| flax/lm1b | YES | YES | YES | YES | YES |
| flax/nlp_seq | YES | YES | YES | YES | YES |
| flax/wmt | YES | YES | YES | YES | YES |
| flax/vae | YES | YES | YES | YES | YES |

## Errors

## `flax/bilstm`
### Example: `flax/bilstm` | Converter: `jax2tflite`
```
RuntimeError('third_party/tensorflow/lite/kernels/concatenation.cc:159 t->dims->data[d] != t0->dims->data[d] (3 != 1)Node number 11 (CONCATENATION) failed to prepare.Node number 32 (WHILE) failed to invoke.')
```
[Back to top](#summary-table)

### Example: `flax/bilstm` | Converter: `jax2tflite+flex`
```
RuntimeError('third_party/tensorflow/lite/kernels/concatenation.cc:159 t->dims->data[d] != t0->dims->data[d] (3 != 1)Node number 11 (CONCATENATION) failed to prepare.Node number 32 (WHILE) failed to invoke.')
```
[Back to top](#summary-table)

## `flax/gnn`
### Example: `flax/gnn` | Converter: `jax2tf_noxla`
```
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

### Example: `flax/gnn` | Converter: `jax2tfjs`
```
Conversion error
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

### Example: `flax/gnn` | Converter: `jax2tflite`
```
Conversion error
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

### Example: `flax/gnn` | Converter: `jax2tflite+flex`
```
Conversion error
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

## `flax/gnn_conv`
### Example: `flax/gnn_conv` | Converter: `jax2tf_noxla`
```
{{function_node __wrapped__UnsortedSegmentSum_device_/job:localhost/replica:0/task:0/device:CPU:0}} segment_ids[0] = 78 is out of range [0, 52) [Op:UnsortedSegmentSum]
```
[Back to top](#summary-table)

### Example: `flax/gnn_conv` | Converter: `jax2tfjs`
```
Conversion error
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

### Example: `flax/gnn_conv` | Converter: `jax2tflite`
```
Conversion error
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

### Example: `flax/gnn_conv` | Converter: `jax2tflite+flex`
```
Conversion error
NotImplementedError("Call to reduce_window cannot be converted with enable_xla=False. Add pooling does not support operands of type <dtype: 'int32'> - See source code for the precise conditions under which it can be converted without XLA.")
```
[Back to top](#summary-table)

## `flax/seq2seq_lstm`
### Example: `flax/seq2seq_lstm` | Converter: `jax2tfjs`
```
Conversion error
ValueError('Unsupported Ops in the model before optimization
Bitcast, BitwiseOr, BitwiseAnd, LeftShift, RightShift, BitwiseXor')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm` | Converter: `jax2tflite`
```
Conversion error
Some ops are not supported by the native TFLite runtime
	tf.Bitcast(tensor<1x4xui32>) -> (tensor<1x4xf32>) : {device = ""}
	tf.BitwiseOr(tensor<1x4xui32>, tensor<ui32>) -> (tensor<1x4xui32>) : {device = ""}
	tf.BitwiseOr(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.BitwiseOr(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>) : {device = ""}
	tf.BitwiseXor(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.BitwiseXor(tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>) : {device = ""}
	tf.BitwiseXor(tensor<ui32>, tensor<ui32>) -> (tensor<ui32>) : {device = ""}
	tf.ConcatV2(tensor<1xui32>, tensor<1xui32>, tensor<i32>) -> (tensor<2xui32>)
	tf.ConcatV2(tensor<2xui32>, tensor<2xui32>, tensor<i32>) -> (tensor<4xui32>) : {device = ""}
	tf.LeftShift(tensor<1xui32>, tensor<ui32>) -> (tensor<1xui32>) : {device = ""}
	tf.LeftShift(tensor<2xui32>, tensor<ui32>) -> (tensor<2xui32>) : {device = ""}
	tf.Pack(tensor<ui32>, tensor<ui32>) -> (tensor<2xui32>) : {axis = 0 : i64}
	tf.RightShift(tensor<1x4xui32>, tensor<ui32>) -> (tensor<1x4xui32>) : {device = ""}
	tf.RightShift(tensor<1xui32>, tensor<ui32>) -> (tensor<1xui32>) : {device = ""}
	tf.RightShift(tensor<2xui32>, tensor<ui32>) -> (tensor<2xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<1x4xui32>, tensor<1x4xui32>) -> (tensor<1x4xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<2xui32>, tensor<2xui32>) -> (tensor<2xui32>) : {device = ""}
	tf.Slice(tensor<1x2xui32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<1x2xui32>) : {device = ""}
	tf.StridedSlice(tensor<2xui32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
```
[Back to top](#summary-table)

## `flax/lm1b`
## `flax/nlp_seq`
## `flax/wmt`
## `flax/vae`

## Description of Converters

Below is a description of all converters that can be found in
[converters.py](converters.py).

### `jax2tf_xla`

This converter simply converts a the forward function of a JAX model to a
Tensorflow function with XLA support linked in. This is considered the baseline
converter and has the largest coverage, because we expect nearly all ops to be
convertible. However, please see
[jax2tf Known Issue](https://github.com/google/jax/tree/main/jax/experimental/jax2tf#known-issues)
for a list of known problems.

### `jax2tf_noxla`

This converter converts a JAX model to a Tensorflow function without XLA
support. This means the Tensorflow XLA ops aren't used. See
[here](https://github.com/google/jax/tree/main/jax/experimental/jax2tf#tensorflow-xla-ops)
for more details.

### `jax2tfjs`

This converter first converts a JAX model to TF SavedModel format without XLA
support. After that, it converts the SavedModel to TensorFlow.js using the
[TF.js converter](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter#calling-a-converter-function-in-python-flaxjax).

### `jax2tflite`

This converter first converts a JAX model to TF SavedModel format without XLA
support. After that, it converts the SavedModel to TFLite using the
[TFLite converter](https://www.tensorflow.org/lite/convert).

### `jax2tflite+flex`

This is similar to the `jax2tflite` path, but then links in the Select ops. See
[here](https://www.tensorflow.org/lite/guide/ops_select) for more details.
