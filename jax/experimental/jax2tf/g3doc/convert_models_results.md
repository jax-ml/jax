# JAX Converters Evaluation Results

*Last generated on: 2022-09-07* (YYYY-MM-DD)

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
| flax/resnet50 | YES | YES | YES | YES | YES |
| flax/seq2seq | YES | YES | [NO](#example-flaxseq2seq--converter-jax2tfjs) |  [NO](#example-flaxseq2seq--converter-jax2tflite) |  YES |
| flax/lm1b | YES | YES | YES | YES | YES |
| flax/nlp_seq | YES | YES | YES | YES | YES |
| flax/wmt | YES | YES | YES | YES | YES |
| flax/vae | YES | YES | YES | YES | YES |

## Errors

## `flax/actor_critic`
## `flax/bilstm`
### Example: `flax/bilstm` | Converter: `jax2tflite`
```
Conversion error
Some ops are not supported by the native TFLite runtime
	tf.Abs(tensor<2xi32>) -> (tensor<2xi32>) : {device = ""}
	tf.Sign(tensor<2xi32>) -> (tensor<2xi32>) : {device = ""}
```
[Back to top](#summary-table)

### Example: `flax/bilstm` | Converter: `jax2tflite+flex`
```
RuntimeError('tensorflow/lite/kernels/concatenation.cc:158 t->dims->data[d] != t0->dims->data[d] (3 != 1)Node number 11 (CONCATENATION) failed to prepare.Node number 32 (WHILE) failed to invoke.')
```
[Back to top](#summary-table)

## `flax/resnet50`
## `flax/seq2seq`
### Example: `flax/seq2seq` | Converter: `jax2tfjs`
```
Conversion error
ValueError('Unsupported Ops in the model before optimization
BitwiseXor, RightShift, Bitcast, BitwiseOr, BitwiseAnd, LeftShift')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq` | Converter: `jax2tflite`
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
