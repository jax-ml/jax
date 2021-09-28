# Converting JAX examples to TFLite/TFjs

*Last generated on: 2021-10-04* (YYYY-MM-DD)

## jax2tf --> TFLite

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | FAIL | NotImplementedError('Call to conv_general_dilated cannot be converted with enable_xla=False. Unimplemented support for window_strides != (1, 1) - See source code for the precise conditions under which convolutions can be converted without XLA.')
| lm1b | FAIL | TypeError("Value passed to parameter 'begin' has DataType uint32 not in list of allowed values: int32, int64")
| mnist | SUCCESS | 
| nlp_seq | FAIL | ConverterError('/Users/marcvanzee/.pyenv/versions/3.7.10/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:750:0: error: \'tf.Expm1\' op is neither a custom op nor a flex op\n/Users/marcvanzee/.pyenv/versions/3.7.10/lib/python3.7/site-packages/tensorflow/python/ops/gen_math_ops.py:3798:0: note: called from\n/Users/marcvanzee/github/jax/jax/experimental/jax2tf/jax2tf.py:819:0: note: called from\n/Users/marcvanzee/github/jax/jax/experimental/jax2tf/jax2tf.py:836:0: note: called from\n/Users/marcvanzee/github/jax/jax/core.py:277:0: note: called from\n/Users/marcvanzee/github/jax/jax/_src/lax/lax.py:192:0: note: called from\n/Users/marcvanzee/github/jax/jax/_src/numpy/lax_numpy.py:661:0: note: called from\n/Users/marcvanzee/github/jax/jax/linear_util.py:166:0: note: called from\n/Users/marcvanzee/github/jax/jax/experimental/jax2tf/jax2tf.py:879:0: note: called from\n/Users/marcvanzee/github/jax/jax/core.py:1645:0: note: called from\n/Users/marcvanzee/.pyenv/versions/3.7.10/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:750:0: note: Error code: ERROR_NEEDS_CUSTOM_OPS\n<unknown>:0: error: failed while converting: \'main\': \nSome ops in the model are custom ops, See instructions to implement custom ops: https://www.tensorflow.org/lite/guide/ops_custom \nCustom ops: Expm1\nDetails:\n\ttf.Expm1(tensor<2x1x2xf32>) -> (tensor<2x1x2xf32>) : {device = ""}\n\n')
| pixelcnn++ | FAIL | NotImplementedError('Call to conv_general_dilated cannot be converted with enable_xla=False. Input padding not supported in TensorFlow. - See source code for the precise conditions under which convolutions can be converted without XLA.')
| ppo | FAIL | NotImplementedError('Call to conv_general_dilated cannot be converted with enable_xla=False. Unimplemented support for window_strides != (1, 1) - See source code for the precise conditions under which convolutions can be converted without XLA.')
| seq2seq | SUCCESS | 
| sst2 | FAIL | NotImplementedError("Call to gather cannot be converted with enable_xla=False. unsupported dimension_numbers 'GatherDimensionNumbers(offset_dims=(1, 2), collapsed_slice_dims=(0,), start_index_map=(0, 1, 2))'; op_shape=(2, 6, 3).")
| vae | FAIL | ModuleNotFoundError("No module named 'utils'")
| wmt | FAIL | TypeError("Value passed to parameter 'begin' has DataType uint32 not in list of allowed values: int32, int64")

## jax2tf --> TFjs

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | FAIL | NotImplementedError('Call to conv_general_dilated cannot be converted with enable_xla=False. Unimplemented support for window_strides != (1, 1) - See source code for the precise conditions under which convolutions can be converted without XLA.')
| lm1b | FAIL | TypeError("Value passed to parameter 'begin' has DataType uint32 not in list of allowed values: int32, int64")
| mnist | SUCCESS | 
| nlp_seq | FAIL | ValueError("Error when tracing gradients for SavedModel.\n\nSee the stack trace above to see the error that was raised when converting a gradient function to a concrete function. You may need to update the custom gradient, or disable saving gradients with the option tf.saved_model.SaveOptions(custom_gradients=False).\n\tProblematic op name: IdentityN\n\tGradient inputs: (<tf.Tensor 'AddV2_12:0' shape=(2, 1, 8) dtype=float32>, <tf.Tensor 'jax2tf_arg_0:0' shape=(8,) dtype=float32>, <tf.Tensor 'jax2tf_arg_1:0' shape=(4, 8) dtype=float32>, <tf.Tensor 'jax2tf_arg_2:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_3:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_4:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_5:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_6:0' shape=(2,) dtype=float32>, <tf.Tensor 'jax2tf_arg_7:0' shape=(4, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_8:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_9:0' shape=(2, 4) dtype=float32>, <tf.Tensor 'jax2tf_arg_10:0' shape=(4, 1, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_11:0' shape=(1, 2, 4) dtype=float32>, <tf.Tensor 'jax2tf_arg_12:0' shape=(4, 1, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_13:0' shape=(4, 1, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_14:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_15:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_16:0' shape=(8, 4) dtype=float32>, <tf.Tensor 'jax2tf_arg_17:0' shape=(2, 1) dtype=float32>)")
| pixelcnn++ | FAIL | NotImplementedError('Call to conv_general_dilated cannot be converted with enable_xla=False. Input padding not supported in TensorFlow. - See source code for the precise conditions under which convolutions can be converted without XLA.')
| ppo | FAIL | NotImplementedError('Call to conv_general_dilated cannot be converted with enable_xla=False. Unimplemented support for window_strides != (1, 1) - See source code for the precise conditions under which convolutions can be converted without XLA.')
| seq2seq | FAIL | ValueError('Unsupported Ops in the model before optimization\nBitcast, BitwiseAnd, BitwiseOr, RightShift, LeftShift, BitwiseXor')
| sst2 | FAIL | NotImplementedError("Call to gather cannot be converted with enable_xla=False. unsupported dimension_numbers 'GatherDimensionNumbers(offset_dims=(1, 2), collapsed_slice_dims=(0,), start_index_map=(0, 1, 2))'; op_shape=(2, 6, 3).")
| vae | FAIL | ModuleNotFoundError("No module named 'utils'")
| wmt | FAIL | TypeError("Value passed to parameter 'begin' has DataType uint32 not in list of allowed values: int32, int64")

## Table generation

See `examples_test.py` for instructions on how to regenerate this table.
