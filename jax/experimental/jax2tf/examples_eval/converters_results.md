# Evaluation Results

*Last generated on: 2021-11-17* (YYYY-MM-DD)

## jax2tf --> TFLite

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | SUCCESS |
| lm1b | SUCCESS |
| mnist | SUCCESS |
| nlp_seq | SUCCESS |
| pixelcnn++ | FAIL | ValueError('Numerical difference JAX vs TF: JAX result=[[[-1.42588705e-01  4.43906128e-01 -4.43267524e-01 ...  2.41633713e-01\n    4.30841953e-01 -3.73984545e-01]\n  [-5.50750673e-01  8.27922702e-01 -5.37640810e-01 ...  7.13608935e-02\n    5.00771403... (CROPPED)
| ppo | SUCCESS |
| seq2seq | SUCCESS |
| sst2 | SUCCESS |
| vae | FAIL | ValueError('Numerical difference JAX vs TF: JAX result=[[[1.00000000e+00 2.96458746e-09 2.83802848e-08 ... 0.00000000e+00\n   9.99999881e-01 6.08000756e-19]\n  [9.81082678e-01 6.93151236e-01 7.65325581e-08 ... 1.04346945e-07\n   7.08327889e-01 3.9745... (CROPPED)
| wmt | SUCCESS |

## jax2tf --> TFjs

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | SUCCESS |
| lm1b | FAIL | ValueError("in user code:\n\n\n    ValueError: Got a non-Tensor value FrozenDict({\n        cache: {\n            decoder: {\n                encoderdecoderblock_0: {\n                    SelfAttention_0: {\n                        cache_index: <tf.Tensor 'StatefulPartitionedCall:1' shape=() dtype=int32>,\n                        cached_key: <tf.Tensor 'StatefulPartitionedCall:2' shape=(2, 1, 1, 2) dtype=float32>,\n                        cached_value: <tf.Tensor 'StatefulPartitionedCall:3' shape=(2, 1, 1, 2) dtype=float32>,\n                    },\n                },\n                posembed_output: {\n                    cache_index: <tf.Tensor 'StatefulPartitionedCall:4' shape=() dtype=uint32>,\n                },\n            },\n        },\n    }) for key 'output_1' in the output of the function __inference_<lambda>_74438 used to generate the SavedModel signature 'serving_default'. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.\n")
| mnist | SUCCESS |
| nlp_seq | FAIL | ValueError("Error when tracing gradients for SavedModel.\n\nSee the stack trace above to see the error that was raised when converting a gradient function to a concrete function. You may need to update the custom gradient, or disable saving gradients with the option tf.saved_model.SaveOptions(custom_gradients=False).\n\tProblematic op name: IdentityN\n\tGradient inputs: (<tf.Tensor 'AddV2_12:0' shape=(2, 1, 8) dtype=float32>, <tf.Tensor 'jax2tf_arg_0:0' shape=(8,) dtype=float32>, <tf.Tensor 'jax2tf_arg_1:0' shape=(4, 8) dtype=float32>, <tf.Tensor 'jax2tf_arg_2:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_3:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_4:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_5:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_6:0' shape=(2,) dtype=float32>, <tf.Tensor 'jax2tf_arg_7:0' shape=(4, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_8:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_9:0' shape=(2, 4) dtype=float32>, <tf.Tensor 'jax2tf_arg_10:0' shape=(4, 1, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_11:0' shape=(1, 2, 4) dtype=float32>, <tf.Tensor 'jax2tf_arg_12:0' shape=(4, 1, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_13:0' shape=(4, 1, 2) dtype=float32>, <tf.Tensor 'jax2tf_arg_14:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_15:0' shape=(4,) dtype=float32>, <tf.Tensor 'jax2tf_arg_16:0' shape=(8, 4) dtype=float32>, <tf.Tensor 'jax2tf_arg_17:0' shape=(2, 1) dtype=float32>)")
| pixelcnn++ | SUCCESS |
| ppo | SUCCESS |
| seq2seq | FAIL | ValueError('Unsupported Ops in the model before optimization\nBitcast, BitwiseAnd, BitwiseOr, LeftShift, BitwiseXor, RightShift')
| sst2 | FAIL | ValueError('Unsupported Ops in the model before optimization\nBitwiseAnd')
| vae | SUCCESS |
| wmt | FAIL | ValueError("in user code:\n\n\n    ValueError: Got a non-Tensor value FrozenDict({\n        cache: {\n            decoder: {\n                encoderdecoderblock_0: {\n                    SelfAttention_0: {\n                        cache_index: <tf.Tensor 'StatefulPartitionedCall:1' shape=() dtype=int32>,\n                        cached_key: <tf.Tensor 'StatefulPartitionedCall:2' shape=(2, 1, 1, 2) dtype=float32>,\n                        cached_value: <tf.Tensor 'StatefulPartitionedCall:3' shape=(2, 1, 1, 2) dtype=float32>,\n                    },\n                },\n                posembed_output: {\n                    cache_index: <tf.Tensor 'StatefulPartitionedCall:4' shape=() dtype=uint32>,\n                },\n            },\n        },\n    }) for key 'output_1' in the output of the function __inference_<lambda>_229578 used to generate the SavedModel signature 'serving_default'. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.\n")

## Table generation

See `examples_test.py` for instructions on how to regenerate this table.
