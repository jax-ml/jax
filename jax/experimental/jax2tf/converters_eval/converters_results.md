# JAX Converters Evaluation Results

*Last generated on: 2022-05-29* (YYYY-MM-DD)

This file contains the evaluation results for all converters in table format.
Please see [README.md](README.md) for more details.

## Summary Table

| Example | jax2tf_xla | jax2tf_to_tfjs | jax2tf_to_tflite |
| --- | --- | --- | --- |
| flax/actor_critic | YES | YES | YES |
| flax/bilstm | YES | YES | [NO](#error-trace-modelflaxbilstm-converterjax2tf_to_tflite) | 
| flax/cnn | YES | YES | YES |
| flax/resnet50 | YES | YES | YES |
| flax/seq2seq_lstm | YES | [NO](#error-trace-modelflaxseq2seq_lstm-converterjax2tf_to_tfjs) |  YES |
| flax/transformer_lm1b | [NO](#error-trace-modelflaxtransformer_lm1b-converterjax2tf_xla) |  [NO](#error-trace-modelflaxtransformer_lm1b-converterjax2tf_to_tfjs) |  [NO](#error-trace-modelflaxtransformer_lm1b-converterjax2tf_to_tflite) | 
| flax/transformer_nlp_seq | YES | YES | YES |
| flax/transformer_wmt | [NO](#error-trace-modelflaxtransformer_wmt-converterjax2tf_xla) |  [NO](#error-trace-modelflaxtransformer_wmt-converterjax2tf_to_tfjs) |  [NO](#error-trace-modelflaxtransformer_wmt-converterjax2tf_to_tflite) | 
| flax/vae | YES | YES | YES |

## Errors

## Error trace: model=flax/bilstm, converter=jax2tf_to_tflite
```
RuntimeError('third_party/tensorflow/lite/kernels/concatenation.cc:158 t->dims->data[d] != t0->dims->data[d] (3 != 1)Node number 11 (CONCATENATION) failed to prepare.Node number 29 (WHILE) failed to invoke.')
```
[Back to top](#summary-table)
## Error trace: model=flax/seq2seq_lstm, converter=jax2tf_to_tfjs
```
ValueError('Unsupported Ops in the model before optimization
RightShift, Bitcast, BitwiseOr, BitwiseAnd, LeftShift, BitwiseXor')
```
[Back to top](#summary-table)
## Error trace: model=flax/transformer_lm1b, converter=jax2tf_xla
```
InvalidArgumentError()
```
[Back to top](#summary-table)
## Error trace: model=flax/transformer_lm1b, converter=jax2tf_to_tfjs
```
ValueError("in user code:


    ValueError: Got a non-Tensor value FrozenDict({
        cache: {
            decoder: {
                encoderdecoderblock_0: {
                    SelfAttention_0: {
                        cache_index: <tf.Tensor 'StatefulPartitionedCall:1' shape=() dtype=int32>,
                        cached_key: <tf.Tensor 'StatefulPartitionedCall:2' shape=(2, 1, 1, 2) dtype=float32>,
                        cached_value: <tf.Tensor 'StatefulPartitionedCall:3' shape=(2, 1, 1, 2) dtype=float32>,
                    },
                },
                posembed_output: {
                    cache_index: <tf.Tensor 'StatefulPartitionedCall:4' shape=() dtype=uint32>,
                },
            },
        },
    }) for key 'output_1' in the output of the function __inference_tf_graph_261080 used to generate the SavedModel signature 'serving_default'. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.
")
```
[Back to top](#summary-table)
## Error trace: model=flax/transformer_lm1b, converter=jax2tf_to_tflite
```
TypeError("The DType <class 'numpy._FloatAbstractDType'> could not be promoted by <class 'numpy.dtype[str_]'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtype[str_]'>, <class 'numpy._FloatAbstractDType'>)")
```
[Back to top](#summary-table)
## Error trace: model=flax/transformer_wmt, converter=jax2tf_xla
```
InvalidArgumentError()
```
[Back to top](#summary-table)
## Error trace: model=flax/transformer_wmt, converter=jax2tf_to_tfjs
```
ValueError("in user code:


    ValueError: Got a non-Tensor value FrozenDict({
        cache: {
            decoder: {
                encoderdecoderblock_0: {
                    SelfAttention_0: {
                        cache_index: <tf.Tensor 'StatefulPartitionedCall:1' shape=() dtype=int32>,
                        cached_key: <tf.Tensor 'StatefulPartitionedCall:2' shape=(2, 1, 1, 2) dtype=float32>,
                        cached_value: <tf.Tensor 'StatefulPartitionedCall:3' shape=(2, 1, 1, 2) dtype=float32>,
                    },
                },
                posembed_output: {
                    cache_index: <tf.Tensor 'StatefulPartitionedCall:4' shape=() dtype=uint32>,
                },
            },
        },
    }) for key 'output_1' in the output of the function __inference_tf_graph_280065 used to generate the SavedModel signature 'serving_default'. Outputs for functions used as signatures must be a single Tensor, a sequence of Tensors, or a dictionary from string to Tensor.
")
```
[Back to top](#summary-table)
## Error trace: model=flax/transformer_wmt, converter=jax2tf_to_tflite
```
TypeError("The DType <class 'numpy._FloatAbstractDType'> could not be promoted by <class 'numpy.dtype[str_]'>. This means that no common DType exists for the given inputs. For example they cannot be stored in a single array unless the dtype is `object`. The full list of DTypes is: (<class 'numpy.dtype[str_]'>, <class 'numpy._FloatAbstractDType'>)")
```
[Back to top](#summary-table)

See `model_test.py` for instructions on how to regenerate this table.
