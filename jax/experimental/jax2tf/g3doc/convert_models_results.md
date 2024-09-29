# JAX Converters Evaluation Results

*Last generated on: 2023-01-05* (YYYY-MM-DD)

This file contains the evaluation results for all converters in table format.

See [models_test_main.py](../tests/models_test_main.py) for instructions on how to
regenerate this table.

See [Description of Converters](#description-of-converters) below for more
details on the different converters.

## Summary Table

| Example | jax2tf_xla | jax2tf_noxla | jax2tfjs | jax2tflite | jax2tflite+flex |
| --- | --- | --- | --- | --- | --- |
| `flax/actor_critic` | YES | YES | YES | YES | YES |
| `flax/actor_critic_[(b, ...)]` | YES | YES | YES | YES | YES |
| `flax/actor_critic_[(_, 4*b, 4*b, _)]` | [NO](#example-flaxactor_critic__-4b-4b-_--converter-jax2tf_xla) | [NO](#example-flaxactor_critic__-4b-4b-_--converter-jax2tf_noxla) | [NO](#example-flaxactor_critic__-4b-4b-_--converter-jax2tfjs) | [NO](#example-flaxactor_critic__-4b-4b-_--converter-jax2tflite) | [NO](#example-flaxactor_critic__-4b-4b-_--converter-jax2tfliteflex) |
| `flax/bilstm` | YES | YES | YES | [NO](#example-flaxbilstm--converter-jax2tflite) | [NO](#example-flaxbilstm--converter-jax2tfliteflex) |
| `flax/bilstm_[(b, _), (_,)]` | [NO](#example-flaxbilstm_b-_-_--converter-jax2tf_xla) | [NO](#example-flaxbilstm_b-_-_--converter-jax2tf_noxla) | [NO](#example-flaxbilstm_b-_-_--converter-jax2tfjs) | [NO](#example-flaxbilstm_b-_-_--converter-jax2tflite) | [NO](#example-flaxbilstm_b-_-_--converter-jax2tfliteflex) |
| `flax/bilstm_[(_, _), (b,)]` | [NO](#example-flaxbilstm__-_-b--converter-jax2tf_xla) | [NO](#example-flaxbilstm__-_-b--converter-jax2tf_noxla) | [NO](#example-flaxbilstm__-_-b--converter-jax2tfjs) | [NO](#example-flaxbilstm__-_-b--converter-jax2tflite) | [NO](#example-flaxbilstm__-_-b--converter-jax2tfliteflex) |
| `flax/cnn` | YES | YES | YES | YES | YES |
| `flax/cnn_[(b, ...)]` | YES | YES | YES | YES | YES |
| `flax/cnn_[(_, b, b, _)]` | [NO](#example-flaxcnn__-b-b-_--converter-jax2tf_xla) | [NO](#example-flaxcnn__-b-b-_--converter-jax2tf_noxla) | [NO](#example-flaxcnn__-b-b-_--converter-jax2tfjs) | [NO](#example-flaxcnn__-b-b-_--converter-jax2tflite) | [NO](#example-flaxcnn__-b-b-_--converter-jax2tfliteflex) |
| `flax/gnn` | YES | [NO](#example-flaxgnn--converter-jax2tf_noxla) | [NO](#example-flaxgnn--converter-jax2tfjs) | [NO](#example-flaxgnn--converter-jax2tflite) | [NO](#example-flaxgnn--converter-jax2tfliteflex) |
| `flax/gnn_conv` | YES | [NO](#example-flaxgnn_conv--converter-jax2tf_noxla) | [NO](#example-flaxgnn_conv--converter-jax2tfjs) | [NO](#example-flaxgnn_conv--converter-jax2tflite) | [NO](#example-flaxgnn_conv--converter-jax2tfliteflex) |
| `flax/resnet50` | YES | YES | YES | YES | YES |
| `flax/resnet50_[(b, ...)]` | YES | YES | YES | [NO](#example-flaxresnet50_b---converter-jax2tflite) | [NO](#example-flaxresnet50_b---converter-jax2tfliteflex) |
| `flax/resnet50_[(_, 4*b, 4*b, _)]` | [NO](#example-flaxresnet50__-4b-4b-_--converter-jax2tf_xla) | [NO](#example-flaxresnet50__-4b-4b-_--converter-jax2tf_noxla) | [NO](#example-flaxresnet50__-4b-4b-_--converter-jax2tfjs) | [NO](#example-flaxresnet50__-4b-4b-_--converter-jax2tflite) | [NO](#example-flaxresnet50__-4b-4b-_--converter-jax2tfliteflex) |
| `flax/seq2seq_lstm` | YES | YES | [NO](#example-flaxseq2seq_lstm--converter-jax2tfjs) | [NO](#example-flaxseq2seq_lstm--converter-jax2tflite) | YES |
| `flax/seq2seq_lstm_[(b, _, _), (b, _, _)]` | YES | YES | [NO](#example-flaxseq2seq_lstm_b-_-_-b-_-_--converter-jax2tfjs) | [NO](#example-flaxseq2seq_lstm_b-_-_-b-_-_--converter-jax2tflite) | [NO](#example-flaxseq2seq_lstm_b-_-_-b-_-_--converter-jax2tfliteflex) |
| `flax/seq2seq_lstm_[(_, b, _), (_, _, _)]` | YES | YES | [NO](#example-flaxseq2seq_lstm__-b-_-_-_-_--converter-jax2tfjs) | [NO](#example-flaxseq2seq_lstm__-b-_-_-_-_--converter-jax2tflite) | [NO](#example-flaxseq2seq_lstm__-b-_-_-_-_--converter-jax2tfliteflex) |
| `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]` | [NO](#example-flaxseq2seq_lstm__-_-_-_-b-_--converter-jax2tf_xla) | [NO](#example-flaxseq2seq_lstm__-_-_-_-b-_--converter-jax2tf_noxla) | [NO](#example-flaxseq2seq_lstm__-_-_-_-b-_--converter-jax2tfjs) | [NO](#example-flaxseq2seq_lstm__-_-_-_-b-_--converter-jax2tflite) | [NO](#example-flaxseq2seq_lstm__-_-_-_-b-_--converter-jax2tfliteflex) |
| `flax/lm1b` | YES | YES | YES | YES | YES |
| `flax/nlp_seq` | YES | YES | YES | YES | YES |
| `flax/lm1b_[(b, _)]` | [NO](#example-flaxlm1b_b-_--converter-jax2tf_xla) | [NO](#example-flaxlm1b_b-_--converter-jax2tf_noxla) | [NO](#example-flaxlm1b_b-_--converter-jax2tfjs) | [NO](#example-flaxlm1b_b-_--converter-jax2tflite) | [NO](#example-flaxlm1b_b-_--converter-jax2tfliteflex) |
| `flax/nlp_seq_[(b, _)]` | YES | YES | YES | [NO](#example-flaxnlp_seq_b-_--converter-jax2tflite) | [NO](#example-flaxnlp_seq_b-_--converter-jax2tfliteflex) |
| `flax/wmt` | YES | YES | YES | YES | YES |
| `flax/wmt_[(b, _), (b, _)]` | [NO](#example-flaxwmt_b-_-b-_--converter-jax2tf_xla) | [NO](#example-flaxwmt_b-_-b-_--converter-jax2tf_noxla) | [NO](#example-flaxwmt_b-_-b-_--converter-jax2tfjs) | [NO](#example-flaxwmt_b-_-b-_--converter-jax2tflite) | [NO](#example-flaxwmt_b-_-b-_--converter-jax2tfliteflex) |
| `flax/wmt_[(_, b), (_, b)]` | [NO](#example-flaxwmt__-b-_-b--converter-jax2tf_xla) | [NO](#example-flaxwmt__-b-_-b--converter-jax2tf_noxla) | [NO](#example-flaxwmt__-b-_-b--converter-jax2tfjs) | [NO](#example-flaxwmt__-b-_-b--converter-jax2tflite) | [NO](#example-flaxwmt__-b-_-b--converter-jax2tfliteflex) |
| `flax/vae` | YES | YES | YES | YES | YES |
| `flax/vae_[(b, ...)]` | YES | YES | YES | YES | YES |
| `flax/vae_[(_, b, b, _)]` | YES | YES | YES | [NO](#example-flaxvae__-b-b-_--converter-jax2tflite) | [NO](#example-flaxvae__-b-b-_--converter-jax2tfliteflex) |

## Errors

## `flax/actor_critic_[(_, 4*b, 4*b, _)]`
### Example: `flax/actor_critic_[(_, 4*b, 4*b, _)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/actor_critic_[(_, 4*b, 4*b, _)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/actor_critic_[(_, 4*b, 4*b, _)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/actor_critic_[(_, 4*b, 4*b, _)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/actor_critic_[(_, 4*b, 4*b, _)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

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

## `flax/bilstm_[(b, _), (_,)]`
### Example: `flax/bilstm_[(b, _), (_,)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(b, _), (_,)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(b, _), (_,)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(b, _), (_,)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(b, _), (_,)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

## `flax/bilstm_[(_, _), (b,)]`
### Example: `flax/bilstm_[(_, _), (b,)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(_, _), (b,)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(_, _), (b,)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(_, _), (b,)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/bilstm_[(_, _), (b,)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

## `flax/cnn_[(_, b, b, _)]`
### Example: `flax/cnn_[(_, b, b, _)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Cannot compute stride for dimension 'b', window_size '2', stride '2'.\nDetails: Cannot divide 'b + -2' by '2'.\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/cnn_[(_, b, b, _)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Cannot compute stride for dimension 'b', window_size '2', stride '2'.\nDetails: Cannot divide 'b + -2' by '2'.\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/cnn_[(_, b, b, _)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Cannot compute stride for dimension 'b', window_size '2', stride '2'.
Details: Cannot divide 'b + -2' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/cnn_[(_, b, b, _)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Cannot compute stride for dimension 'b', window_size '2', stride '2'.
Details: Cannot divide 'b + -2' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/cnn_[(_, b, b, _)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Cannot compute stride for dimension 'b', window_size '2', stride '2'.
Details: Cannot divide 'b + -2' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
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
{{function_node __wrapped__UnsortedSegmentSum_device_/job:localhost/replica:0/task:0/device:CPU:0}} segment_ids[0] = 92 is out of range [0, 52) [Op:UnsortedSegmentSum]
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

## `flax/resnet50_[(b, ...)]`
### Example: `flax/resnet50_[(b, ...)]` | Converter: `jax2tflite`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 8 but expected 1 for dimension 0 of input 0.')
```
[Back to top](#summary-table)

### Example: `flax/resnet50_[(b, ...)]` | Converter: `jax2tflite+flex`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 8 but expected 1 for dimension 0 of input 0.')
```
[Back to top](#summary-table)

## `flax/resnet50_[(_, 4*b, 4*b, _)]`
### Example: `flax/resnet50_[(_, 4*b, 4*b, _)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/resnet50_[(_, 4*b, 4*b, _)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/resnet50_[(_, 4*b, 4*b, _)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/resnet50_[(_, 4*b, 4*b, _)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/resnet50_[(_, 4*b, 4*b, _)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Cannot divide '-1*b' by '2'.
See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#division-of-shape-polynomials-is-partially-supported.

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

## `flax/seq2seq_lstm`
### Example: `flax/seq2seq_lstm` | Converter: `jax2tfjs`
```
Conversion error
ValueError('Unsupported Ops in the model before optimization
BitwiseOr, BitwiseAnd, BitwiseXor, LeftShift, Bitcast, RightShift')
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
	tf.Slice(tensor<2x2xui32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<1x2xui32>) : {device = ""}
	tf.StridedSlice(tensor<2xui32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
```
[Back to top](#summary-table)

## `flax/seq2seq_lstm_[(b, _, _), (b, _, _)]`
### Example: `flax/seq2seq_lstm_[(b, _, _), (b, _, _)]` | Converter: `jax2tfjs`
```
Conversion error
ValueError('Unsupported Ops in the model before optimization
BitwiseOr, BitwiseAnd, BitwiseXor, LeftShift, Bitcast, RightShift')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(b, _, _), (b, _, _)]` | Converter: `jax2tflite`
```
Conversion error
Some ops are not supported by the native TFLite runtime
	tf.Bitcast(tensor<?x4xui32>) -> (tensor<?x4xf32>) : {device = ""}
	tf.BitwiseOr(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.BitwiseOr(tensor<?x4xui32>, tensor<ui32>) -> (tensor<?x4xui32>) : {device = ""}
	tf.BitwiseOr(tensor<?xui32>, tensor<?xui32>) -> (tensor<?xui32>) : {device = ""}
	tf.BitwiseXor(tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.BitwiseXor(tensor<?xui32>, tensor<?xui32>) -> (tensor<?xui32>) : {device = ""}
	tf.BitwiseXor(tensor<ui32>, tensor<ui32>) -> (tensor<ui32>) : {device = ""}
	tf.ConcatV2(tensor<1xui32>, tensor<1xui32>, tensor<i32>) -> (tensor<2xui32>)
	tf.ConcatV2(tensor<?xui32>, tensor<?xui32>, tensor<i32>) -> (tensor<?xui32>) : {device = ""}
	tf.LeftShift(tensor<1xui32>, tensor<ui32>) -> (tensor<1xui32>) : {device = ""}
	tf.LeftShift(tensor<?xui32>, tensor<ui32>) -> (tensor<?xui32>) : {device = ""}
	tf.Pack(tensor<ui32>, tensor<ui32>) -> (tensor<2xui32>) : {axis = 0 : i64}
	tf.RightShift(tensor<1xui32>, tensor<ui32>) -> (tensor<1xui32>) : {device = ""}
	tf.RightShift(tensor<?x4xui32>, tensor<ui32>) -> (tensor<?x4xui32>) : {device = ""}
	tf.RightShift(tensor<?xui32>, tensor<ui32>) -> (tensor<?xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<1xui32>, tensor<1xui32>) -> (tensor<1xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<?x4xui32>, tensor<?x4xui32>) -> (tensor<?x4xui32>) : {device = ""}
	tf.SelectV2(tensor<i1>, tensor<?xui32>, tensor<?xui32>) -> (tensor<?xui32>) : {device = ""}
	tf.Slice(tensor<2x2xui32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<1x2xui32>) : {device = ""}
	tf.StridedSlice(tensor<2xui32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
	tf.StridedSlice(tensor<?xui32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<?xui32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
	tf.ZerosLike(tensor<?x4xui32>) -> (tensor<?x4xui32>) : {device = ""}
	tf.ZerosLike(tensor<?xui32>) -> (tensor<?xui32>) : {device = ""}
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(b, _, _), (b, _, _)]` | Converter: `jax2tflite+flex`
```
RuntimeError('TF Lite does not support TensorFlow data type: uint32FlexDelegate: Tensor jax2tf_apply_with_vars_/Seq2seq/Decoder_0/scan/while/body/DecoderLSTM_0/random_bits/jit__threefry_random_bits_original_/jit_threefry_2x32_/strided_slice_2(68) buffer size mismatch 8(2) != 4(1)failed to copy data from TF tensorNode number 662 (TfLiteFlexDelegate) failed to invoke.Node number 25 (WHILE) failed to invoke.')
```
[Back to top](#summary-table)

## `flax/seq2seq_lstm_[(_, b, _), (_, _, _)]`
### Example: `flax/seq2seq_lstm_[(_, b, _), (_, _, _)]` | Converter: `jax2tfjs`
```
Conversion error
ValueError('Unsupported Ops in the model before optimization
BitwiseOr, BitwiseAnd, BitwiseXor, LeftShift, Bitcast, RightShift')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(_, b, _), (_, _, _)]` | Converter: `jax2tflite`
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
	tf.Slice(tensor<2x2xui32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<1x2xui32>) : {device = ""}
	tf.StridedSlice(tensor<2xui32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<1xui32>) : {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64}
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(_, b, _), (_, _, _)]` | Converter: `jax2tflite+flex`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 2 but expected 1 for dimension 1 of input 0.')
```
[Back to top](#summary-table)

## `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]`
### Example: `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]` | Converter: `jax2tf_xla`
```
IndexError('Cannot use NumPy slice indexing on an array dimension whose size is not statically known (b). Try using lax.dynamic_slice/dynamic_update_slice')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]` | Converter: `jax2tf_noxla`
```
IndexError('Cannot use NumPy slice indexing on an array dimension whose size is not statically known (b). Try using lax.dynamic_slice/dynamic_update_slice')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]` | Converter: `jax2tfjs`
```
Conversion error
IndexError('Cannot use NumPy slice indexing on an array dimension whose size is not statically known (b). Try using lax.dynamic_slice/dynamic_update_slice')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]` | Converter: `jax2tflite`
```
Conversion error
IndexError('Cannot use NumPy slice indexing on an array dimension whose size is not statically known (b). Try using lax.dynamic_slice/dynamic_update_slice')
```
[Back to top](#summary-table)

### Example: `flax/seq2seq_lstm_[(_, _, _), (_, b, _)]` | Converter: `jax2tflite+flex`
```
Conversion error
IndexError('Cannot use NumPy slice indexing on an array dimension whose size is not statically known (b). Try using lax.dynamic_slice/dynamic_update_slice')
```
[Back to top](#summary-table)

## `flax/lm1b_[(b, _)]`
### Example: `flax/lm1b_[(b, _)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/lm1b_[(b, _)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/lm1b_[(b, _)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/lm1b_[(b, _)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/lm1b_[(b, _)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

## `flax/nlp_seq_[(b, _)]`
### Example: `flax/nlp_seq_[(b, _)]` | Converter: `jax2tflite`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 2 but expected 1 for dimension 0 of input 0.')
```
[Back to top](#summary-table)

### Example: `flax/nlp_seq_[(b, _)]` | Converter: `jax2tflite+flex`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 2 but expected 1 for dimension 0 of input 0.')
```
[Back to top](#summary-table)

## `flax/wmt_[(b, _), (b, _)]`
### Example: `flax/wmt_[(b, _), (b, _)]` | Converter: `jax2tf_xla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(b, _), (b, _)]` | Converter: `jax2tf_noxla`
```
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive\n\nThis error arises for arithmetic or comparison operations with shapes that\nare non-constant, and the result of the operation cannot be represented as\na polynomial of dimension variables, or a boolean constant (for comparisons).\n\nPlease see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables\nfor more details.\n")
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(b, _), (b, _)]` | Converter: `jax2tfjs`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(b, _), (b, _)]` | Converter: `jax2tflite`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(b, _), (b, _)]` | Converter: `jax2tflite+flex`
```
Conversion error
InconclusiveDimensionOperation("Dimension polynomial comparison 'b' == '2' is inconclusive

This error arises for arithmetic or comparison operations with shapes that
are non-constant, and the result of the operation cannot be represented as
a polynomial of dimension variables, or a boolean constant (for comparisons).

Please see https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#computing-with-dimension-variables
for more details.
")
```
[Back to top](#summary-table)

## `flax/wmt_[(_, b), (_, b)]`
### Example: `flax/wmt_[(_, b), (_, b)]` | Converter: `jax2tf_xla`
```
IndexError('Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found slice(None, b, None). To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays within JIT compiled functions).')
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(_, b), (_, b)]` | Converter: `jax2tf_noxla`
```
IndexError('Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found slice(None, b, None). To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays within JIT compiled functions).')
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(_, b), (_, b)]` | Converter: `jax2tfjs`
```
Conversion error
IndexError('Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found slice(None, b, None). To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays within JIT compiled functions).')
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(_, b), (_, b)]` | Converter: `jax2tflite`
```
Conversion error
IndexError('Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found slice(None, b, None). To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays within JIT compiled functions).')
```
[Back to top](#summary-table)

### Example: `flax/wmt_[(_, b), (_, b)]` | Converter: `jax2tflite+flex`
```
Conversion error
IndexError('Array slice indices must have static start/stop/step to be used with NumPy indexing syntax. Found slice(None, b, None). To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice (JAX does not support dynamically sized arrays within JIT compiled functions).')
```
[Back to top](#summary-table)

## `flax/vae_[(_, b, b, _)]`
### Example: `flax/vae_[(_, b, b, _)]` | Converter: `jax2tflite`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 8 but expected 1 for dimension 1 of input 0.')
```
[Back to top](#summary-table)

### Example: `flax/vae_[(_, b, b, _)]` | Converter: `jax2tflite+flex`
```
ValueError('Cannot set tensor: Dimension mismatch. Got 8 but expected 1 for dimension 1 of input 0.')
```
[Back to top](#summary-table)


## Description of Converters

Below is a description of all converters that can be found in
[converters.py](converters.py).

### `jax2tf_xla`

This converter simply converts a the forward function of a JAX model to a
Tensorflow function with XLA support linked in. This is considered the baseline
converter and has the largest coverage, because we expect nearly all ops to be
convertible. However, please see
[jax2tf Known Issue](https://github.com/jax-ml/jax/tree/main/jax/experimental/jax2tf#known-issues)
for a list of known problems.

### `jax2tf_noxla`

This converter converts a JAX model to a Tensorflow function without XLA
support. This means the Tensorflow XLA ops aren't used. See
[here](https://github.com/jax-ml/jax/tree/main/jax/experimental/jax2tf#tensorflow-xla-ops)
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
