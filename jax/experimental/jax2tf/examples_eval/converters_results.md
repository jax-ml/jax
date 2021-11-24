# Evaluation Results

*Last generated on: 2021-11-22* (YYYY-MM-DD)

## jax2tf --> TFLite

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | SUCCESS |
| lm1b | FAIL | ValueError('For JAX vs TF (enable_xla=False): Numerical difference jax_result=[[[ 1.0656209e+00  9.3210316e-01 -7.5562042e-01  5.7160920e-01\n    4.7576640e-04 -8.3388436e-01 -6.6835815e-01  8.1217813e-01]]\n\n [[ 1.0656208e+00  9.3210304e-01 -7.5562... (CROPPED)
| mnist | SUCCESS |
| nlp_seq | SUCCESS |
| pixelcnn++ | FAIL | ValueError('For JAX vs TF (enable_xla=False): Numerical difference jax_result=[[[[-1.42588705e-01  4.43906128e-01 -4.43267524e-01 ...  2.41633713e-01\n     4.30841953e-01 -3.73984545e-01]\n   [-4.31387722e-01  5.41435003e-01 -2.79315412e-01 ... -4.81... (CROPPED)
| ppo | FAIL | ValueError('For JAX vs TF (enable_xla=False): Numerical difference jax_result=[[-2.0793843 -2.079457  -2.0794344 -2.079432  -2.0793545 -2.0794942\n  -2.079453  -2.0795226]] vs tf_result=[[-2.0794415 -2.0794415 -2.0794415 -2.0794415 -2.0794415 -2.0794... (CROPPED)
| seq2seq | SUCCESS |
| sst2 | FAIL | RuntimeError('tensorflow/lite/kernels/concatenation.cc:158 t->dims->data[d] != t0->dims->data[d] (3 != 1)Node number 11 (CONCATENATION) failed to prepare.Node number 7 (WHILE) failed to invoke.')
| vae | FAIL | ValueError('For JAX vs TF (enable_xla=False): Numerical difference jax_result=[[[[1.00000000e+00 1.59054339e-10 2.23751954e-06 ... 0.00000000e+00\n    9.99998808e-01 2.93910057e-12]\n   [1.00000000e+00 1.49735279e-04 1.17003319e-05 ... 2.77897666e-16... (CROPPED)
| wmt | SUCCESS |

## jax2tf --> TFjs

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | SUCCESS |
| lm1b | FAIL | ValueError("Error when tracing gradients for SavedModel.\n\nCheck the error log to see the error that was raised when converting a gradient function to a concrete function. You may need to update the custom gradient, or disable saving gradients with ... (CROPPED)
| mnist | SUCCESS |
| nlp_seq | FAIL | ValueError("Error when tracing gradients for SavedModel.\n\nCheck the error log to see the error that was raised when converting a gradient function to a concrete function. You may need to update the custom gradient, or disable saving gradients with ... (CROPPED)
| pixelcnn++ | SUCCESS |
| ppo | SUCCESS |
| seq2seq | FAIL | ValueError('Unsupported Ops in the model before optimization\nLeftShift, BitwiseOr, Bitcast, BitwiseAnd, RightShift, BitwiseXor')
| sst2 | FAIL | ValueError('Unsupported Ops in the model before optimization\nBitwiseAnd')
| vae | SUCCESS |
| wmt | FAIL | ValueError("Error when tracing gradients for SavedModel.\n\nCheck the error log to see the error that was raised when converting a gradient function to a concrete function. You may need to update the custom gradient, or disable saving gradients with ... (CROPPED)

## jax2tf (enable_xla=True)

### The Flax Examples
[URL to examples](https://github.com/google/flax/tree/main/examples)

Description: List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.

| Example | Result | Error Message |
| --- | --- | --- |
| imagenet | SUCCESS |
| lm1b | FAIL | TypeError("Value passed to parameter 'start_indices' has DataType uint32 not in list of allowed values: int32, int64")
| mnist | SUCCESS |
| nlp_seq | SUCCESS |
| pixelcnn++ | FAIL | ValueError('For JAX vs TF (enable_xla=True): Numerical difference jax_result=[[[[-1.42588705e-01  4.43906128e-01 -4.43267524e-01 ...  2.41633713e-01\n     4.30841953e-01 -3.73984545e-01]\n   [-4.31387722e-01  5.41435003e-01 -2.79315412e-01 ... -4.817... (CROPPED)
| ppo | SUCCESS |
| seq2seq | SUCCESS |
| sst2 | SUCCESS |
| vae | SUCCESS |
| wmt | FAIL | TypeError("Value passed to parameter 'start_indices' has DataType uint32 not in list of allowed values: int32, int64")

## Table generation

See `examples_test.py` for instructions on how to regenerate this table.
