# Evaluating JAX Converters for TFLite and TensorFlow.js

This directory evaluates various JAX converters by attempting to convert a range
of examples and reporting the errors. It also contains documentation for the
current limitations.

## How does the evaluation work?

When running [converters_eval_main.py](converters_eval_main.py), each converter
in [converters.py](converters.py) is evaluated on a list of examples in
[examples.py](examples.py). Each example is a JAX function that may consist of
a large number of JAX primitives (e.g., a Convolutional Neural Network).

Given a JAX function `f`, and a converter function `c: F -> F`, the evaluation
runs as follows:

1. Execute `f' = c(f)` and report an error if this fails.
2. Given random input `X`, assert `almost_close(f(X), f'(X), rtol)`.

The value of `rtol` depends on the tolerances specified by jax2tf in
[jax2tf_limitations.py](../tests/jax2tf_limitations.py). For instance, for
`conv_general_dilated` the expected tolerance is `1e-4` on CPU, so if we run a
very large ResNet model, we can expect rather high tolerances here.

The evaluation results are written in table format to
[converters_results.md](converters_results.md).

## jax2tf without XLA: Known Limitations

Since the TF.js and TFLite converters don't have the XLA compiler linked in, we
cannot use a number of JAX ops directly, because they don't have a corresponding
op in TF. For this, we provide support on a case-by-case basis.

To track this, we use a list of known limitations of the `jax2tf` emitter when
XLA support is not available in [no_xla_limitations.md](../g3doc/no_xla_limitations.md).

## Description of Converters

Below is a description of all converters that can be found in
[converters.py](converters.py).

### `jax2tf_xla`

This converter simply converts a JAX model to TF SavedModel with XLA support.
This is considered the baseline converter and has the largest coverage, because
we expect all ops to be convertible. However, please see
[jax2tf Known Issue](https://github.com/google/jax/tree/main/jax/experimental/jax2tf#known-issues)
for a list of known problems.

### `jax2tf_to_tflite`

This converter first converts a JAX model to TF SavedModel format without XLA
support. Please see [no_xla_limitations.md](../g3doc/no_xla_limitations.md) for a list
of known limitations for this conversion step.

After that, it converts the SavedModel to TFLite using the
[TFLite converter](https://www.tensorflow.org/lite/convert).

### `jax2tf_to_tfjs`

This converter first converts a JAX model to TF SavedModel format without XLA
support. Please see [no_xla_limitations.md](../g3doc/no_xla_limitations.md) for a list
of known limitations for this conversion step.

After that, it converts the SavedModel to TF.js using the
[TF.js converter](https://www.tensorflow.org/js/guide/conversioni).
