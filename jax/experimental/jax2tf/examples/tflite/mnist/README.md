# Image classification on the MNIST dataset for on-device machine learning inference

This directory contains the code that demonstrates how to:

1. Train a simple convolutional neural network on the MNIST dataset; and
2. Use jax2tf — that helps stage JAX programs out as TensorFlow graphs — into a TensorFlow [SavedModel](https://www.tensorflow.org/guide/saved_model) that can be used again for conversion into a [TensorFlow Lite (TF Lite)](https://www.tensorflow.org/lite/) format for on-device machine learning inference.

This example is based on the [TensorFlow Lite demo](https://developer.android.com/codelabs/digit-classifier-tflite) of building a handwritten digit classifier app for Android with TensorFlow Lite. The model training code is based on the [Flax Linen example](https://github.com/google/flax/tree/master/linen_examples/mnist) for classification on the MNIST dataset.

## Requirements

* This example uses [Flax](http://github.com/google/flax) and [TensorFlow Datasets](https://www.tensorflow.org/datasets). The code downloads the MNIST dataset from TensorFlow Datasets and preprocesses it before feeding the data into the neural network as input.

## Training the model

To train the model in JAX with the Flax library, convert it into a TensorFlow SavedModel, and export it into the TF Lite format, run the following command:

```shell
python mnist.py
```

The training itself should take about 1 minute, assuming the dataset has already been downloaded. The dataset is loaded directly into a `data/` directory under `/tmp/jax2tf/mnist`. 

This example's training loop runs for 10 epochs for demonstration purposes. After training, the model is converted to SavedModels and then — TF Lite. In the end, it is saved as `mnist.tflite` under `/tmp/jax2tf/mnist/`.

### Convert to TensorFlow function using the jax2tf converter

You can write a prediction function from the trained model and use the jax2tf converter to convert it to a TF function as follows:

```python
def predict(image):
   return CNN().apply({'params': optimizer.target}, image)
```
```python
# Convert your Flax model to TF function.
tf_predict = tf.function(
    jax2tf.convert(predict, enable_xla=False),
    input_signature=[
        tf.TensorSpec(shape=[1, 28, 28, 1], dtype=tf.float32, name='input')
    ],
    autograph=False)
```

### Convert the trained model to the TF Lite format

The [TF Lite converter](https://www.tensorflow.org/lite/convert#python_api_) provides a number of ways to convert a TensorFlow model to the TF format. For example, you can  use the `from_concrete_functions` API as follows:

```python
# Convert your TF function to the TF Lite format.
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf_predict.get_concrete_function()])

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_float_model = converter.convert()
```

Note that the use of `tf.lite.OpsSet.SELECT_TF_OPS` in supported ops is required here to run regular TensorFlow ops (as compared to TensorFlow Lite ops) in the TF Lite runtime.

### Apply quantization

The next step is to perform [quantization](https://www.tensorflow.org/lite/performance/post_training_quantization). Because the converted format is no different than the one converted from a regular TensorFlow SavedModel, you can apply quantization:

```python
# Re-convert the model to TF Lite using quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
```

## Deploy the TF Lite model to your Android app

To deploy the TF Lite model to your Android app, follow the instructions in the [Build a handwritten digit classifier app](https://developer.android.com/codelabs/digit-classifier-tflite) codelab.

When using a TF Lite model that has been converted with the support for select TensorFlow ops, the client must also use the TF Lite runtime that includes the necessary library of TensorFlow ops. In this example, you can use the standard TensorFlow [AAR](https://developer.android.com/studio/projects/android-library#aar-contents) file, by specifying the [nightly dependencies](https://www.tensorflow.org/lite/guide/ops_select#android_aar).

```
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly'
}
```

You can also specify `abiFilters` to reduce the size of the TensorFlow op dependencies.

```
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

## Limitations

* The TF Lite runtime currently doesn't support XLA ops. Therefore, you need to specify `enable_xla=False` to use the jax2tf conversion purely based on TensorFlow (not XLA). The support is limited to a subset of JAX ops.
