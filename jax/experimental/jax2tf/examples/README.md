
This directory contains a number of examples of using the jax2tf converter to:

  * save SavedModel from trained MNIST models, using both pure JAX and Flax.
  * load the SavedModel into the TensorFlow model server and use it for
    inference. We show also how to use a batch-polymorphic saved model.
  * reuse the feature-extractor part of the trained MNIST model in
    TensorFlow Hub, and in a larger TensorFlow Keras model.

Preparing the model for jax2tf
==============================

The most important detail for using jax2tf with SavedModel is to express
the trained model as a pair:

  * `func`: a two-argument function with signature:
   `(params: Parameters, inputs: Inputs) -> Outputs`.
   Both arguments must be a `numpy.ndarray` or
   nested tuple/list/dictionaries thereof.
  * `params`: of type `Parameters`, the model parameters, to be used as the
  first input for `func`.

You can see in [mnist_lib.py](mnist_lib.py) how this can be done for two 
implementations of MNIST, one using pure JAX (`PureJaxMNIST`) and a CNN
one using Flax (`FlaxMNIST`). Other Flax models can be arranged similarly,
and the same strategy should work for other neural-network libraries for JAX.

Generating TensorFlow SavedModel
=====================

Once you have the model in this form, you can use the
`saved_model_lib.save_model` function [saved_model_lib.py](saved_model_lib.py)
to generate the SavedModel. There is very little in that function that is
specific to jax2tf. The goal of jax2tf is to convert JAX functions
into functions that behave as if they had been written with TensorFlow.
Therefore, if you are familiar with how to generate SavedModel, you can most
likely just use your own code for this.

The file `saved_model_main.py` is an executable that shows who to perform the following
sequence of steps:

   * train an MNIST model, and obtain a pair of an inference function and the parameters.
   * convert the inference function with jax2tf, for one of more batch sizes.
   * save a SavedModel and dump its contents.
   * reload the SavedModel and run it with TensorFlow to test that the inference
   function produces the same results as the JAX inference function.
   * optionally plot images with the training digits and the inference results.

There are a number of flags to select the Flax model (`--model=mnist_flag`),
to skip the training and just test a previously loaded
SavedModel (`--nogenerate_model`), to choose the saving path, etc.

By default, this example will convert the inference function for three separate
batch sizes: 1, 16, 128. You can see this in the dumped SavedModel. If you
pass the argument `--serving_batch_size=-1`, then the conversion to TF will
be done using jax2tf's
[experimental shape-polymorphism feature](https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion).
As a result, the inference function will be traced only once and the SavedModel
will contain a single batch-polymorphic TensorFlow graph.

Using the TensorFlow model server
=================================

The executable `servo_main.py` extends the `saved_model_main.py` with code to
show how to use the SavedModel with TensorFlow model server. All the flags
of `saved_model_main.py` also apply to `servo_main.py`. In particular, you
can select the Flax MNIST model: `--model=mnist_flax`,
batch-polymorphic conversion to TensorFlow `--serving_batch_size=-1`,
skip the training, reuse a previously trained model: `--nogenerate_model`.

If you want to start your own model server, you should pass the
`--nostart_model_server` flag and also `--serving_url` to point to the
HTTP REST API end point of your model server. You can see the path of the
trained and saved model in the output.

Open-source model server
------------------------

We have tried this example with the OSS model server, using a
[TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker).
Specifically, you need to install Docker and run

```
docker pull tensorflow/serving
```

The actual starting of the model server is done by the `servo_main.py` script.
The script also does a sanity check that the model server is serving a model
with the right batch size and image shapes, and makes a few requests to the
server to perform inference.

Instructions for using the Google internal version of the model server: TBA.


Reusing models with TensorFlow Hub and TensorFlow Keras
=======================================================

The SavedModel produced by the example in `saved_model_main.py` already
implements the [reusable saved models interface](https://www.tensorflow.org/hub/reusable_saved_models).
The executable `keras_reuse_main.py` extends `saved_model_main.py` with code
to include a jax2tf SavedModel into a larger TensorFlow Keras model.

In order to demonstrate reuse, we have added a keyword argument `with_classifier`
to the MNIST models. When this flag is set to False, the model will not include
the last classification layer. The SavedModel will compute just the logits.

We show in `keras_reuse_main.py` how to use `hub.KerasLayer` to turn the
SavedModel into a Keras layer, to which we add a Keras classification layer,
that is then trained in TensorFlow.

For ease of training, we show this with the high-level Keras API, but it should
also be possible in low-level TensorFlow (using the restored_features under
a `tf.GradientTape` and doing gradient updates manually).

All the flags of `saved_model_main.py` also apply to `keras_reuse_main.py`.
In particular, you can select the Flax MNIST model: `--model=mnist_flax`.
