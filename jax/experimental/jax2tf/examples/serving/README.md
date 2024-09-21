Using jax2tf with TensorFlow serving
====================================

This is a supplement to the
[examples/README.md](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/examples/README.md)
with example code and
instructions for using `jax2tf` with the open source TensorFlow model server.
Specific instructions for Google-internal versions of model server are in the `internal` subdirectory.

The goal of `jax2tf` is to convert JAX functions
into Python functions that behave as if they had been written with TensorFlow.
These functions can be traced and saved in a SavedModel using **standard TensorFlow
code, so the user has full control over what metadata is saved in the
SavedModel**.

The only difference in the SavedModel produced with jax2tf is that the
function graphs may contain
[XLA TF ops](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#caveats)
that require enabling CPU/GPU XLA for execution in the model server. This
is achieved using a command-line flag. There are no other differences compared
to using SavedModel produced by TensorFlow.

This serving example uses
[saved_model_main.py](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/examples/saved_model_main.py)
for saving the SavedModel and adds code specific to interacting with the
model server:
[model_server_request.py](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/examples/serving/model_server_request.py).

0.  *Set up JAX and TensorFlow serving*.

If you have already installed JAX and TensorFlow serving, you can skip most of these steps, but do set the
environment variables `JAX2TF_EXAMPLES` and `DOCKER_IMAGE`.

The following will clone locally a copy of the JAX sources, and will install the `jax`, `jaxlib`, and `flax` packages.
We also need to install TensorFlow for the `jax2tf` feature and the rest of this example.
We use the `tf_nightly` package to get an up-to-date version.

  ```shell
  git clone https://github.com/jax-ml/jax
  JAX2TF_EXAMPLES=$(pwd)/jax/jax/experimental/jax2tf/examples
  pip install -e jax
  pip install flax jaxlib tensorflow_datasets tensorflow_serving_api tf_nightly
  ```

We then install [TensorFlow serving in a Docker image](https://www.tensorflow.org/tfx/serving/docker),
again using a recent "nightly" version. Install Docker and then run.

  ```shell
  DOCKER_IMAGE=tensorflow/serving:nightly
  docker pull ${DOCKER_IMAGE}
  ```

1.  *Set some variables*.

    ```shell
    # Shortcuts
    # Where to save SavedModels
    MODEL_PATH=/tmp/jax2tf/saved_models
    # The example model. The options are "mnist_flax" and "mnist_pure_jax"
    MODEL=mnist_flax
    # The batch size for the SavedModel. Use -1 for batch-polymorphism,
    # or a strictly positive value for a fixed batch size.
    SERVING_BATCH_SIZE_SAVE=-1
    # The batch size to send to the model. Must be equal to SERVING_BATCH_SIZE_SAVE
    # if not -1.
    SERVING_BATCH_SIZE=16
    # Increment this when you make changes to the model parameters after the
    # initial model generation (Step 1 below).
    MODEL_VERSION=$(( 1 + ${MODEL_VERSION:-0} ))
    ```

2.  *Train and export a model.* You will use `saved_model_main.py` to train
   and export a SavedModel (see more details in README.md). Execute:

    ```shell
    python ${JAX2TF_EXAMPLES}/saved_model_main.py --model=${MODEL} \
        --model_path=${MODEL_PATH} --model_version=${MODEL_VERSION} \
        --serving_batch_size=${SERVING_BATCH_SIZE_SAVE} \
        --compile_model \
        --noshow_model
    ```

    The SavedModel will be in ${MODEL_PATH}/${MODEL}/${MODEL_VERSION}.
    You can inspect the SavedModel.

    ```shell
    saved_model_cli show --all --dir ${MODEL_PATH}/${MODEL}/${MODEL_VERSION}
    ```

    If you see a `-1` as the first element of the `shape`, then you have
    a batch polymorphic model.

3.  *Start a local model server* with XLA compilation enabled. Execute:

    ```shell
    docker run -p 8500:8500 -p 8501:8501 \
       --mount type=bind,source=${MODEL_PATH}/${MODEL}/,target=/models/${MODEL} \
       -e MODEL_NAME=${MODEL} -t --rm --name=serving ${DOCKER_IMAGE}  \
       --xla_cpu_compilation_enabled=true &
    ```

    Note that we are forwarding the ports 8500 (for the gRPC server) and
    8501 (for the HTTP REST server).

    You do not need to redo this step if you change the model parameters and
    regenerate the SavedModel, as long as you bumped the ${MODEL_VERSION}.
    The running model server will automatically load newer model versions.


4.  *Send RPC requests.* Execute:

    ```shell
    python ${JAX2TF_EXAMPLES}/serving/model_server_request.py --model_spec_name=${MODEL} \
        --use_grpc --prediction_service_addr=localhost:8500 \
        --serving_batch_size=${SERVING_BATCH_SIZE} \
        --count_images=128
    ```

    If you see an error `Input to reshape is a tensor with 12544 values, but the requested shape has 784`
    then your serving batch size is 16 (= 12544 / 784) while the model loaded
    in the model server has batch size 1 (= 784 / 784). You should check that
    you are using the same --serving_batch_size for the model generation and
    for sending the requests (or are using batch-polymorphic model generation.)

5.  *Experiment with different models and batch sizes*.

    - You can set `MODEL=mnist_pure_jax` to use a simpler model, using just
    pure JAX, then restart from Step 1.

    - If you created a batch-polymorphic models (`SERVING_BATCH_SIZE_SAVE=-1`)
    then you can vary `SERVING_BATCH_SIZE` and retry from Step 4.

    - You can change the batch size at which the model is converted from JAX
    and saved. Set `SERVING_BATCH_SIZE_SAVE=16` and `SERVING_BATCH_SIZE=16` 
    and redo Step 2 and Step 4 (do not need to restart the model server).
    In Step 4, you should pass a `--count_images`
    parameter that is a multiple of the serving batch size you choose.
