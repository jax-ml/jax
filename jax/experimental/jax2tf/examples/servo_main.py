# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demonstrates use of a jax2tf model in TensorFlow model server.

Includes the flags from saved_model_main.py.

If you want to start your own model server, you should pass the
`--nostart_model_server` flag and also `--serving_url` to point to the
HTTP REST API end point of your model server. You can see the path of the
trained and saved model in the output.

See README.md.
"""
import atexit
import logging
import subprocess
import threading
import time
from absl import app
from absl import flags

from jax.experimental.jax2tf.examples import mnist_lib
from jax.experimental.jax2tf.examples import saved_model_main

import numpy as np
import requests

import tensorflow as tf  # type: ignore
import tensorflow_datasets as tfds  # type: ignore


flags.DEFINE_integer("count_images", 10, "How many images to test")
flags.DEFINE_bool("start_model_server", True,
                  "Whether to start/stop the model server.")
flags.DEFINE_string("serving_url", "http://localhost:8501/v1/models/jax_model",
                    "The HTTP endpoint for the model server")

FLAGS = flags.FLAGS


def mnist_predict_request(serving_url: str, images):
  """Predicts using the model server.

  Args:
    serving_url: The URL for the model server.
    images: A batch of images of shape F32[B, 28, 28, 1]

  Returns:
    a batch of one-hot predictions, of shape F32[B, 10]
  """
  request = {"inputs": images.tolist()}
  response = requests.post(f"{serving_url}:predict", json=request)
  response_json = response.json()
  if response.status_code != 200:
    raise RuntimeError("Model server error: " + response_json["error"])

  predictions = np.array(response_json["outputs"])
  return predictions


def main(_):
  if FLAGS.count_images % FLAGS.serving_batch_size != 0:
    raise ValueError("count_images must be a multiple of serving_batch_size")

  saved_model_main.train_and_save()
  # Strip the version number from the model directory
  servo_model_dir = saved_model_main.savedmodel_dir(with_version=False)

  if FLAGS.start_model_server:
    model_server_proc = _start_localhost_model_server(servo_model_dir)

  try:
    _mnist_sanity_check(FLAGS.serving_url, FLAGS.serving_batch_size)
    test_ds = mnist_lib.load_mnist(
        tfds.Split.TEST, batch_size=FLAGS.serving_batch_size)
    images_and_labels = tfds.as_numpy(
        test_ds.take(FLAGS.count_images // FLAGS.serving_batch_size))

    for (images, labels) in images_and_labels:
      predictions_one_hot = mnist_predict_request(FLAGS.serving_url, images)
      predictions_digit = np.argmax(predictions_one_hot, axis=1)
      label_digit = np.argmax(labels, axis=1)
      logging.info(
          f" predicted = {predictions_digit} labelled digit {label_digit}")
  finally:
    if FLAGS.start_model_server:
      model_server_proc.kill()
      model_server_proc.communicate()


def _mnist_sanity_check(serving_url: str, serving_batch_size: int):
  """Checks that we can reach a model server with a model that matches MNIST."""
  logging.info("Checking that model server serves a compatible model.")
  response = requests.get(f"{serving_url}/metadata")
  response_json = response.json()
  if response.status_code != 200:
    raise IOError("Model server error: " + response_json["error"])

  try:
    signature_def = response_json["metadata"]["signature_def"]["signature_def"]
    serving_default = signature_def[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    assert serving_default["method_name"] == "tensorflow/serving/predict"
    inputs, = serving_default["inputs"].values()
    b, w, h, c = [int(d["size"]) for d in inputs["tensor_shape"]["dim"]]
    assert b == -1 or b == serving_batch_size, (
        f"Found input batch size {b}. Expecting {serving_batch_size}")
    assert (w, h, c) == mnist_lib.input_shape
  except Exception as e:
    raise IOError(
        f"Unexpected response from model server: {response_json}") from e


def _start_localhost_model_server(model_dir):
  """Starts the model server on localhost, using docker.

  Ignore this if you have a different way to start the model server.
  """
  cmd = ("docker run -p 8501:8501 --mount "
         f"type=bind,source={model_dir}/,target=/models/jax_model "
         "-e MODEL_NAME=jax_model -t --rm --name=serving tensorflow/serving")
  cmd_args = cmd.split(" ")
  logging.info("Starting model server")
  logging.info(f"Running {cmd}")
  proc = subprocess.Popen(
      cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  model_server_ready = False

  def model_server_output_reader():
    for line in iter(proc.stdout.readline, b""):
      line_str = line.decode("utf-8").strip()
      if "Exporting HTTP/REST API at:localhost:8501" in line_str:
        nonlocal model_server_ready
        model_server_ready = True
      logging.info(f"Model server: {line_str}")

  output_thread = threading.Thread(target=model_server_output_reader, args=())
  output_thread.start()

  def _stop_model_server():
    logging.info("Stopping the model server")
    subprocess.run("docker container stop serving".split(" "),
                   check=True)

  atexit.register(_stop_model_server)

  wait_iteration_sec = 2
  wait_remaining_sec = 10
  while not model_server_ready and wait_remaining_sec > 0:
    logging.info("Waiting for the model server to be ready...")
    time.sleep(wait_iteration_sec)
    wait_remaining_sec -= wait_iteration_sec

  if wait_remaining_sec <= 0:
    raise IOError("Model server failed to start properly")
  return proc


if __name__ == "__main__":
  app.run(main)
