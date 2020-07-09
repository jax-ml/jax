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

from functools import partial
import itertools
import unittest

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from jax import image
from jax import test_util as jtu

from jax.config import config
from jax.config import flags

# We use TensorFlow and PIL as reference implementations.
try:
  import tensorflow as tf
except ImportError:
  tf = None

try:
  from PIL import Image as PIL_Image
except ImportError:
  PIL_Image = None

config.parse_flags_with_absl()

FLAGS = flags.FLAGS

float_dtypes = jtu.dtypes.all_floating
inexact_dtypes = jtu.dtypes.inexact

class ImageTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
       {"testcase_name": "_shape={}_target={}_method={}_antialias={}".format(
          jtu.format_shape_dtype_string(image_shape, dtype),
          jtu.format_shape_dtype_string(target_shape, dtype), method,
          antialias),
        "dtype": dtype, "image_shape": image_shape,
        "target_shape": target_shape,
        "method": method, "antialias": antialias}
       for dtype in [np.float32]
       for target_shape, image_shape in itertools.combinations_with_replacement(
        [[2, 3, 2, 4], [2, 6, 4, 4], [2, 33, 17, 4], [2, 50, 38, 4]], 2)
       for method in ["bilinear", "lanczos3", "lanczos5", "bicubic"]
       for antialias in [False, True])) 
  @unittest.skipIf(not tf, "Test requires TensorFlow")
  def testResizeAgainstTensorFlow(self, dtype, image_shape, target_shape, method,
                                  antialias):
    # TODO(phawkins): debug this. There is a small mismatch between TF and JAX
    # for some cases of non-antialiased bicubic downscaling; we would expect
    # exact equality.
    if method == "bicubic" and any(x < y for x, y in
                                   zip(target_shape, image_shape)):
      raise unittest.SkipTest("non-antialiased bicubic downscaling mismatch")
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: (rng(image_shape, dtype),)
    def tf_fn(x):
      out = tf.image.resize(x, tf.constant(target_shape[1:-1]), method=method,
                             antialias=antialias).numpy()
      print(out.dtype)
      return out
    jax_fn = partial(image.resize, shape=target_shape, method=method,
                     antialias=antialias)
    self._CheckAgainstNumpy(tf_fn, jax_fn, args_maker, check_dtypes=True,
                            tol=1e-4)


  @parameterized.named_parameters(jtu.cases_from_list(
       {"testcase_name": "_shape={}_target={}_method={}".format(
          jtu.format_shape_dtype_string(image_shape, dtype),
          jtu.format_shape_dtype_string(target_shape, dtype), method),
        "dtype": dtype, "image_shape": image_shape,
        "target_shape": target_shape,
        "method": method}
       for dtype in [np.float32]
       for target_shape, image_shape in itertools.combinations_with_replacement(
        [[3, 2], [6, 4], [33, 17], [50, 38]], 2)
       for method in ["bilinear", "lanczos3", "bicubic"]))
  @unittest.skipIf(not PIL_Image, "Test requires PIL")
  def testResizeAgainstPIL(self, dtype, image_shape, target_shape, method):
    rng = jtu.rand_uniform(self.rng())
    args_maker = lambda: (rng(image_shape, dtype),)
    def pil_fn(x):
      pil_methods = {
        "bilinear": PIL_Image.BILINEAR,
        "bicubic": PIL_Image.BICUBIC,
        "lanczos3": PIL_Image.LANCZOS,
      }
      img = PIL_Image.fromarray(x)
      print(img, x.shape)
      out = np.asarray(img.resize(target_shape[::-1], pil_methods[method]))
      print(out.shape, target_shape)
      return out
    jax_fn = partial(image.resize, shape=target_shape, method=method,
                     antialias=True)
    self._CheckAgainstNumpy(pil_fn, jax_fn, args_maker, check_dtypes=True,
                            tol=1e-4)

  @parameterized.named_parameters(jtu.cases_from_list(
       {"testcase_name": "_shape={}_target={}_method={}".format(
          jtu.format_shape_dtype_string(image_shape, dtype),
          jtu.format_shape_dtype_string(target_shape, dtype), method),
        "dtype": dtype, "image_shape": image_shape, "target_shape": target_shape,
        "method": method}
       for dtype in inexact_dtypes
       for image_shape, target_shape in [
         ([3, 1, 2], [6, 1, 4]),
         ([1, 3, 2, 1], [1, 6, 4, 1]),
       ]
       for method in ["linear", "lanczos3", "lanczos5", "cubic"]))
  def testResizeUp(self, dtype, image_shape, target_shape, method):
    data = [64, 32, 32, 64, 50, 100]
    expected_data = {}
    expected_data["linear"] = [
        64.0, 56.0, 40.0, 32.0, 56.0, 52.0, 44.0, 40.0, 40.0, 44.0, 52.0, 56.0,
        36.5, 45.625, 63.875, 73.0, 45.5, 56.875, 79.625, 91.0, 50.0, 62.5,
        87.5, 100.0
    ]
    expected_data["lanczos3"] = [
        75.8294, 59.6281, 38.4313, 22.23, 60.6851, 52.0037, 40.6454, 31.964,
        35.8344, 41.0779, 47.9383, 53.1818, 24.6968, 43.0769, 67.1244, 85.5045,
        35.7939, 56.4713, 83.5243, 104.2017, 44.8138, 65.1949, 91.8603, 112.2413
    ]
    expected_data["lanczos5"] = [
        77.5699, 60.0223, 40.6694, 23.1219, 61.8253, 51.2369, 39.5593, 28.9709,
        35.7438, 40.8875, 46.5604, 51.7041, 21.5942, 43.5299, 67.7223, 89.658,
        32.1213, 56.784, 83.984, 108.6467, 44.5802, 66.183, 90.0082, 111.6109
    ]
    expected_data["cubic"] = [
        70.1453, 59.0252, 36.9748, 25.8547, 59.3195, 53.3386, 41.4789, 35.4981,
        36.383, 41.285, 51.0051, 55.9071, 30.2232, 42.151, 65.8032, 77.731,
        41.6492, 55.823, 83.9288, 98.1026, 47.0363, 62.2744, 92.4903, 107.7284
    ]
    x = np.array(data, dtype=dtype).reshape(image_shape)
    output = image.resize(x, target_shape, method)
    expected = np.array(expected_data[method], dtype=dtype).reshape(target_shape)
    self.assertAllClose(output, expected, atol=1e-04)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
