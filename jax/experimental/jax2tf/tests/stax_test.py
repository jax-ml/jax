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

import functools
from absl.testing import absltest
import jax
from jax import test_util as jtu
import numpy as np
import os
import sys

from jax.experimental.jax2tf.tests import tf_test_util

# Import ../../../../examples/resnet50.py
def from_examples_import_resnet50():
  this_dir = os.path.dirname(os.path.abspath(__file__))
  examples_dir = os.path.abspath(os.path.join(this_dir, "..", "..",
                                              "..", "..", "examples"))
  assert os.path.isfile(os.path.join(examples_dir, "resnet50.py"))
  try:
    sys.path.append(examples_dir)
    import resnet50  # type: ignore
    return resnet50
  finally:
    sys.path.pop()

# The next line is rewritten on copybara import.
resnet50 = from_examples_import_resnet50()


from jax.config import config

config.parse_flags_with_absl()


class StaxTest(tf_test_util.JaxToTfTestCase):

  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_res_net(self):
    key = jax.random.PRNGKey(0)
    shape = (224, 224, 3, 1)
    init_fn, apply_fn = resnet50.ResNet50(1000)
    _, params = init_fn(key, shape)
    infer = functools.partial(apply_fn, params)
    images = np.array(jax.random.normal(key, shape))
    self.ConvertAndCompare(infer, images, rtol=0.5)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
