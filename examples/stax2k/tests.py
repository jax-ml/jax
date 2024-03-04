# ---
# Copyright 2024 The JAX Authors.
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

from absl.testing import absltest
from jax._src import test_util as jtu

import models

class CoreTest(jtu.JaxTestCase):

  def test_dense(self):
    models.DenseLayer(128, 3, 5).check_types()

  def test_relu(self):
    models.Relu(128, 5).check_types()

  def test_sequential(self):
    models.Sequential([ models.DenseLayer(128, 3, 5)
                      , models.Relu(128, 5)]).check_types()

  def test_loss(self):
    models.WithCategoricalLoss(models.Sequential(
      [ models.DenseLayer(128, 3, 5)
      , models.Relu(128, 5)])).check_types()

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
