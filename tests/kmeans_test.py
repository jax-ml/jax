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

from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config
from jax.experimental.kmeans import _kmeans
from jax.experimental.kmeans import py_vq
import numpy as np
from scipy import cluster

config.parse_flags_with_absl()


class KmeansTest(jtu.JaxTestCase):

  def test_py_vq(self, atol=1e-5):
    code_book = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    obs = np.array([[0.5, 0.5, 0.5], [0.9, 0.9, 0.9], [1, 1, 1], [2, 2, 2],
                    [2.1, 2.0, 2.1], [2.45, 2.45, 2.75], [3.0, 3.1, 3.0]])
    codes, distances = py_vq(obs, code_book, False)
    expected_codes = np.array([0, 0, 0, 1, 1, 2, 2])
    self.assertArraysEqual(codes, expected_codes)
    for i in range(obs.shape[0]):
      expected_distance = np.linalg.norm(obs[i] - code_book[expected_codes[i]])
      self.assertAllClose(distances[i], expected_distance**2, atol=atol)

  def test_kmeans(self, atol=1e-5):
    code_book_initial = np.array([[0.87190622, 0.38836327, 0.90234106],
                                  [0.63983508, 0.33962291, 0.51053291],
                                  [0.32624428, 0.01307636, 0.65207341]])
    obs = np.array([np.array([float(i), float(i), float(i)]) for i in range(5)])
    # 1st step of kmeans
    print(_kmeans(obs, code_book_initial, False))
    _, code_book, _ = _kmeans(obs, code_book_initial, False)
    expected_code_book = np.array([[2.5, 2.5, 2.5], [0., 0., 0.], [0., 0., 0.]])
    self.assertAllClose(code_book, expected_code_book, atol=atol)
    # 2nd step of kmeans
    _, code_book, _ = _kmeans(obs, code_book, False)
    expected_code_book = np.array([[3., 3., 3.], [0.5, 0.5, 0.5], [0., 0., 0.]])
    self.assertAllClose(code_book, expected_code_book, atol=atol)
    # This is a fixed point of kmeans. Compare it with scipy implementation.
    code_book_sc, _ = cluster.vq.kmeans(obs, k_or_guess=code_book_initial)
    # The scipy implementation drops the zero rows. Hence
    self.assertAllClose(code_book_sc, code_book[:2, :], atol=atol)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
