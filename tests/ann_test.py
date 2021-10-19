# Copyright 2021 Google LLC
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
from absl.testing import parameterized

from jax import lax
from jax.experimental import ann
from jax._src import test_util as jtu

from jax.config import config

config.parse_flags_with_absl()


class AnnTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_qy={}_db={}_k={}_recall={}".format(
                  jtu.format_shape_dtype_string(qy_shape, dtype),
                  jtu.format_shape_dtype_string(db_shape, dtype), k, recall),
          "qy_shape": qy_shape, "db_shape": db_shape, "dtype": dtype,
          "k": k, "recall": recall }
        for qy_shape in [(200, 128), (128, 128)]
        for db_shape in [(128, 500), (128, 3000)]
        for dtype in jtu.dtypes.all_floating
        for k in [10, 50] for recall in [0.9, 0.95]))
  def test_approx_max_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    scores = lax.dot(qy, db)
    _, gt_args = lax.top_k(scores, k)
    _, ann_args = ann.approx_max_k(scores, k, recall_target=recall)
    gt_args_sets = [set(x) for x in gt_args]
    hits = sum(
        len(list(x
                 for x in ann_args_per_q
                 if x in gt_args_sets[q]))
        for q, ann_args_per_q in enumerate(ann_args))
    self.assertGreater(hits / (qy_shape[0] * k), recall)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_qy={}_db={}_k={}_recall={}".format(
                  jtu.format_shape_dtype_string(qy_shape, dtype),
                  jtu.format_shape_dtype_string(db_shape, dtype), k, recall),
          "qy_shape": qy_shape, "db_shape": db_shape, "dtype": dtype,
          "k": k, "recall": recall }
        for qy_shape in [(200, 128), (128, 128)]
        for db_shape in [(128, 500), (128, 3000)]
        for dtype in jtu.dtypes.all_floating
        for k in [10, 50] for recall in [0.9, 0.95]))
  def test_approx_min_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    scores = lax.dot(qy, db)
    _, gt_args = lax.top_k(-scores, k)
    _, ann_args = ann.approx_min_k(scores, k, recall_target=recall)
    gt_args_sets = [set(x) for x in gt_args]
    hits = sum(
        len(list(x
                 for x in ann_args_per_q
                 if x in gt_args_sets[q]))
        for q, ann_args_per_q in enumerate(ann_args))
    self.assertGreater(hits / (qy_shape[0] * k), recall)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
