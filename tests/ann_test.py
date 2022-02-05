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

from functools import partial

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import jax
from jax import lax
from jax.experimental import ann
from jax._src import test_util as jtu
from jax._src.util import prod

from jax.config import config

config.parse_flags_with_absl()

ignore_jit_of_pmap_warning = partial(
    jtu.ignore_warning,message=".*jit-of-pmap.*")

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
        for k in [1, 10, 50] for recall in [0.9, 0.95]))
  def test_approx_max_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    scores = lax.dot(qy, db)
    _, gt_args = lax.top_k(scores, k)
    _, ann_args = ann.approx_max_k(scores, k, recall_target=recall)
    self.assertEqual(k, len(ann_args[0]))
    ann_recall = ann.ann_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreater(ann_recall, recall)

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
        for k in [1, 10, 50] for recall in [0.9, 0.95]))
  def test_approx_min_k(self, qy_shape, db_shape, dtype, k, recall):
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    scores = lax.dot(qy, db)
    _, gt_args = lax.top_k(-scores, k)
    _, ann_args = ann.approx_min_k(scores, k, recall_target=recall)
    self.assertEqual(k, len(ann_args[0]))
    ann_recall = ann.ann_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreater(ann_recall, recall)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_shape={}_k={}_max_k={}".format(
                  jtu.format_shape_dtype_string(shape, dtype), k, is_max_k),
          "shape": shape, "dtype": dtype, "k": k, "is_max_k": is_max_k }
        for dtype in [np.float32]
        for shape in [(4,), (5, 5), (2, 1, 4)]
        for k in [1, 3]
        for is_max_k in [True, False]))
  def test_autodiff(self, shape, dtype, k, is_max_k):
    vals = np.arange(prod(shape), dtype=dtype)
    vals = self.rng().permutation(vals).reshape(shape)
    if is_max_k:
      fn = lambda vs: ann.approx_max_k(vs, k=k)[0]
    else:
      fn = lambda vs: ann.approx_min_k(vs, k=k)[0]
    jtu.check_grads(fn, (vals,), 2, ["fwd", "rev"], eps=1e-2)


  @parameterized.named_parameters(
      jtu.cases_from_list({
          "testcase_name":
              "_qy={}_db={}_k={}_recall={}".format(
                  jtu.format_shape_dtype_string(qy_shape, dtype),
                  jtu.format_shape_dtype_string(db_shape, dtype), k, recall),
          "qy_shape": qy_shape, "db_shape": db_shape, "dtype": dtype,
          "k": k, "recall": recall }
        for qy_shape in [(200, 128), (128, 128)]
        for db_shape in [(2048, 128)]
        for dtype in jtu.dtypes.all_floating
        for k in [1, 10] for recall in [0.9, 0.95]))
  def test_pmap(self, qy_shape, db_shape, dtype, k, recall):
    num_devices = jax.device_count()
    rng = jtu.rand_default(self.rng())
    qy = rng(qy_shape, dtype)
    db = rng(db_shape, dtype)
    db_size = db.shape[0]
    gt_scores = lax.dot_general(qy, db, (([1], [1]), ([], [])))
    _, gt_args = lax.top_k(-gt_scores, k)  # negate the score to get min-k
    db_per_device = db_size//num_devices
    sharded_db = db.reshape(num_devices, db_per_device, 128)
    db_offsets = np.arange(num_devices, dtype=np.int32) * db_per_device
    def parallel_topk(qy, db, db_offset):
      scores = lax.dot_general(qy, db, (([1],[1]),([],[])))
      ann_vals, ann_args = ann.approx_min_k(scores, k=k, reduction_dimension=1,
                                            recall_target=recall,
                                            reduction_input_size_override=db_size,
                                            aggregate_to_topk=False)
      return (ann_vals, ann_args + db_offset)
    # shape = qy_size, num_devices, approx_dp
    ann_vals, ann_args = jax.pmap(
        parallel_topk,
        in_axes=(None, 0, 0),
        out_axes=(1, 1))(qy, sharded_db, db_offsets)
    # collapse num_devices and approx_dp
    ann_vals = lax.collapse(ann_vals, 1, 3)
    ann_args = lax.collapse(ann_args, 1, 3)
    ann_vals, ann_args = lax.sort_key_val(ann_vals, ann_args, dimension=1)
    ann_args = lax.slice_in_dim(ann_args, start_index=0, limit_index=k, axis=1)
    ann_recall = ann.ann_recall(np.asarray(ann_args), np.asarray(gt_args))
    self.assertGreater(ann_recall, recall)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
