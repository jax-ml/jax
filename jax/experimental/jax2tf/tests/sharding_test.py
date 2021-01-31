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
"""Tests for the jax2tf conversion of sharded_jit."""

from absl.testing import absltest
from typing import Sequence

import re

import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from jax import test_util as jtu
import numpy as np
from jax.interpreters import sharded_jit
from jax.interpreters.sharded_jit import PartitionSpec as P

from jax.experimental.jax2tf.tests import tf_test_util

import tensorflow as tf  # type: ignore[import]

from jax.config import config
config.parse_flags_with_absl()


class ShardedJitHloTest(tf_test_util.JaxToTfTestCase):
  """Tests that inspect the HLO for the sharding annotations.

  These tests can run on any device.
  """
  def _check_sharding_annotations(self, f_jax, args, expected: Sequence[str]):
    """Check expected patterns in the HLO generated from f_jax and its conversion."""
    jax_hlo = jax.xla_computation(f_jax)(*args).as_hlo_text()
    self.AssertShardingAnnotations("JAX", jax_hlo, expected)

    f_tf = jax2tf.convert(f_jax)
    tf_hlo = tf.function(f_tf, jit_compile=True).\
      experimental_get_compiler_ir(*args)(stage="optimized_hlo")
    self.AssertShardingAnnotations("TF", tf_hlo, expected)


  def AssertShardingAnnotations(self, what: str, hlo: str,
                                expected: Sequence[str]):
    """
    Args:
      what: either 'JAX' or 'TF'
      hlo: the text for the HLO module
      expected: a sequence of regexps that must occur in the hlo text. Each
        regexp must match a line, in order.
    """
    next_expected_idx = 0
    failure_msg = [f"Cannot find some expected sharding annotations in HLO from {what}:"]
    for hlo_line in hlo.split("\n"):
      failure_msg.append(hlo_line)
      if re.search(expected[next_expected_idx], hlo_line):
        failure_msg.append(
            f">>> Found[{next_expected_idx}] {expected[next_expected_idx]}")
        next_expected_idx += 1
        if next_expected_idx >= len(expected):
          break
    else:
      failure_msg.append(
          f"!!! Not found[{next_expected_idx}] {expected[next_expected_idx]}")
      raise self.failureException("\n".join(failure_msg))


  def test_in_out(self):
    """Test input and output sharding annotations."""
    def jax_func(x, y):
      return jnp.dot(x, y)

    sharded_jax_func = sharded_jit.sharded_jit(
        jax_func, in_parts=(P(1, 2), P(2, 1)), out_parts=P(1, 2))
    xshape = (3, 8)
    x = np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (8, 5)
    y = np.arange(np.prod(yshape), dtype=np.float32).reshape(yshape)
    self._check_sharding_annotations(
        sharded_jax_func, [x, y],
        [r'f32\[3,8\].*sharding={devices=\[1,2\]',
         r'f32\[8,5\].*sharding={devices=\[2,1\]',
         r'f32\[3,5\].*dot.*sharding={devices=\[1,2\]'])


  def test_with_sharding_constraint(self):
    """A sharding constraint in the middle."""
    def jax_func(x, y):
      logits1 = jnp.dot(x, y)
      return jnp.sin(sharded_jit.with_sharding_constraint(logits1, P(3, 1)))

    sharded_jax_func = sharded_jit.sharded_jit(
        jax_func, in_parts=(P(1, 2), P(2, 1)), out_parts=P(1, 2))
    xshape = (3, 8)
    x = np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (8, 5)
    y = np.arange(np.prod(yshape), dtype=np.float32).reshape(yshape)
    self._check_sharding_annotations(
        sharded_jax_func, [x, y],
        [r'f32\[3,8\].*sharding={devices=\[1,2\]',
         r'f32\[8,5\].*sharding={devices=\[2,1\]',
         r'f32\[3,5\].*dot.*sharding={devices=\[3,1\]',
         r'f32\[3,5\].*sine.*sharding={devices=\[1,2\]'])

  def test_replicated(self):
    """A replicated input and output."""
    def jax_func(x, y):
      return jnp.dot(x, y)

    sharded_jax_func = sharded_jit.sharded_jit(
        jax_func, in_parts=(P(1, 2), None), out_parts=None)
    xshape = (3, 8)
    x = np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (8, 5)
    y = np.arange(np.prod(yshape), dtype=np.float32).reshape(yshape)
    self._check_sharding_annotations(
        sharded_jax_func, [x, y],
        [r'f32\[3,8\].*sharding={devices=\[1,2\]',
         r'f32\[8,5\].*sharding={replicated}',
         r'f32\[3,5\].*dot.*sharding={replicated}'])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
