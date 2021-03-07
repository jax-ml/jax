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

import logging
import re
from typing import Sequence
import unittest

from absl.testing import absltest

import jax
from jax import test_util as jtu
from jax.config import config

from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import tf_test_util
from jax.interpreters import sharded_jit
from jax.interpreters.sharded_jit import PartitionSpec as P
import jax.numpy as jnp

import numpy as np

import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

LOG_HLO = True

class ShardedJitHloTest(tf_test_util.JaxToTfTestCase):
  """Tests that inspect the HLO for the sharding annotations.

  These tests can run on any device.
  """

  def _check_sharding_annotations(self,
                                  f_jax,
                                  args,
                                  *,
                                  expected: Sequence[str],
                                  expected_opt: Sequence[str],
                                  num_partitions=2):
    """Check expected patterns in the HLO generated from f_jax and its conversion.

    We run this check on CPU also, which is useful for debugging locally.
    We currently check the unoptimized HLO against `expected` on CPU and TPU,
    and we check the optimized HLO against `expected_opt` on TPU only and
    only for JAX.
    """
    if jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("Sharding HLO tests not useful for GPU")

    jax_comp = jax.xla_computation(f_jax)(*args)
    jax_hlo = jax_comp.as_hlo_text()
    if LOG_HLO:
      logging.info(f"[{self._testMethodName}] got JAX HLO {jax_hlo}")
    self.AssertShardingAnnotations("JAX before optimizations", jax_hlo, expected)

    if jtu.device_under_test() == "tpu":
      backend = jax.lib.xla_bridge.get_backend()
      num_replicas = 1
      device_assignment = np.arange(num_partitions * num_replicas)
      device_assignment = np.reshape(device_assignment, (-1, num_partitions))
      use_spmd_partitioning = num_partitions > 1
      compile_options = jax.lib.xla_bridge.get_compile_options(
          num_replicas=num_replicas,
          num_partitions=num_partitions,
          device_assignment=device_assignment,
          use_spmd_partitioning=use_spmd_partitioning,
      )
      jax_optimized_hlo = backend.compile(
          jax_comp, compile_options).hlo_modules()[0].to_string()
      if LOG_HLO:
        logging.info(f"[{self._testMethodName}] got JAX optimized HLO for "
                     f"platform {backend.platform} {jax_optimized_hlo}")
      self.AssertShardingAnnotations("JAX after optimizations",
                                     jax_optimized_hlo, expected_opt)

    f_tf = jax2tf.convert(f_jax)
    device_name = f"/device:{jtu.device_under_test().upper()}:0"
    tf_hlo = tf.function(f_tf, jit_compile=True, autograph=False).\
      experimental_get_compiler_ir(*args)(stage="hlo",
                                          device_name=device_name)
    if LOG_HLO:
      logging.info(f"[{self._testMethodName}] got TF OPT HLO {tf_hlo}")
    self.AssertShardingAnnotations("TF before optimizations", tf_hlo, expected)
    tf_optimized_hlo = tf.function(f_tf, jit_compile=True).\
      experimental_get_compiler_ir(*args)(stage="optimized_hlo",
                                          device_name=device_name)
    if LOG_HLO:
      logging.info(f"[{self._testMethodName}] XXX got TF OPT HLO "
                   f"for {device_name}: {tf_optimized_hlo}")

  def AssertShardingAnnotations(self, what: str, hlo: str,
                                expected: Sequence[str]):
    """Args:

      what: either 'JAX' or 'TF'
      hlo: the text for the HLO module
      expected: a sequence of regexps that must occur in the hlo text. Each
      regexp must match a line, in order.
    """
    next_expected_idx = 0
    failure_msg = [
        f"Cannot find some expected sharding annotations in HLO from {what}:"
    ]
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
    sharded_jax_func = sharded_jit.sharded_jit(
        jnp.dot, in_parts=(P(1, 2), P(2, 1)), out_parts=P(1, 2))
    xshape = (3, 8)
    x = np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (8, 5)
    y = np.arange(np.prod(yshape), dtype=np.float32).reshape(yshape)
    self._check_sharding_annotations(
        sharded_jax_func, [x, y],
        expected=[
            r"f32\[3,8\].*sharding={devices=\[1,2\]",
            r"f32\[8,5\].*sharding={devices=\[2,1\]",
            r"f32\[3,5\].*sharding={devices=\[1,2\]"
        ],
        expected_opt=[
            # TODO(necula): relax ordering
            r"f32\[4,5\].*sharding={devices=\[2,1\]",
            r"f32\[3,4\].*sharding={devices=\[1,2\]",
            r"f32\[3,5\].*convolution",
            r"f32\[3,5\].*all-reduce",
        ],
        num_partitions=2)

  def test_with_sharding_constraint(self):
    """A sharding constraint in the middle."""

    def jax_func(x, y):
      logits1 = jnp.dot(x, y)
      return jnp.sin(sharded_jit.with_sharding_constraint(logits1, P(2, 1)))

    sharded_jax_func = sharded_jit.sharded_jit(
        jax_func, in_parts=(P(1, 2), P(2, 1)), out_parts=P(1, 2))
    xshape = (6, 8)
    x = np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (8, 10)
    y = np.arange(np.prod(yshape), dtype=np.float32).reshape(yshape)
    self._check_sharding_annotations(
        sharded_jax_func, [x, y],
        expected=[
            r"f32\[6,8\].*sharding={devices=\[1,2\]",
            r"f32\[8,10\].*sharding={devices=\[2,1\]",
            r"f32\[6,10\].*sharding={devices=\[2,1\]",
            r"f32\[6,10\].*sine.*sharding={devices=\[1,2\]"
        ],
        expected_opt=[
            # TODO(necula): relax ordering
            r"f32\[4,10\].*sharding={devices=\[2,1\]",
            r"f32\[6,4\].*sharding={devices=\[1,2\]",
        ],
        num_partitions=2)

  def test_replicated(self):
    """A replicated input and output."""

    sharded_jax_func = sharded_jit.sharded_jit(
        jnp.dot, in_parts=(P(1, 2), None), out_parts=None)
    xshape = (3, 8)
    x = np.arange(np.prod(xshape), dtype=np.float32).reshape(xshape)
    yshape = (8, 5)
    y = np.arange(np.prod(yshape), dtype=np.float32).reshape(yshape)
    self._check_sharding_annotations(
        sharded_jax_func, [x, y],
        expected=[
            r"f32\[3,8\].*sharding={devices=\[1,2\]",
            r"f32\[8,5\].*sharding={replicated}",
            r"f32\[3,5\].*sharding={replicated}"
        ],
        expected_opt=[
            # TODO(necula): relax ordering
            r"f32\[8,5\].*sharding={replicated}",
            r"f32\[3,4\].*sharding={devices=\[1,2\]",
        ],
        num_partitions=2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
