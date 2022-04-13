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
"""Tests for the jax2tf conversion of pjit."""

import functools
import logging
import re
from typing import Any, Sequence
import unittest

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
from jax.config import config

from jax.experimental import jax2tf
from jax.experimental import pjit
from jax.experimental.jax2tf.tests import tf_test_util
from jax.interpreters.pxla import PartitionSpec as P
import jax.numpy as jnp
import jax._src.lib.xla_bridge

import numpy as np

import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

def setUpModule():
  jtu.set_spmd_lowering_flag(True)

def tearDownModule():
  jtu.restore_spmd_lowering_flag()


LOG_HLO = True

class ShardedJitHloTest(tf_test_util.JaxToTfTestCase):
  """Tests that inspect the HLO for the sharding annotations.

  These tests can run on any device.
  """

  def _check_sharding_annotations(self,
                                  f_jax,
                                  args: Sequence[Any],
                                  *,
                                  expected: Sequence[str],
                                  expected_opt: Sequence[str],
                                  num_partitions=2):
    """Check expected patterns in the HLO generated from f_jax and its conversion.

    We run this check on CPU also, which is useful for debugging locally.
    We currently check the unoptimized HLO against `expected` on CPU and TPU,
    and we check the optimized HLO against `expected_opt` on TPU only and
    only for JAX.

    See `self.AssertShardingAnnotations` for documentation of `expected`
    and `expected_opt`.
    """
    if jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("Sharding HLO tests not useful for GPU")

    jax_comp = f_jax.lower(*args).compiler_ir(dialect="hlo")
    jax_hlo = jax_comp.as_hlo_text()
    if LOG_HLO:
      logging.info("[%s] got JAX HLO %s", self._testMethodName, jax_hlo)
    self.AssertShardingAnnotations("JAX before optimizations", jax_hlo, expected)

    if jtu.device_under_test() == "tpu":
      backend = jax._src.lib.xla_bridge.get_backend()
      num_replicas = 1
      device_assignment = np.arange(num_partitions * num_replicas)
      device_assignment = np.reshape(device_assignment, (-1, num_partitions))
      use_spmd_partitioning = num_partitions > 1
      compile_options = jax._src.lib.xla_bridge.get_compile_options(
          num_replicas=num_replicas,
          num_partitions=num_partitions,
          device_assignment=device_assignment,
          use_spmd_partitioning=use_spmd_partitioning,
      )
      jax_optimized_hlo = backend.compile(
          jax_comp, compile_options).hlo_modules()[0].to_string()
      if LOG_HLO:
        logging.info("[%s] got JAX optimized HLO for platform %s %s",
                     self._testMethodName, backend.platform, jax_optimized_hlo)
      self.AssertShardingAnnotations("JAX after optimizations",
                                     jax_optimized_hlo, expected_opt)

    f_tf = jax2tf.convert(f_jax)
    device_name = f"/device:{jtu.device_under_test().upper()}:0"
    tf_hlo = (tf.function(f_tf, jit_compile=True, autograph=False)
              .experimental_get_compiler_ir(*args)(stage="hlo",
                                                   device_name=device_name))
    if LOG_HLO:
      logging.info("[%s] got TF HLO %s", self._testMethodName, tf_hlo)
    self.AssertShardingAnnotations("TF before optimizations", tf_hlo, expected)
    tf_optimized_hlo = (
        tf.function(f_tf, jit_compile=True)
        .experimental_get_compiler_ir(*args)(stage="optimized_hlo",
                                             device_name=device_name))
    if LOG_HLO:
      logging.info("[%s] got TF optimized HLO for %s: %s", self._testMethodName,
                   device_name, tf_optimized_hlo)

  def AssertShardingAnnotations(self, what: str, hlo: str,
                                expected: Sequence[str]):
    """Args:

      what: either 'JAX' or 'TF', used for messages only.
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

  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic1D(self):

    @functools.partial(pjit.pjit,
                       in_axis_resources=(P("x"), P("x")),
                       out_axis_resources=None)
    def jax_func(x, y):
      return x + y

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hlo = jax_func.lower(x, x).compiler_ir(dialect="hlo").as_hlo_text()
    print(f"HLO is {hlo}")
    print(f"JAXPR is {jax.make_jaxpr(jax_func)(x, x)}")
    self._check_sharding_annotations(
        jax_func, [x, x],
        expected=[
            r"f32\[8,10\].*sharding={devices=\[2,1\]",  # x and y
            r"f32\[8,10\].*sharding={replicated",  # output
        ],
        expected_opt=[
            r"f32\[4,10\].*sharding={devices=\[2,1\]",  # x and y
            # TODO: why don't we see "sharding={replicated"
            r"f32\[8,10\]",  # output
        ],
        num_partitions=2)

  @jtu.with_mesh([("x", 2), ("y", 2)])
  def test_pjit_basic2D(self):
    @functools.partial(pjit.pjit,
                       in_axis_resources=(P(None, "x", "y"), P("y")),
                       out_axis_resources=P("x"))
    def jax_func(x, y):
      return x @ y

    x_shape = (8, 6, 4)
    y_shape = (4, 2)
    x = jnp.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape), dtype=np.float32).reshape(y_shape)
    self._check_sharding_annotations(
        jax_func,
        [x, y],
        expected=[
            r"f32\[8,6,4\].*sharding={devices=\[1,2,2\]0,1,2,3",  # x
            r"f32\[4,2\].*sharding={devices=\[2,1,2\]0,2,1,3 last_tile_dim_replicate",  # y
            r"f32\[8,6,2\].*sharding={devices=\[2,1,1,2\]0,1,2,3 last_tile_dim_replicate",  # output
        ],
        expected_opt=[
            # TODO: relax ordering
            r"f32\[2,2\].*sharding={devices=\[2,1,2\]0,2,1,3 last_tile_dim_replicate|f32\[8,3,2\].*sharding={devices=\[1,2,2\]0,1,2,3",
            r"f32\[2,2\].*sharding={devices=\[2,1,2\]0,2,1,3 last_tile_dim_replicate|f32\[8,3,2\].*sharding={devices=\[1,2,2\]0,1,2,3",
            # TODO: why we cannot see sharding={devices=\[2,1,1,2\]0,1,2,3 last_tile_dim_replicate?
            r"bf16\[4,6,2\]",  # output
        ],
        num_partitions=4)

  @jtu.with_mesh([("x", 2), ("y", 2)])
  def test_pjit_TwoMeshAxisSharding(self):
    @functools.partial(pjit.pjit,
             in_axis_resources=P(("x", "y"),),
             out_axis_resources=P(("x", "y"),))
    def jax_func(x, y):
      return x @ y

    x_shape = (24, 8)
    y_shape = (8, 2)
    x = jnp.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape), dtype=np.float32).reshape(y_shape)
    self._check_sharding_annotations(
        jax_func,
        [x, y],
        expected=[
            r"f32\[24,8\].*sharding={devices=\[4,1\]0,1,2,3",  # x
            r"f32\[8,2\].*sharding={devices=\[4,1\]0,1,2,3",  # y
            r"f32\[24,2\].*sharding={devices=\[4,1\]0,1,2,3",  # output
        ],
        expected_opt=[
            # TODO: relax ordering
            r"f32\[2,2\].*sharding={devices=\[4,1\]0,1,2,3|f32\[6,8\].*sharding={devices=\[4,1\]0,1,2,3",
            r"f32\[2,2\].*sharding={devices=\[4,1\]0,1,2,3|f32\[6,8\].*sharding={devices=\[4,1\]0,1,2,3",
            # TODO: why we cannot see .*sharding={devices=\[4,1\]0,1,2,3
            r"f32\[6,2\]",  # output
        ],
        num_partitions=4)

  @jtu.with_mesh([("x", 2), ("y", 1)])
  def test_pjit_ShardingConstraint(self):
    @functools.partial(pjit.pjit, in_axis_resources=None,
                       out_axis_resources=None)
    def jax_func(x):  # x: f32[12, 8]
      y = jnp.tile(x, (2, 1))  # y: f32[24, 8]
      y = pjit.with_sharding_constraint(y, P("x", "y"))
      return y[0:y.shape[0] // 4]  # res: f32[6, 8]

    shape = (12, 8)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    self._check_sharding_annotations(
        jax_func, [x],
        expected=[
            r"f32\[12,8\].*sharding={replicated}",  # x
            r"f32\[24,8\].*sharding={devices=\[2,1\]0,1",  # y
            r"f32\[6,8\].*sharding={replicated}",  # output
        ],
        expected_opt=[
            r"f32\[12,8\].*sharding={replicated}",  # x
            # TODO: why can't we see "sharding={devices=\[2,1\]0,1"
            r"f32\[12,8\]",  # y
            # TODO: why can't we see "sharding={replicated}" ?
            r"f32\[6,8\]",  # output
        ],
        num_partitions=2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
