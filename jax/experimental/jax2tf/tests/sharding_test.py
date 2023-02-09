# Copyright 2021 The JAX Authors.
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

from functools import partial
import logging
import os
import re
from typing import Any, Sequence
import unittest

from absl.testing import absltest

import jax
from jax._src import test_util as jtu
from jax.config import config
from jax import lax
from jax.experimental import jax2tf
from jax.experimental import pjit
from jax.experimental.maps import xmap
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from jax._src.lib import xla_bridge

import numpy as np

import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

# Must come after initializing the flags
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation

skip_eager_for_partitioning = Jax2TfLimitation(
    "pjit functions with partitioning must be under tf.function",
    modes="eager", skip_tf_run=True)

prev_xla_flags = None
def setUpModule():
  global prev_xla_flags
  prev_xla_flags = os.getenv("XLA_FLAGS")
  flags_str = prev_xla_flags or ""
  # Don't override user-specified device count, or other XLA flags.
  if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (flags_str +
                               " --xla_force_host_platform_device_count=8")
  # Clear any cached backends so new CPU backend will pick up the env var.
  xla_bridge.get_backend.cache_clear()
  jtu.set_spmd_lowering_flag(True)


def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()
  jtu.restore_spmd_lowering_flag()


def _log_sharding_annotations(test,
                              f_jax,
                              args: Sequence[Any],
                              *,
                              num_partitions=2,
                              num_variables=0,
                              experimental_native_lowering="default"):
  """Log the HLO generated from f_jax and its conversion.

  Ideally this would check the sharding of intermediate results in JAX and
  TF, but this has turned out to be very brittle and broke down for
  StableHLO lowering (the sharding annotation are now binary-encoded
  attributes). We kept the logging aspect of this function, which should
  help some debugging.
  """
  if jtu.device_under_test() == "gpu":
    raise unittest.SkipTest("Sharding HLO tests not useful for GPU")

  jax_comp = f_jax.lower(*args).compiler_ir(dialect="mhlo")
  jax_hlo = str(jax_comp)
  logging.info("[%s] got JAX HLO %s", test._testMethodName, jax_hlo)

  # We only dump JAX optimized code on the TPU
  if jtu.device_under_test() == "tpu":
    backend = xla_bridge.get_backend()
    num_replicas = 1
    device_assignment = np.arange(num_partitions * num_replicas)
    device_assignment = np.reshape(device_assignment, (-1, num_partitions))
    use_spmd_partitioning = num_partitions > 1
    compile_options = xla_bridge.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=use_spmd_partitioning,
    )
    jax_optimized_hlo = backend.compile(
        jax_hlo, compile_options).hlo_modules()[0].to_string()
    logging.info("[%s] got JAX optimized HLO for platform %s %s",
                 test._testMethodName, backend.platform, jax_optimized_hlo)

  f_tf_base = jax2tf.convert(f_jax, with_gradient=False,
                              experimental_native_lowering=experimental_native_lowering)
  if num_variables > 0:
    args_vars = [tf.Variable(a) for a in args[:num_variables]]
    args = args[:num_variables]
    f_tf = lambda *inputs: f_tf_base(*args_vars, *inputs)
  else:
    f_tf = f_tf_base
  f_tf_fun = tf.function(f_tf, jit_compile=True, autograph=False)
  logging.info("[%s] Got TF graph %s",
                test._testMethodName,
                f_tf_fun.get_concrete_function(*args).graph.as_graph_def())
  device_name = f"/device:{jtu.device_under_test().upper()}:0"
  tf_hlo = (f_tf_fun
            .experimental_get_compiler_ir(*args)(stage="hlo",
                                                  device_name=device_name))
  logging.info("[%s] got TF HLO %s", test._testMethodName, tf_hlo)
  tf_optimized_hlo = (
      tf.function(f_tf, jit_compile=True, autograph=False)
      .experimental_get_compiler_ir(*args)(stage="optimized_hlo",
                                            device_name=device_name))
  logging.info("[%s] got TF optimized HLO for %s: %s", test._testMethodName,
                device_name, tf_optimized_hlo)


class ShardedJitHloTest(tf_test_util.JaxToTfTestCase):
  """Tests that inspect the HLO for the sharding annotations.

  These tests can run on any device.
  """


  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic1D(self):

    @partial(pjit.pjit,
                       in_axis_resources=(P("x"), P("x")),
                       out_axis_resources=None)
    def jax_func(x, y):
      return x + y

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hlo = jax_func.lower(x, x).compiler_ir(dialect="hlo").as_hlo_text()
    logging.info("HLO is %s", hlo)
    logging.info("JAXPR is %s", jax.make_jaxpr(jax_func)(x, x))
    _log_sharding_annotations(self,
        jax_func, [x, x],
        num_partitions=2)

  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic1D_variable(self):
    # The first argument is a tf.Variable
    @partial(pjit.pjit,
                       in_axis_resources=(P("x"), P("x")),
                       out_axis_resources=None)
    def jax_func(x, y):
      return x + y

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hlo = jax_func.lower(x, x).compiler_ir(dialect="hlo").as_hlo_text()
    logging.info("HLO is %s", hlo)
    logging.info("JAXPR is %s", jax.make_jaxpr(jax_func)(x, x))
    _log_sharding_annotations(self,
        jax_func, [x, x],
        num_partitions=2,
        num_variables=1)

  @jtu.with_mesh([("x", 2), ("y", 2)])
  def test_pjit_basic2D(self):
    @partial(pjit.pjit,
                       in_axis_resources=(P(None, "x", "y"), P("y")),
                       out_axis_resources=P("x"))
    def jax_func(x, y):
      return x @ y

    x_shape = (8, 6, 4)
    y_shape = (4, 2)
    x = jnp.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape), dtype=np.float32).reshape(y_shape)
    _log_sharding_annotations(self,
        jax_func,
        [x, y],
        num_partitions=4)

  @jtu.with_mesh([("x", 2), ("y", 2)])
  def test_pjit_TwoMeshAxisSharding(self):
    @partial(pjit.pjit,
             in_axis_resources=P(("x", "y"),),
             out_axis_resources=P(("x", "y"),))
    def jax_func(x, y):
      return x @ y

    x_shape = (24, 8)
    y_shape = (8, 2)
    x = jnp.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape), dtype=np.float32).reshape(y_shape)
    _log_sharding_annotations(self,
        jax_func,
        [x, y],
        num_partitions=4)

  @jtu.with_mesh([("x", 2), ("y", 1)])
  def test_pjit_ShardingConstraint(self):
    @partial(pjit.pjit, in_axis_resources=None,
                       out_axis_resources=None)
    def jax_func(x):  # x: f32[12, 8]
      y = jnp.tile(x, (2, 1))  # y: f32[24, 8]
      y = pjit.with_sharding_constraint(y, P("x", "y"))
      return y[0:y.shape[0] // 4]  # res: f32[6, 8]

    shape = (12, 8)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    _log_sharding_annotations(self,
        jax_func, [x],
        num_partitions=2)


class PjitTest(tf_test_util.JaxToTfTestCase):

  def create_test_mesh(self, *axis_names):
    """Creates a mesh with 2 axes"""
    assert len(axis_names) == 2, axis_names
    nr_devices = len(jax.devices())
    mesh_shape = (2, 1) if nr_devices >= 2 else (1, 1)
    return jtu.create_global_mesh(mesh_shape, axis_names)

  @jtu.with_mesh([("axis", 2)])
  def test_pjit_basic1D(self):
    def func_jax(x):
      return x + x

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    func_pjit = pjit.pjit(func_jax,
                          in_axis_resources=P("axis"),
                          out_axis_resources=None)
    res_jax = func_pjit(x)
    func_tf = jax2tf.convert(func_pjit)

    # Run in tf.function JIT compile mode
    res_tf = tf.function(func_tf, autograph=False, jit_compile=True)(x)
    self.assertAllClose(res_tf.numpy(), res_jax)

    # Run in tf.function mode
    res_tf = tf.function(func_tf, autograph=False)(x)
    self.assertAllClose(res_tf.numpy(), res_jax)

    # Run the converted function in TF eager mode
    with self.assertRaisesRegex(
        ValueError,
        r"A jit function with sharded .* arguments or results must be used under a `tf.function` context"):
      func_tf(x)

    # However, if we use REPLICATED sharding we can run in eager mode
    res_tf = jax2tf.convert(pjit.pjit(func_jax,
                                      in_axis_resources=None,
                                      out_axis_resources=None))(x)
    self.assertAllClose(res_tf.numpy(), res_jax)

  @jtu.with_mesh([("x", 1)])
  def test_pjit_closed_over_const(self):
    const = jnp.full((4, 3), 7, dtype=np.float32)
    @partial(pjit.pjit, in_axis_resources=(P("x"), None), out_axis_resources=None)
    def func_jax(x, y):
      return x + y * const

    with self.create_test_mesh("x", "y"):
      self.ConvertAndCompare(func_jax, jnp.ones((4, 3), dtype=np.float32),
                             jnp.ones((1, 1), dtype=np.float32),
                             limitations=[skip_eager_for_partitioning])

  def test_pjit_closed_over_global_device_array(self):
    global_mesh = self.create_test_mesh("x", "y")

    input1 = np.arange(16).reshape(2, 8)
    input2_raw = np.arange(16).reshape(8, 2)
    input2_array = jax.make_array_from_callback(input2_raw.shape,
                                                jax.sharding.NamedSharding(global_mesh, P("x", "y")),
                                                lambda idx: input2_raw[idx])
    @partial(pjit.pjit,
             in_axis_resources=(P("y", "x"),),
             out_axis_resources=None)
    def jax_func(input_data):
      return jnp.matmul(input_data, input2_array)

    with global_mesh:
      self.ConvertAndCompare(jax_func, input1,
                             limitations=[skip_eager_for_partitioning])

  def test_nested_pjit(self):
    global_mesh = self.create_test_mesh("x", "y")
    x = np.arange(16).reshape(2, 8)

    def func_jax(x):
      # We have a pjit nested inside the function to be converted
      inner_func = pjit.pjit(
          jnp.sin,
          in_axis_resources=(P("y", "x"),),
          out_axis_resources=None)
      return inner_func(x)

    with global_mesh:
      self.ConvertAndCompare(func_jax, x,
                             limitations=[skip_eager_for_partitioning])

  def test_xmap_basic(self):
    local_devices = list(jax.local_devices())
    if len(local_devices) < 2:
      raise unittest.SkipTest("Test requires at least 4 local devices")
    def f(a, b):
      return a * 2, b * 4
    devices = np.array(local_devices[:2]).reshape((1, 2))
    with Mesh(devices, ('x', 'y')):
      fm = xmap(f,
                in_axes=({0: 'a', 1: 'b'}, ['c', ...]),
                out_axes=({0: 'a', 1: 'b'}, ['c', ...]),
                axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
      ashape = (16, 8, 5)
      a = jnp.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = jnp.arange(np.prod(bshape)).reshape(bshape)

      res_jax = fm(a, b)
      self.assertAllClose(res_jax, (a * 2, b * 4))

      # xmap works only with native lowering
      _log_sharding_annotations(self, fm, [a, b],
                                experimental_native_lowering=True)
      res_tf = tf.function(
          jax2tf.convert(fm, experimental_native_lowering=True),
          autograph=False, jit_compile=True)(a, b)
      self.assertAllClose(res_tf, res_jax)

  @jtu.with_mesh([('x', 1), ('y', 2)])
  def test_xmap_collective_reduce(self):
    fm = xmap(lambda a, b: (lax.psum(a * 2, 'a'), b * 4),
              in_axes=(['a', 'b', ...], {0: 'c'}),
              out_axes=(['b', ...], {0: 'c'}),
              axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})
    ashape = (16, 8, 5)
    a = jnp.arange(np.prod(ashape)).reshape(ashape)
    bshape = (2, 7)
    b = jnp.arange(np.prod(bshape)).reshape(bshape)
    res_jax = fm(a, b)
    self.assertAllClose(res_jax, ((a * 2).sum(0), b * 4))

    _log_sharding_annotations(self, fm, [a, b],
                              experimental_native_lowering=True)
    res_tf = tf.function(
          jax2tf.convert(fm, experimental_native_lowering=True),
          autograph=False, jit_compile=True)(a, b)
    self.assertAllClose(res_tf, res_jax)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
