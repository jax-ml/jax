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
from absl.testing import parameterized

import jax
from jax._src import test_util as jtu
from jax.config import config
from jax import lax
from jax.experimental import jax2tf
from jax.experimental import pjit
from jax.experimental.maps import xmap
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
from jax._src import xla_bridge

import numpy as np

import tensorflow as tf  # type: ignore[import]

config.parse_flags_with_absl()

# Must come after initializing the flags
from jax.experimental.jax2tf.tests import tf_test_util
from jax.experimental.jax2tf.tests.jax2tf_limitations import Jax2TfLimitation


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


def check_sharding_annotations(test,
                              f_jax,
                              args: Sequence[Any],
                              *,
                              num_replicas=1,
                              num_partitions=2,
                              num_variables=0,
                              experimental_native_lowering="default",
                              checks=()):
  """Log the HLO generated from f_jax and its conversion.

  Ideally this would check the sharding of intermediate results in JAX and
  TF, but this has turned out to be very brittle and broke down for
  StableHLO lowering (the sharding annotation are now binary-encoded
  attributes). We kept the logging aspect of this function, which should
  help some debugging.

  Args:
    checks: a list of tuples, with a regular expression and a number of times
      it is expected to occur in the TF-generated HLO.
  """
  if jtu.device_under_test() == "gpu":
    raise unittest.SkipTest("Sharding HLO tests not useful for GPU")

  jax_comp = f_jax.lower(*args).compiler_ir(dialect="mhlo")
  jax_hlo = str(jax_comp)
  logging.info("[%s] got JAX HLO %s", test._testMethodName, jax_hlo)

  # We only dump JAX optimized code on the TPU
  if jtu.device_under_test() == "tpu":
    backend = xla_bridge.get_backend()
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
  tf_hlo_generator = f_tf_fun.experimental_get_compiler_ir(*args)
  tf_hlo = tf_hlo_generator(stage="hlo", device_name=device_name)
  logging.info("[%s] got TF HLO %s", test._testMethodName, tf_hlo)
  tf_optimized_hlo = tf_hlo_generator(stage="optimized_hlo",
                                      device_name=device_name)
  logging.info("[%s] got TF optimized HLO for %s: %s", test._testMethodName,
                device_name, tf_optimized_hlo)

  for check_re, check_count in checks:
    count = len(re.findall(check_re, tf_hlo))
    test.assertEqual(
        count, check_count,
        (f"regular expression `{check_re}` expected to occur {check_count}"
         f" but occurs {count} times in the TF HLO.\nThis is the TF HLO:\n{tf_hlo}"))


class ShardedJitHloTest(tf_test_util.JaxToTfTestCase):
  """Tests that inspect the HLO for the sharding annotations.

  These tests can run on any device.
  """


  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic1D(self):

    @partial(pjit.pjit,
                       in_shardings=(P("x"), P("x")),
                       out_shardings=None)
    def jax_func(x, y):
      return x + y

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hlo = jax_func.lower(x, x).compiler_ir(dialect="hlo").as_hlo_text()
    logging.info("HLO is %s", hlo)
    logging.info("JAXPR is %s", jax.make_jaxpr(jax_func)(x, x))
    check_sharding_annotations(self,
        jax_func, [x, x],
        num_partitions=2)

  @jtu.with_mesh([("x", 2)])
  def test_pjit_basic1D_variable(self):
    # The first argument is a tf.Variable
    @partial(pjit.pjit,
                       in_shardings=(P("x"), P("x")),
                       out_shardings=None)
    def jax_func(x, y):
      return x + y

    shape = (8, 10)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    hlo = jax_func.lower(x, x).compiler_ir(dialect="hlo").as_hlo_text()
    logging.info("HLO is %s", hlo)
    logging.info("JAXPR is %s", jax.make_jaxpr(jax_func)(x, x))
    check_sharding_annotations(self,
        jax_func, [x, x],
        num_partitions=2,
        num_variables=1)

  @jtu.with_mesh([("x", 2), ("y", 2)])
  def test_pjit_basic2D(self):
    @partial(pjit.pjit,
                       in_shardings=(P(None, "x", "y"), P("y")),
                       out_shardings=P("x"))
    def jax_func(x, y):
      return x @ y

    x_shape = (8, 6, 4)
    y_shape = (4, 2)
    x = jnp.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape), dtype=np.float32).reshape(y_shape)
    check_sharding_annotations(self,
        jax_func,
        [x, y],
        num_partitions=4)

  @jtu.with_mesh([("x", 2), ("y", 2)])
  def test_pjit_TwoMeshAxisSharding(self):
    @partial(pjit.pjit,
             in_shardings=P(("x", "y"),),
             out_shardings=P(("x", "y"),))
    def jax_func(x, y):
      return x @ y

    x_shape = (24, 8)
    y_shape = (8, 2)
    x = jnp.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape), dtype=np.float32).reshape(y_shape)
    check_sharding_annotations(self,
        jax_func,
        [x, y],
        num_partitions=4)

  @parameterized.named_parameters(
      dict( testcase_name=f"_nested_pjit={nested_pjit}", nested_pjit=nested_pjit)
      for nested_pjit in (True, False)
  )
  @jtu.with_mesh([("x", 2), ("y", 1)])
  def test_pjit_ShardingConstraint(self, nested_pjit=True):
    @partial(pjit.pjit, in_shardings=None,
                       out_shardings=None)
    def jax_func(x):  # x: f32[12, 8]
      y = jnp.cos(x)
      if nested_pjit:
        y = pjit.pjit(lambda y: y, in_shardings=P("x", "y"),
                      out_shardings=P("x", "y"))(y)
      else:
        y = pjit.with_sharding_constraint(y, P("x", "y"))
      return jnp.sin(y)  # res: f32[6, 8]

    shape = (12, 8)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    count_sharding = 2 if nested_pjit else 1
    check_sharding_annotations(self,
        jax_func, [x],
        num_partitions=2,
        checks=[
            (r"custom_call_target.*Sharding.*sharding.*devices=\[2,1\]", count_sharding),
            (r"custom_call_target.*Sharding.*sharding.*replicated", 2),
        ])

  @parameterized.named_parameters(
      dict( testcase_name=f"_nested_pjit={nested_pjit}", nested_pjit=nested_pjit)
      for nested_pjit in (True, False)
  )
  @jtu.with_mesh([("x", 2), ("y", 1)])
  def test_pjit_ShardingConstraintReplicated(self, nested_pjit=True):
    shape = (12, 8)
    @partial(pjit.pjit, in_shardings=(P("x", "y"),),
             out_shardings=P("y", "x"))
    def jax_func(x):
      y = jnp.cos(x)
      if nested_pjit:
        y = pjit.pjit(lambda y: y, in_shardings=None, out_shardings=None)(y)
      else:
        y = pjit.with_sharding_constraint(y, None)
      return jnp.sin(y)

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    count_replicated = 2 if nested_pjit else 1
    check_sharding_annotations(
        self,
        jax_func, [x],
        num_partitions=2,
        checks=[
            (r"custom_call_target.*Sharding.*sharding.*devices=\[2,1\]", 1),
            (r"custom_call_target.*Sharding.*sharding.*replicated", count_replicated),
            (r"custom_call_target.*Sharding.*sharding.*devices=\[1,2\]", 1)
        ])


class ShardingTest(tf_test_util.JaxToTfTestCase):
  """
  To verify that the tests do run indeed on multiple devices you can run

     perftools/gputools/profiler/jfprof.sh jax/experimental/jax2tf/tests:sharding_test_tpu -- -c opt --test_filter=ShardingTest.test_shmap_all_to_all --test_arg=--vmodule=jax2tf=3 --
  """
  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 2:
      raise unittest.SkipTest("Test requires at least 2 local devices")
    self.devices = np.array(jax.devices()[:2])  # use 2 devices

    if jtu.device_under_test() == "tpu":
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
      tf.config.experimental_connect_to_cluster(resolver)
      # Do TPU init at beginning since it will wipe out all HBMs.
      self.topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    else:
      self.topology = None

  def device_assignment(self,
                        computation_shape=(1, 1, 1, 2),
                        num_replicas=1):
    self.assertEqual(jtu.device_under_test(), "tpu")
    return tf.tpu.experimental.DeviceAssignment.build(
      self.topology, computation_shape=computation_shape,
      num_replicas=num_replicas)

  def test_pjit_basic1D(self):
    @partial(pjit.pjit, in_shardings=(P("x"),),
             out_shardings=None)
    def f_jax(a):
      return a + a

    shape = (8, 10)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a):
      f_converted = jax2tf.convert(f_jax,
                                   experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2],
            ))[0]
      else:
        res = f_converted(a)
      return res

    with Mesh(self.devices, axis_names=("x",)):
      res_jax = f_jax(a)
      res_tf = f_tf(a)
      self.assertAllClose(res_tf.numpy(), res_jax)

  def test_pjit_closed_over_const(self):

    const = jnp.full((4, 3), 7, dtype=np.float32)
    a = np.ones((4, 3), dtype=np.float32)
    b = np.ones((1, 1), dtype=np.float32)
    @partial(pjit.pjit, in_shardings=(P("x"), None),
             out_shardings=None)
    def f_jax(a, b):
      return a + b * const

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a, b):
      f_converted = jax2tf.convert(f_jax, experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a), tf.convert_to_tensor(b)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
            )[0]
      else:
        res = f_converted(a, b)
      return res

    with Mesh(self.devices, axis_names=("x",)):
      res_jax = f_jax(a, b)
      res_tf = f_tf(a, b)
      self.assertAllClose(res_tf, res_jax)

  def test_nested_pjit(self):
    if not config.jax_array:
      raise unittest.SkipTest("Test works only with jax_array")
    a = np.arange(16., dtype=np.float32).reshape(2, 8)

    def f_jax(a):
      # We have a pjit nested inside the function to be converted
      inner_func = pjit.pjit(
          jnp.sin, in_shardings=(P("x"),), out_shardings=None
      )
      return jnp.cos(inner_func(a))

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a):
      f_converted = jax2tf.convert(f_jax, experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
            )[0]
      else:
        res = f_converted(a)
      return res

    with Mesh(self.devices, axis_names=("x",)):
      res_jax = f_jax(a)
      res_tf = f_tf(a)
      self.assertAllClose(res_tf, res_jax)

  @parameterized.named_parameters(
      dict( testcase_name=f"_func={func}", func=func)
      for func in ("pjit_sharded", "pjit_replicated",
                   "nested_pjit_sharded", "nested_pjit_replicated")
  )
  def test_pjit_eager_error(self, func="pjit_sharded"):
    if config.jax2tf_default_experimental_native_lowering:
      raise unittest.SkipTest("There is no error in eager mode for native serialization")
    # Define some test functions
    @partial(pjit.pjit, in_shardings=(P("x"),),
             out_shardings=None)
    def f_pjit_sharded(a):
      return a + a

    @partial(pjit.pjit, in_shardings=None,
             out_shardings=None)
    def f_pjit_replicated(a):
      return a + a

    def f_nested_pjit_sharded(a):
      return a + pjit.pjit(jnp.sin, in_shardings=(P("x"),), out_shardings=None)(a)

    def f_nested_pjit_replicated(a):
      return a + pjit.pjit(jnp.sin, in_shardings=None, out_shardings=None)(a)

    shape = (8, 10)
    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    if func == "pjit_sharded":
      f_jax = f_pjit_sharded
    elif func == "pjit_replicated":
      f_jax = f_pjit_replicated
    elif func == "nested_pjit_sharded":
      f_jax = f_nested_pjit_sharded
    elif func == "nested_pjit_replicated":
      f_jax = f_nested_pjit_replicated
    else:
      assert False

    with Mesh(self.devices, axis_names=("x",)):
      _ = f_jax(a)
      with self.assertRaisesRegex(
          ValueError,
          "function with sharded arguments or results must be used under a `tf.function` context"):
        jax2tf.convert(f_jax)(a)

  def test_xmap_basic(self):
    devices = np.reshape(self.devices, (1, 2))

    f_jax = xmap(lambda a, b: (a * 2, b * 4),
                 in_axes=({0: 'a', 1: 'b'}, ['c', ...]),
                 out_axes=({0: 'a', 1: 'b'}, ['c', ...]),
                 axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a, b):
      f_converted = jax2tf.convert(f_jax, experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a), tf.convert_to_tensor(b)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
            )
        res = (res[0], res[1])
      else:
        res = f_converted(a, b)
      return res

    with Mesh(devices, ('x', 'y')):
      ashape = (16, 8, 5)
      a = np.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = np.arange(np.prod(bshape)).reshape(bshape)

      res_jax = f_jax(a, b)
      self.assertAllClose(res_jax, (a * 2, b * 4))

      # jax2tf for xmap works only with native lowering
      check_sharding_annotations(self, f_jax, [a, b],
                                 experimental_native_lowering=True)
      res_tf = f_tf(a, b)
      self.assertAllClose(res_tf, res_jax)

  def test_xmap_collective_reduce(self):
    devices = np.reshape(self.devices, (1, 2))

    f_jax = xmap(lambda a, b: (lax.psum(a * 2, 'a'), b * 4),
              in_axes=(['a', 'b', ...], {0: 'c'}),
              out_axes=(['b', ...], {0: 'c'}),
              axis_resources={'a': 'x', 'b': 'y', 'c': 'x'})

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a, b):
      f_converted = jax2tf.convert(f_jax, experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a), tf.convert_to_tensor(b)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
            )
        res = (res[0], res[1])
      else:
        res = f_converted(a, b)
      return res

    with Mesh(devices, ('x', 'y')):
      ashape = (16, 8, 5)
      a = np.arange(np.prod(ashape)).reshape(ashape)
      bshape = (2, 7)
      b = np.arange(np.prod(bshape)).reshape(bshape)
      res_jax = f_jax(a, b)
      self.assertAllClose(res_jax, ((a * 2).sum(0), b * 4))

      check_sharding_annotations(self, f_jax, [a, b],
                                 experimental_native_lowering=True)
      res_tf = f_tf(a, b)
      self.assertAllClose(res_tf, res_jax)

  @jtu.ignore_warning(category=UserWarning,
                      message="all_to_all .* are only implemented properly for TPUs and GPUs .*")
  def test_shmap_all_to_all(self):
    if jtu.device_under_test() == "cpu":
      raise unittest.SkipTest("TODO(b/268295912): ShardingRemover crash")
    mesh = Mesh(self.devices, axis_names=('x'))

    @partial(pjit.pjit,
             in_shardings=(P('x', None),), out_shardings=P(None, 'x'))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P(None, 'x'))
    def f_jax(b):  # b: f32[2, 4]
      return lax.all_to_all(b, 'x', split_axis=1, concat_axis=1, tiled=True)

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a):
      f_converted = jax2tf.convert(f_jax, experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
            )[0]
      else:
        res = f_converted(a)
      return res

    with mesh:
      a = np.arange(np.prod(4 * 4)).reshape((4, 4))
      res_jax = f_jax(a)  # res: f32[2, 8]
      b0, b1 = np.split(a, 2, axis=0)  # The shard_map in_specs splits on axis 0
      b00, b01 = np.split(b0, 2, axis=1)  # split_axis=1
      b10, b11 = np.split(b1, 2, axis=1)
      b0 = np.concatenate([b00, b10], axis=1)  # concat_axis=1
      b1 = np.concatenate([b01, b11], axis=1)
      res = np.concatenate([b0, b1], axis=1)  # out_specs concatenates on axis 1
      self.assertAllClose(res_jax, res)
      res_tf = f_tf(a)
      self.assertAllClose(res_tf, res_jax)

  @unittest.skip("TODO(b/268295912): ShardingRemover crash")
  def test_repro_xla_bug_shmap_collective_permute(self):
    mesh = Mesh(self.devices, axis_names=('x'))

    @partial(pjit.pjit,
             in_shardings=(P('x', None),), out_shardings=P('x', None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P('x', None))
    def f_jax(b):  # b: f32[2, 4]
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(b, 'x', perm=perm)

    with mesh:
      a = np.arange(np.prod(4 * 4)).reshape((4, 4))
      res_jax = f_jax(a)
      b0, b1 = np.split(a, 2, axis=0)  # The shard_map splits on axis 0
      b0, b1 = b1, b0
      expected = np.concatenate([b0, b1], axis=0)  # out_specs concatenates on axis 0
      self.assertAllClose(res_jax, expected)

      check_sharding_annotations(self, f_jax, [a],
                                 experimental_native_lowering=True,
                                 num_partitions=2, num_replicas=1)
      # XLA bug: invoke the f_tf without tpu.replicate
      f_tf = tf.function(
          jax2tf.convert(f_jax, experimental_native_lowering=True),
          autograph=False, jit_compile=True)

      res_tf = f_tf(a)
      self.assertAllClose(res_tf, expected)

  def test_shmap_collective_permute(self):
    if jtu.device_under_test() == "cpu":
      raise unittest.SkipTest("TODO(b/268295912): ShardingRemover crash")
    mesh = Mesh(self.devices, axis_names=('x'))

    @partial(pjit.pjit,
             in_shardings=(P('x', None),), out_shardings=P('x', None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P('x', None),), out_specs=P('x', None))
    def f_jax(b):  # b: f32[2, 4]
      axis_size = lax.psum(1, 'x')
      perm = [(j, (j + 1) % axis_size) for j in range(axis_size)]
      return lax.ppermute(b, 'x', perm=perm)

    @tf.function(autograph=False, jit_compile=True)
    def f_tf(a):
      f_converted = jax2tf.convert(f_jax, experimental_native_lowering=True)
      if jtu.device_under_test() == "tpu":
        res = tf.compat.v1.tpu.rewrite(
            f_converted, [tf.convert_to_tensor(a)],
            device_assignment=self.device_assignment(
                computation_shape=[1, 1, 1, 2])
            )[0]
      else:
        res = f_converted(a)
      return res

    with mesh:
      a = np.arange(np.prod(4 * 4)).reshape((4, 4))
      res_jax = f_jax(a)
      b0, b1 = np.split(a, 2, axis=0)  # The shard_map splits on axis 0
      b0, b1 = b1, b0
      expected = np.concatenate([b0, b1], axis=0)  # out_specs concatenates on axis 0
      self.assertAllClose(res_jax, expected)
      res_tf = f_tf(a)
      self.assertAllClose(res_tf, expected)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
