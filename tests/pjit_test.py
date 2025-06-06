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

from collections import OrderedDict, namedtuple
import re
from functools import partial, wraps
import logging
import json
import math
import textwrap
import threading
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import concurrent.futures

import jax
import jax.numpy as jnp
from jax._src import core
from jax._src import config
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src import dtypes
from jax import stages
from jax import lax
from jax._src.lax import lax as lax_internal
from jax.lax import with_sharding_constraint
from jax._src import prng
from jax.sharding import PartitionSpec as P, Mesh
from jax.experimental import multihost_utils
from jax._src.shard_map import shard_map
from jax._src.compilation_cache import is_persistent_cache_enabled
from jax.experimental.custom_partitioning import (
    custom_partitioning, SdyShardingRule, BATCHING)
from jax.experimental import primal_tangent_dtype
from jax._src import array
from jax._src.sharding import Sharding, common_devices_indices_map
from jax._src import op_shardings
from jax._src import sharding_impls
from jax._src.sharding_impls import (
    AUTO, UNSPECIFIED, NamedSharding, GSPMDSharding, PositionalSharding,
    SingleDeviceSharding, parse_flatten_op_sharding)
from jax._src.pjit import (pjit, mesh_cast, auto_axes, explicit_axes,
                           use_auto_axes, use_explicit_axes, reshard,
                           _pjit_lower_cached)
from jax._src.layout import Format, DeviceLocalLayout as DLL
from jax._src.named_sharding import DuplicateSpecError
from jax._src import mesh as mesh_lib
from jax._src.mesh import AxisType
from jax._src.interpreters import pxla
from jax._src import xla_bridge
from jax._src.lib import xla_client as xc
from jax._src.lib import _jax
from jax._src.util import curry, unzip2

config.parse_flags_with_absl()

jtu.request_cpu_devices(8)

def create_array(global_shape, global_mesh, mesh_axes, global_data=None,
                 dtype=np.float32):
  if global_data is None:
    global_data = np.arange(
        math.prod(global_shape), dtype=dtype).reshape(global_shape)

  if isinstance(mesh_axes, Sharding):
    sharding = mesh_axes
  else:
    sharding = NamedSharding(global_mesh, mesh_axes)

  return array.make_array_from_callback(
      global_shape, sharding, lambda idx: global_data[idx]), global_data


def _check_instance(self, x):
  self.assertIsInstance(x, array.ArrayImpl)


@curry
def check_1d_2d_mesh(f, set_mesh):
  return parameterized.named_parameters(
    {"testcase_name": "_" + name, "mesh": mesh, "resources": resources}
    for name, mesh, resources in (
      ("2", (("x", 2),), "x"),
      ("2x1", (("x", 2), ("y", 1)), ("x", "y")),
      ("2x2", (("x", 2), ("y", 2)), ("x", "y")),
    ))(jtu.with_mesh_from_kwargs(f) if set_mesh else f)


# TODO(skye): make the buffer donation utils part of JaxTestCase
@jtu.pytest_mark_if_available('multiaccelerator')
class PJitTest(jtu.BufferDonationTestCase):

  @jtu.with_mesh([('x', 1)])
  def testDeviceBufferAval(self):

    @partial(pjit, in_shardings=None, out_shardings=P('x'))
    def f(x):
      return x

    shape = (2, 2)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x)
    expected = x
    self.assertAllClose(actual, expected, check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 1)
    self.assertAllClose(
        np.asarray(actual.addressable_shards[0].data), expected, check_dtypes=False)
    # Repro for a bug on addressable_shards aval
    _ = repr(actual.addressable_shards)

  @jtu.with_mesh([('x', 2)])
  def testBasic1D(self):
    @partial(pjit,
             in_shardings=(P('x'), P('x')),
             out_shardings=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertAllClose(actual, expected, check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data), expected,
                        check_dtypes=False)

  @jtu.with_mesh([('x', 2)])
  def testJitOfPjitDisallowed(self):
    @partial(pjit,
             in_shardings=(P('x'), P('x')),
             out_shardings=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    out = jax.jit(f)(x, x + 1)
    self.assertArraysEqual(out, x + x + 1)

  @jtu.with_mesh([('x', 2)])
  def testUnevenShardingConstraint(self):
    @partial(pjit,
             in_shardings=(P('x'), P('x')),
             out_shardings=None)
    def f(x, y):
      x = x[:3]
      y = y[:3]
      x = with_sharding_constraint(x, P('x'))
      y = with_sharding_constraint(y, P('x'))
      out = x + y
      return jnp.pad(out, [[0, 1]])

    shape = (4,)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertAllClose(actual[:3], expected[:3], check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data)[:3],
                        expected[:3], check_dtypes=False)

  def testBasic1DWithMeshContextManager(self):
    @partial(pjit,
             in_shardings=(P('x'), P('x')),
             out_shardings=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    with jtu.create_mesh((2,), ('x')) as mesh:
      actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertEqual(mesh, jtu.create_mesh((2,), ('x')))
    self.assertAllClose(actual, expected, check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data), expected,
                        check_dtypes=False)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testBasic2D(self):
    @partial(pjit,
             in_shardings=(P(None, 'x', 'y'), P('y')),
             out_shardings=P('x'))
    def f(x, y):
      return x @ y

    x_shape = (8, 6, 4)
    y_shape = (4, 2)
    x = jnp.arange(math.prod(x_shape)).reshape(x_shape)
    y = jnp.arange(math.prod(y_shape)).reshape(y_shape)
    actual = f(x, y)
    expected = x @ y
    self.assertAllClose(actual, expected, check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 4)

    split0, split1 = np.split(expected, 2)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data), split0,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[1].data), split0,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[2].data), split1,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[3].data), split1,
                        check_dtypes=False)

  def testDifferentNestedMesh(self):
    with jtu.create_mesh((2, 1), ("x", "y")) as m1:
      with jtu.create_mesh((2, 2), ("a", "b")) as m2:
        self.assertEqual(mesh_lib.thread_resources.env.physical_mesh, m2)
      self.assertEqual(mesh_lib.thread_resources.env.physical_mesh, m1)
    self.assertEqual(mesh_lib.thread_resources.env.physical_mesh,
                     mesh_lib.EMPTY_ENV.physical_mesh)

  def testSameNestedMesh(self):
    mesh = jtu.create_mesh((2, 1), ("a", "b"))
    thread_resources = mesh_lib.thread_resources
    with mesh as m1:
      with mesh as m2:
        self.assertEqual(thread_resources.env.physical_mesh, m2)
      self.assertEqual(thread_resources.env.physical_mesh, m1)
    self.assertEqual(thread_resources.env.physical_mesh,
                     mesh_lib.EMPTY_ENV.physical_mesh)

  def testMeshDecorator(self):
    x = jnp.arange(8)
    mesh_shape = (2, 2)
    size = math.prod(mesh_shape)
    if len(jax.devices()) < size:
      raise unittest.SkipTest(f"Test requires {size} global devices.")
    mesh_devices = np.array(jax.devices()[:size]).reshape(mesh_shape)

    @jax.sharding.Mesh(mesh_devices, ('x', 'y'))
    def dec():
      return pjit(lambda x: x, in_shardings=P('x'), out_shardings=None)(x)
    out = dec()
    self.assertArraysEqual(out, x)

  def testMeshHashRace(self):
    mesh = jtu.create_mesh((2, 1), ('a', 'testMeshHashRace'))
    self.assertFalse(hasattr(mesh, '_hash'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
      fs = []
      for _ in range(5):
        fs.append(pool.submit(lambda: hash(mesh)))
      for f in concurrent.futures.as_completed(fs):
        f.result()
    self.assertTrue(hasattr(mesh, '_hash'))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testTwoMeshAxisSharding(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=jax.sharding.PartitionSpec(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    actual = f(x, x + 1)
    expected = x @ (x + 1)
    self.assertAllClose(actual, expected, check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 4)

    splits = np.split(expected, 4)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data), splits[0],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[1].data), splits[1],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[2].data), splits[2],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[3].data), splits[3],
                        check_dtypes=False)

  @jtu.with_mesh([('x', 2)])
  @jtu.run_on_devices('cpu', 'gpu', 'tpu')
  def testBufferDonation(self):
    @partial(pjit, in_shardings=P('x'), out_shardings=P('x'), donate_argnums=0)
    def f(x, y):
      return x + y

    shard = pjit(lambda x: x, in_shardings=P('x'), out_shardings=P('x'))
    x = shard(jnp.ones((2, 5)) * 4)
    y = shard(jnp.ones((2, 5)) * 2)
    expected = x + y
    self.assertAllClose(f(x, y), expected)
    self.assertNotDeleted(y)
    self.assertDeleted(x)

  @jtu.run_on_devices('cpu', 'gpu', 'tpu')
  def testBufferDonationWithNames(self):
    mesh = jtu.create_mesh((2,), ('x'))
    s = NamedSharding(mesh, P('x'))

    @partial(pjit, out_shardings=s, donate_argnames='inp2')
    def f(inp1, inp2):
      return inp1 + inp2

    x = jax.device_put(np.ones((2, 5)) * 4, s)
    y = jax.device_put(np.ones((2, 5)) * 2, s)
    expected = x + y
    self.assertAllClose(f(x, y), expected)
    self.assertNotDeleted(x)
    self.assertDeleted(y)

  @jtu.run_on_devices('cpu', 'gpu', 'tpu')
  def testBufferDonationWithKwargs(self):
    mesh = jtu.create_mesh((2,), ('x'))
    s = NamedSharding(mesh, P('x'))

    @partial(pjit, out_shardings=s, donate_argnames=('inp2', 'inp3'))
    def f(inp1, inp2, inp3):
      return inp1 + inp2 + inp3, inp3

    x = jax.device_put(np.ones((2, 5)) * 4, s)
    y = jax.device_put(np.ones((2, 5)) * 2, s)
    z = jax.device_put(np.ones((2, 5)), s)

    expected = x + y + z
    self.assertAllClose(f(x, inp2=y, inp3=z)[0], expected)
    self.assertNotDeleted(x)
    self.assertDeleted(y)
    self.assertDeleted(z)

  @jtu.run_on_devices('cpu', 'gpu', 'tpu')
  def testBufferDonationWithPyTreeKwargs(self):
    mesh = jtu.create_mesh((2,), ('x'))
    s = NamedSharding(mesh, P('x'))

    @partial(pjit, out_shardings=s, donate_argnames='inp2')
    def f(inp1, inp2, inp3):
      return jax.tree.map(lambda x, y, z: x + y + z, inp1, inp2, inp3)

    x = np.ones((2, 5)) * 4
    x_tree = jax.device_put({"a": {"b": x}, "c": x}, s)

    y = np.ones((2, 5)) * 2
    y_tree = jax.device_put({"a": {"b": y}, "c": y}, s)

    z = np.ones((2, 5))
    z_tree = jax.device_put({"a": {"b": z}, "c": z}, s)

    expected = x + y + z
    out = f(x_tree, inp2=y_tree, inp3=z_tree)
    jax.tree.map(lambda o: self.assertAllClose(o, expected), out)
    jax.tree.map(self.assertNotDeleted, x_tree)
    jax.tree.map(self.assertDeleted, y_tree)
    jax.tree.map(self.assertNotDeleted, z_tree)

  @jtu.run_on_devices('tpu', 'cpu', 'gpu')
  def testBufferDonationWithOutputShardingInference(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P('x'))
    rs = NamedSharding(mesh, P())

    @partial(pjit, donate_argnames=('inp2', 'inp3'))
    def f(inp1, inp2, inp3):
      return (
          jax.lax.with_sharding_constraint(inp1, rs),
          inp1,
          jax.lax.with_sharding_constraint(inp2, rs),
          inp2,
          jax.lax.with_sharding_constraint(inp3, rs),
          inp3,
      )

    x = np.ones((2, 5)) * 4
    x_tree = jax.device_put({'a': {'b': x}, 'c': x}, s)

    y = np.ones((2, 7)) * 2
    y_tree = jax.device_put({'a': {'b': y}, 'c': y}, s)

    z = np.ones((2, 11))
    z_tree = jax.device_put({'a': {'b': z}, 'c': z}, s)

    out = f(x_tree, y_tree, z_tree)
    jax.tree.map(self.assertNotDeleted, x_tree)
    jax.tree.map(self.assertDeleted, y_tree)
    jax.tree.map(self.assertDeleted, z_tree)

  @jtu.run_on_devices('tpu')
  def testBufferDonationWithOutputShardingInferenceAndTokens(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P('x'))

    def _callback(x):
      self.assertIsInstance(x, jax.Array)

    @partial(pjit, donate_argnames=('x'))
    def f(x):
      # Just to get tokens.
      jax.experimental.io_callback(_callback, None, x, ordered=True)
      jax.experimental.io_callback(_callback, None, x, ordered=True)
      return x * x

    x = np.ones((2, 5)) * 4
    x = jax.device_put(x, s)
    f(x)
    jax.effects_barrier()
    self.assertDeleted(x)

  @jtu.run_on_devices('tpu', 'cpu', 'gpu')
  def testBufferDonationNotDonated(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P('x'))

    @partial(pjit, donate_argnames=('x'))
    def f(x):
      return x @ x.T

    x = jax.device_put(np.arange(16).reshape(8, 2), s)
    f(x)
    self.assertNotDeleted(x)

  @jtu.run_on_devices('tpu', 'cpu', 'gpu')
  def testBufferDonationMixedConstrainedness(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P())
    s2 = NamedSharding(mesh, P(P.UNCONSTRAINED, P.UNCONSTRAINED))

    @partial(pjit, donate_argnames=('x', 'y'), out_shardings=(s2, s))
    def f(x, y):
      return x * 2, y * 2

    x1 = jax.device_put(np.arange(16).reshape(8, 2), s)
    x2 = jax.device_put(np.arange(16).reshape(8, 2), s)
    txt = f.lower(x1, x2).as_text()
    self.assertIn("jax.buffer_donor = true", txt)
    self.assertIn("tf.aliasing_output = 1 : i32", txt)
    f(x1, x2)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testShardingConstraintStablehlo(self):
    @partial(pjit, in_shardings=None, out_shardings=None)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, P('x', 'y'))
      return y * 2

    shape = (8, 8)
    x = np.arange(math.prod(shape)).reshape(shape)
    expected = (x + 1) * 2
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    _check_instance(self, actual)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data), expected,
                        check_dtypes=False)

    hlo = f.lower(np.ones(shape)).compiler_ir()
    if config.use_shardy_partitioner.value:
      # Annotation from with_sharding_constraint
      self.assertIn('<@mesh, [{"x"}, {"y"}]>', str(hlo))
      # Annotation from pjit
      self.assertIn('sharding = #sdy.sharding<@mesh, [{}, {}]>}', str(hlo))
    else:
      # Annotation from with_sharding_constraint
      self.assertIn('sharding = "{devices=[2,1]<=[2]}"', str(hlo))
      # Annotation from pjit
      self.assertIn('sharding = "{replicated}"', str(hlo))

  def testShardingConstraintWithArray(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P(None))

    @partial(pjit, in_shardings=s, out_shardings=s)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, NamedSharding(mesh, P('x', 'y')))
      return y * 2

    shape = (8, 8)
    x = np.arange(math.prod(shape)).reshape(shape)
    expected = (x + 1) * 2
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, array.ArrayImpl)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(actual, expected, check_dtypes=False)

    hlo = f.lower(np.ones(shape)).compiler_ir(dialect="hlo")
    # Annotation from with_sharding_constraint
    self.assertIn('sharding={devices=[2,1]<=[2]}', hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  def testShardingConstraintWithArrayOpSharding(self):
    if config.use_shardy_partitioner.value:
      self.skipTest("Shardy doesn't support PositionalSharding")
    shape = (8, 8)
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P(None))
    ops = pxla.to_gspmd_sharding(
        NamedSharding(mesh, P('x', 'y')), len(shape))

    @partial(pjit, in_shardings=s, out_shardings=s)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, ops)
      return y * 2

    x = np.arange(math.prod(shape)).reshape(shape)
    expected = (x + 1) * 2
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, array.ArrayImpl)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(actual, expected, check_dtypes=False)

    hlo = f.lower(np.ones(shape)).compiler_ir(dialect="hlo")
    # Annotation from with_sharding_constraint
    self.assertIn('sharding={devices=[2,1]<=[2]}', hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  def testShardingConstraintPyTreeWithArray(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))

    @jax.jit
    def f(x):
      return with_sharding_constraint(x, NamedSharding(mesh, P('x', 'y')))

    shape = (8, 8)
    v = np.arange(math.prod(shape)).reshape(shape)
    x = [v, v * 2]
    out = f(x)

    self.assertArraysEqual(out[0], v)
    self.assertArraysEqual(out[1], v * 2)
    self.assertLen(out[0].addressable_shards, 2)
    self.assertLen(out[1].addressable_shards, 2)

    hlo = f.lower(x).compiler_ir(dialect="hlo")
    # Annotations from with_sharding_constraint
    self.assertIn('sharding={devices=[2,1]<=[2]}', hlo.as_hlo_text())
    self.assertIn('sharding={devices=[2,1]<=[2]}', hlo.as_hlo_text())

  def testShardingConstraintPyTreeWithUnconstrainedDimsWithJit(self):

    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    @jax.jit
    def f(x):
      x = with_sharding_constraint(
          x, [NamedSharding(mesh, P(P.UNCONSTRAINED, 'y', None)),
              NamedSharding(mesh, P('x', P.UNCONSTRAINED, None))])
      x = x.copy()
      x[0]['a'] *= 2
      return x

    shape = (2, 8, 8)
    v = np.arange(math.prod(shape)).reshape(shape)
    x = [{'a': v, 'b': v * 2}, v * 3]
    actual = f(x)

    expected = x.copy()
    expected[0]['a'] *= 2
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual[0]['a'].addressable_shards, 4)

    mlir_str = str(f.lower(x).compiler_ir())
    if config.use_shardy_partitioner.value:
      self.assertIn('<@mesh, [{?}, {"y"}, {}]>', mlir_str)
      self.assertIn('<@mesh, [{"x"}, {?}, {}]>', mlir_str)
    else:
      self.assertIn("unspecified_dims=[0]", mlir_str)
      self.assertIn("unspecified_dims=[1]", mlir_str)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testShardingConstraintPyTreeVmapWithUnconstrainedDims(self):

    @partial(pjit, in_shardings=None, out_shardings=None)
    def f(x):
      x = jax.vmap(lambda x: with_sharding_constraint(
          x, [P(P.UNCONSTRAINED, 'y'),
              P('x', P.UNCONSTRAINED)]))(x)
      x = x.copy()
      x[0]['a'] *= 2
      return x

    shape = (2, 8, 8)
    v = np.arange(math.prod(shape)).reshape(shape)
    x = [{'a': v, 'b': v * 2}, v * 3]

    mlir_str = str(f.lower(x).compiler_ir())
    if config.use_shardy_partitioner.value:
      self.assertIn('<@mesh, [{?}, {?}, {"y"}]>', mlir_str)
      self.assertIn('<@mesh, [{?}, {"x"}, {?}]>', mlir_str)
    else:
      self.assertIn("unspecified_dims=[0,1]", mlir_str)
      self.assertIn("unspecified_dims=[0,2]", mlir_str)

  def testCaching(self):
    def f(x):
      assert should_be_tracing
      return jnp.sin(x) * 2

    x = np.arange(16).reshape(4, 4)
    devices = np.array(list(jax.local_devices())[:4])
    if devices.size < 4:
      raise unittest.SkipTest("Test requires 4 devices")
    devices = devices.reshape((2, 2))
    with jax.sharding.Mesh(devices, ('x', 'y')):
      should_be_tracing = True
      pjit(f, in_shardings=P(('x', 'y')), out_shardings=None)(x)
      should_be_tracing = False
      pjit(f, in_shardings=P(('x', 'y')), out_shardings=None)(x)
    # Re-create the mesh to make sure that has no influence on caching
    with jax.sharding.Mesh(devices, ('x', 'y')):
      should_be_tracing = False
      pjit(f, in_shardings=P(('x', 'y')), out_shardings=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testNested(self):
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    f = pjit(
        lambda x: x.sum() + h.sum(),
        in_shardings=P('x', 'y'),
        out_shardings=None,
    )
    g = pjit(
        lambda x: f(jnp.sin(x)), in_shardings=P('x', None), out_shardings=None
    )
    x = jnp.arange(16.).reshape((4, 4))
    y = g(x)
    self.assertAllClose(y, jnp.sin(x).sum() + h.sum())
    _check_instance(self, y)

  @check_1d_2d_mesh(set_mesh=True)
  def testAutodiff(self, mesh, resources):
    if len(mesh) != 2: return
    assert resources == ('x', 'y')
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    f = pjit(
        lambda x: x.sum(1) * h.sum(),
        in_shardings=P('x', 'y'),
        out_shardings=P(('x', 'y')),
    )
    g = pjit(
        lambda x: f(jnp.sin(x * 4 + 2)),
        in_shardings=P('x', None),
        out_shardings=P(('x', 'y')),
    )
    jtu.check_grads(g, (jnp.arange(16.).reshape((4, 4)) / 100,), order=2)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testAutodiffCache(self):
    f = pjit(lambda x: jnp.sin(x).sum(), in_shardings=P('x'), out_shardings=None)
    x = jnp.arange(16, dtype=jnp.float32)

    jax.grad(f)(x)  # Warm up the cache.
    with jtu.count_pjit_cpp_cache_miss() as count:
      jax.grad(f)(x)
    self.assertEqual(count(), 0)  # no cache miss i.e. cache hit

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testEvalJaxpr(self):
    x, y = jnp.arange(4.), jnp.arange(5.)
    f = pjit(
        lambda x, y: x.sum() + jnp.sin(y),
        in_shardings=(P('x'), P('y')),
        out_shardings=P('y'),
    )
    f_jaxpr = jax.make_jaxpr(f)(x, y)
    f_eval = core.jaxpr_as_fun(f_jaxpr)
    r, = f_eval(x, y)
    self.assertAllClose(r, x.sum() + jnp.sin(y))

  @jtu.with_mesh([('x', 2)])
  def testNonArrayArg(self):
    self.assertEqual(
        pjit(lambda x: x + 2, in_shardings=None, out_shardings=None)(1), 3
    )

  @jtu.with_mesh([('x', 2)])
  def testNonHashableAxisResources(self):
    x = jnp.arange(4)
    y = pjit(
        lambda x: {'b': x['a'] + 2},
        in_shardings=({'a': P('x')},),
        out_shardings={'b': P('x')},
    )({'a': x})
    self.assertAllClose(y, {'b': x + 2})

  @jtu.with_mesh([('x', 2)])
  def testGradOfConstraint(self):
    # Make sure that we can compute grads through sharding constraints
    h = lambda x: jnp.sin(with_sharding_constraint(x, P('x'))).sum()
    f = pjit(lambda x: jax.grad(h)(x), in_shardings=None, out_shardings=None)
    x = jnp.arange(8, dtype=jnp.float32)
    out = f(x)
    self.assertAllClose(out, jnp.cos(x))
    self.assertLen(out.devices(), 2)

  @jtu.with_mesh([('x', 2)])
  def testNoopPartitionSpecs(self):
    noops = [P(), P(None), P(()), P((), None), P(None, None, ())]
    x = jnp.arange(8).reshape((2, 2, 2))
    for spec in noops:
      y = pjit(lambda x: x * 2, in_shardings=spec, out_shardings=spec)(x)
      self.assertAllClose(y, x * 2)

  @jtu.with_mesh([('x', 2)])
  def testVMap(self):
    f = pjit(lambda x, y: (x + y, x), in_shardings=P('x'), out_shardings=P('x'))
    x = jnp.arange(4)
    y = jnp.arange(5*4).reshape((5, 4))
    z, w = jax.vmap(f, in_axes=(None, 0), out_axes=(0, None))(x, y)
    self.assertAllClose(z, x[jnp.newaxis] + y)
    self.assertAllClose(w, x)
    self.assertEqual(
        z.sharding._to_xla_hlo_sharding(z.ndim).tile_assignment_dimensions(),
        [1, 2])
    self.assertEqual(
        w.sharding._to_xla_hlo_sharding(w.ndim).tile_assignment_dimensions(), [2])

  @jtu.with_mesh([('x', 2)])
  def testVMapShardingConstraint(self):
    f = pjit(
        lambda x: with_sharding_constraint(x, P('x')),
        in_shardings=P(),
        out_shardings=P('x'),
    )
    x = jnp.arange(5*4).reshape((5, 4))
    jaxpr = jax.make_jaxpr(jax.vmap(f))(x)
    pjit_eqn, = jaxpr.eqns
    constraint_eqn, = pjit_eqn.params['jaxpr'].eqns
    op = constraint_eqn.params['sharding']._to_xla_hlo_sharding(x.ndim)
    self.assertTrue(op.is_tiled())
    self.assertListEqual(op.tile_assignment_dimensions(), [1, 2])
    self.assertListEqual(op.tile_assignment_devices(), [0, 1])
    self.assertFalse(op_shardings.is_op_sharding_replicated(op))

  @jtu.with_mesh([('x', 2)])
  def testVMapShardingConstraintWithSpmdAxis(self):
    f = pjit(
        jax.vmap(
            lambda x: with_sharding_constraint(x, P(None)),
            spmd_axis_name='x',
        ),
        in_shardings=P('x'),
        out_shardings=P('x'),
    )
    x = jnp.arange(16 * 4).reshape((16, 4))
    jaxpr = jax.make_jaxpr(f)(x)
    pjit_eqn, = jaxpr.eqns
    constraint_eqn, = pjit_eqn.params['jaxpr'].eqns
    op = constraint_eqn.params['sharding']._to_xla_hlo_sharding(x.ndim)
    self.assertTrue(op.is_tiled())
    self.assertListEqual(op.tile_assignment_dimensions(), [2, 1])
    self.assertListEqual(op.tile_assignment_devices(), [0, 1])
    self.assertFalse(op_shardings.is_op_sharding_replicated(op))

  @jtu.with_mesh([('x', 2)])
  def testLowerWithDuckTyping(self):
    x = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    # Make sure this doesn't crash
    pjit(lambda x: x + 4, in_shardings=P('x'), out_shardings=P('x')).lower(x)

  @jtu.with_mesh([('x', 2)])
  def testLowerDonateArgnumsAvailable(self):
    x = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    def f(*args):
      x, *_ = args
      return x
    f_low = pjit(f, donate_argnums=(0,),
                 in_shardings=P('x'), out_shardings=P('x')).lower(x)
    f_com = f_low.compile()
    f_low.donate_argnums == f_com.donate_argnums == (0,)

  @jtu.with_mesh([('x', 2)])
  def testLowerDonateArgnumsAvailableWithNames(self):
    x = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    def f(inp1):
      return inp1

    f_low = pjit(f, in_shardings=P('x'), out_shardings=P('x'),
                 donate_argnames=('inp1',)).lower(x)
    f_com = f_low.compile()
    f_low.donate_argnums == f_com.donate_argnums == (0,)

  @unittest.skip('Fails in OSS builds on GPU with jax at HEAD and latest '
                 'jaxlib on pypi.')
  def testInfeed(self):
    devices = np.array(jax.local_devices())
    nr_devices = len(devices)
    shape = (nr_devices * 3, nr_devices * 5)

    def f_for_jit(x):
      token = lax.create_token(x)
      (y,), token = lax.infeed(
          token, shape=(core.ShapedArray(x.shape, np.float32),))
      (z,), token = lax.infeed(
          token, shape=(core.ShapedArray(x.shape, np.float32),))
      (w,), token = lax.infeed(
          token, shape=(core.ShapedArray(x.shape, np.float32),))

      return x + y + z + w

    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    y = x * 2.
    z = x * 3.
    w = x * 4.

    # Transfer data to infeed before executing the function. For GPUs, the
    # execution of the compiled function is blocking, so transferring data
    # to infeed before executing ensures that the execution does not deadlock
    # waiting for the infeed data.
    logging.info('Transferring to infeed for the jit call')
    d = devices[0]
    d.transfer_to_infeed((y,))
    d.transfer_to_infeed((z,))
    d.transfer_to_infeed((w,))

    # JIT
    logging.info('Making jit call')
    res0 = jax.jit(f_for_jit)(x)
    self.assertAllClose(res0, x + y + z + w, check_dtypes=True)

    # PJIT
    def f_for_pjit(x):
      token = lax.create_token(x)
      # A replicated infeed
      (y,), token = lax.infeed(
          token,
          shape=(core.ShapedArray(x.shape, np.float32),),
          partitions=(None,))
      # An infeed sharded on first axis
      (z,), token = lax.infeed(
          token,
          shape=(core.ShapedArray(x.shape, np.float32),),
          partitions=(P(nr_devices, 1),))
      # An infeed sharded on second axis
      (w,), token = lax.infeed(
          token,
          shape=(core.ShapedArray(x.shape, np.float32),),
          partitions=(P(1, nr_devices),))
      return x + y + z + w

    logging.info('Transferring to infeed for the pjit call')
    for didx, d in enumerate(devices):
      # Transfer the whole array to all devices for replicated.
      d.transfer_to_infeed((y,))
      # For sharded infeed, transfer only the needed slices to each device.
      d.transfer_to_infeed(z[3 * didx:3 * didx + 3, :])
      d.transfer_to_infeed((w[:, 5 * didx:5 * didx + 5],))

    with jax.sharding.Mesh(devices, ['d']):
      logging.info('Making pjit call')
      res = pjit(f_for_pjit, in_shardings=(P('d'),), out_shardings=P('d'))(x)

    self.assertAllClose(res0, res, check_dtypes=True)

  def testOutfeed(self):
    if xla_bridge.using_pjrt_c_api():
      raise unittest.SkipTest('outfeed not implemented in PJRT C API')
    if config.use_shardy_partitioner.value:
      self.skipTest(
          'b/355263220: outfeed lowering not supported by Shardy')

    devices = np.array(jax.local_devices())
    nr_devices = len(devices)
    shape = (nr_devices * 3, nr_devices * 5)

    def f(x):
      token = lax.create_token(x)
      token = lax.outfeed(token, x, partitions=(None,))
      token = lax.outfeed(token, x, partitions=((nr_devices, 1),))
      token = lax.outfeed(token, x, partitions=((1, nr_devices),))
      return x

    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    def _dispatch():
      with jax.sharding.Mesh(devices, ['d']):
        logging.info('Making pjit call')
        pjit(f, in_shardings=(P('d'),), out_shardings=P('d'))(x)
    execution = threading.Thread(target=_dispatch)
    execution.start()

    # Check the expected outfeed for all devices.
    def check_outfeed(x_fn):
      for didx, d in enumerate(devices):
        x = x_fn(didx)
        y = d.transfer_from_outfeed(
            xc.Shape.array_shape(
                xc.PrimitiveType.F32, x.shape
            ).with_major_to_minor_layout_if_absent()
        )
        self.assertAllClose(x, y, check_dtypes=True)

    logging.info('Transferring from outfeed for the pjit call')

    # Note, when checking results of multiple outfeeds, the loop structure
    # should be such that we check a given outfeed for all devices before
    # moving on to the next outfeed. If there are any collectives generated
    # by pjit, a loop structutre like:
    #     for each device:
    #         check outfeed#0;
    #         check outfeed#1;
    #
    # Could cause a deadlock if there is a collective scheduled between the
    # 2 outfeeds, as device #0, after processing outfeed#0 will execute the
    # collective, waiting for other devices to join, but other devices won't
    # execute their collective until their outfeed#0 is executed. This is
    # because, for GPU for example, execution of an outfeed on GPU is blocked
    # till the corresponding `transfer_from_outfeed` is executed on the host.

    # Transfer the whole array from all devices for replicated.
    check_outfeed(lambda didx: x)
    # For sharded outfeed, the results are sliced.
    check_outfeed(lambda didx: x[3 * didx:3 * didx + 3, :])
    check_outfeed(lambda didx: x[:, 5 * didx:5 * didx + 5])

    execution.join()

  @jtu.with_mesh([('x', 2)])
  def testWithCustomPRNGKey(self):
    if not config.enable_custom_prng.value:
      raise unittest.SkipTest("test requires jax_enable_custom_prng")
    key = prng.random_seed(87, impl=prng.rbg_prng_impl)
    # Make sure this doesn't crash
    pjit(lambda x: x, in_shardings=None, out_shardings=None)(key)

  def test_lower_with_wrapper_error(self):
    @jax.jit
    def f(x):
      return x

    self.assertAllClose(1., f(1.))
    self.assertAllClose(1., f.lower(1.).compile()(1.))
    wrapped_f = wraps(f)(lambda x: f(x + 1))

    with self.assertRaisesRegex(AttributeError, "has no attribute 'lower'"):
      wrapped_f.lower(1.)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompile(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    expected = x @ (x + 1)

    lowered = f.lower(x, x + 1)
    compiled = lowered.compile()
    actual = compiled(x, x + 1)

    self.assertEqual(lowered.in_avals, compiled.in_avals)
    self.assertEqual(
        lowered.in_avals,
        ((core.ShapedArray(x.shape, x.dtype, weak_type=False),) * 2, {}))

    splits = np.split(expected, 4)
    self.assertAllClose(np.asarray(actual.addressable_shards[0].data), splits[0],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[1].data), splits[1],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[2].data), splits[2],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.addressable_shards[3].data), splits[3],
                        check_dtypes=False)

    for obj in [lowered, compiled]:
      self.assertFalse(obj._no_kwargs)
      self.assertEqual(obj.in_tree, jax.tree.flatten(((0, 0), {}))[1])

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileWithKwargs(self):
    @pjit
    def f(x, y, **kwargs):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    exe = f.lower(x, x + 1, a=1, b=2).compile()
    out = exe(x, x + 1, a=1, b=2)
    self.assertArraysEqual(out, x @ (x + 1))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileInTreeMismatch(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    exe = f.lower(x, x + 1).compile()

    self.assertRaisesRegex(
        TypeError,
        'Function compiled with input pytree does not match the input pytree it'
        ' was called with',
        lambda: exe([x], [x + 1]),
    )

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileArgTypeMismatch(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    exe = f.lower(x_f32, x_f32).compile()
    with self.assertRaisesRegex(
        TypeError,
        r"Argument types differ .*"
        r"The mismatches are:\n"
        r"Argument 'x' compiled with.*float32.*and called with.*int32.*\n"
        r"Argument 'y' compiled with.*float32.*and called with.*int32.*"):
      exe(x_i32, x_i32)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerAsText(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1)
    self.assertIsInstance(f.as_text(), str)
    self.assertIsInstance(f.as_text(dialect='hlo'), str)
    self.assertIsInstance(f.as_text(dialect='stablehlo'), str)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompilerIR(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1)
    self.assertIsNotNone(f.compiler_ir())
    self.assertIsNotNone(f.compiler_ir(dialect='hlo'))
    self.assertIsNotNone(f.compiler_ir(dialect='stablehlo'))

  @jtu.with_mesh([('x', 2)])
  def testLowerPartitionsAttribute(self):
    @partial(pjit,
             in_shardings=(P('x'), P('x')),
             out_shardings=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    hlo = f.lower(x, x + 1).as_text("stablehlo")
    self.assertIn("mhlo.num_replicas = 1", hlo)
    self.assertIn("mhlo.num_partitions = 2", hlo)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileCompilerIR(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    self.assertIsNotNone(f.runtime_executable())

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileAsText(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    self.assertIsInstance(f.as_text(), (str, type(None)))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCostAnalysis(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1)
    f.cost_analysis()  # doesn't raise

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileCostAnalysis(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    f.cost_analysis()  # doesn't raise

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileMemoryAnalysis(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    f.memory_analysis()  # doesn't raise

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileExecutable(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)

    f = f.lower(x, x + 1).compile()
    self.assertIsNotNone(f.runtime_executable())

  @jtu.with_mesh([('x', 2)])
  def test_static_argnums(self):
    @partial(pjit, in_shardings=None, out_shardings=None,
             static_argnums=(1,))
    def f(x, y):
      return x + (3 if y == 'hi' else 4)

    self.assertEqual(f(1, 'hi' ), 4)
    self.assertEqual(f(1, 'bye'), 5)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def testLowerCompileWithAvals(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    aval = core.ShapedArray(shape, dtypes.canonicalize_dtype(jnp.int64))
    x = jnp.arange(math.prod(shape)).reshape(shape)
    exe = f.lower(aval, x).compile()
    self.assertIsInstance(exe, stages.Compiled)
    self.assertArraysEqual(exe(x, x), x @ x)

  def test_local_sharded_key_array_sda(self):
    input_shape = (8, 4)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    seeds = jnp.arange(
        math.prod(input_shape), dtype=np.uint32).reshape(input_shape)

    with mesh:
      def make_keys(seeds):
        make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
        return make_key(seeds)

      f = pjit(make_keys, in_shardings=P(None), out_shardings=P(None))

      out = f(seeds)
      self.assertTrue(jax.dtypes.issubdtype(out.dtype, jax.dtypes.prng_key))
      self.assertEqual(out.shape, input_shape)
      jax.random.key_data(out)  # doesn't crash

  def test_with_sharding_constraint_is_compatible_error(self):
    mesh = jtu.create_mesh((1, 1, 2), ('replica', 'data', 'mdl'))

    with mesh:
      def f(x):
        y = with_sharding_constraint(x, P(None, ('mdl',), None, None))
        z = y + 2
        return z
      pjit_f = pjit(f, in_shardings=P(None), out_shardings=P(None))

      with self.assertRaisesRegex(
          ValueError,
          r"One of with_sharding_constraint.*Sharding "
          r"NamedSharding.*PartitionSpec\(None, 'mdl', None, None\).*\) is only "
          "valid for values of rank at least 4, but was applied to a value of rank 1"):
        pjit_f(jnp.array([1, 2, 3]))

  def test_pretty_print(self):
    f = pjit(lambda x: x**2)
    g = pjit(lambda x: f(x) + f(x))
    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(g)(x)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            let lambda = { lambda ; a:f32[1]. let b:f32[1] = integer_pow[y=2] a in (b,) } in
            { lambda ; c:f32[1]. let
                d:f32[1] = pjit[
                  name=<lambda>
                  jaxpr={ lambda ; c:f32[1]. let
                      e:f32[1] = pjit[name=<lambda> jaxpr=lambda] c
                      f:f32[1] = pjit[name=<lambda> jaxpr=lambda] c
                      d:f32[1] = add e f
                    in (d,) }
                ] c
              in (d,) }
        """).strip(),
    )

  def test_pretty_print_pjit_id(self):
    f = pjit(lambda x, y: x)
    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(lambda y: y + f(y, y))(x)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            { lambda ; a:f32[1]. let
                b:f32[1] = pjit[
                  name=<lambda>
                  jaxpr={ lambda ; a:f32[1] c:f32[1]. let  in (a,) }
                ] a a
                d:f32[1] = add a b
              in (d,) }
        """).strip(),
    )

  def test_pretty_print_with_constant_pjit_arg(self):
    f = pjit(lambda x, y: x * y)
    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(lambda x: f(x, np.float32(1.0)))(x)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            { lambda ; a:f32[1]. let
                b:f32[1] = pjit[
                  name=<lambda>
                  jaxpr={ lambda ; a:f32[1] c:f32[]. let b:f32[1] = mul a c in (b,) }
                ] a 1.0:f32[]
              in (b,) }
        """).strip(),
    )

  def test_pretty_print_with_aliased_args(self):
    f = pjit(lambda x, y, z: x * y * z)
    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(lambda x: f(x, x, x))(x)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            { lambda ; a:f32[1]. let
                b:f32[1] = pjit[
                  name=<lambda>
                  jaxpr={ lambda ; a:f32[1] c:f32[1] d:f32[1]. let
                      e:f32[1] = mul a c
                      b:f32[1] = mul e d
                    in (b,) }
                ] a a a
              in (b,) }
        """).strip(),
    )

  def test_pretty_print_with_literal_outvar(self):
    f = pjit(lambda x: (np.int32(2), x))
    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(f)(x)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            { lambda ; a:f32[1]. let
                b:i32[] c:f32[1] = pjit[
                  name=<lambda>
                  jaxpr={ lambda ; a:f32[1]. let  in (2:i32[], a) }
                ] a
              in (b, c) }
        """).strip(),
    )

  def test_pretty_print_with_closure(self):
    @pjit
    def g(x, y):
      @pjit
      def f(x):
        return x * y
      return f(x) + f(y)

    x = jnp.array([4.2], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(g)(x, x)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            let f = { lambda ; a:f32[1] b:f32[1]. let c:f32[1] = mul b a in (c,) } in
            { lambda ; d:f32[1] e:f32[1]. let
                g:f32[1] = pjit[
                  name=g
                  jaxpr={ lambda ; d:f32[1] e:f32[1]. let
                      h:f32[1] = pjit[name=f jaxpr=f] e d
                      i:f32[1] = pjit[name=f jaxpr=f] e e
                      g:f32[1] = add h i
                    in (g,) }
                ] d e
              in (g,) }
        """).strip(),
    )

  def test_pretty_print_with_name_clash(self):
    @pjit
    def g(x, y):
      @pjit
      def f(x):
        return x

      return f(x)*f(x) + f(y)*f(y)

    x = jnp.array([4.2], dtype=jnp.float32)
    y = jnp.array([4.2, 2.4], dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(g)(x, y)
    self.assertEqual(
        jaxpr.pretty_print(use_color=False),
        textwrap.dedent("""
            let f = { lambda ; a:f32[1]. let  in (a,) } in
            let f1 = { lambda ; b:f32[2]. let  in (b,) } in
            { lambda ; c:f32[1] d:f32[2]. let
                e:f32[2] = pjit[
                  name=g
                  jaxpr={ lambda ; c:f32[1] d:f32[2]. let
                      g:f32[1] = pjit[name=f jaxpr=f] c
                      h:f32[1] = pjit[name=f jaxpr=f] c
                      i:f32[1] = mul g h
                      j:f32[2] = pjit[name=f jaxpr=f1] d
                      k:f32[2] = pjit[name=f jaxpr=f1] d
                      l:f32[2] = mul j k
                      e:f32[2] = add i l
                    in (e,) }
                ] c d
              in (e,) }
            """).strip(),
    )

  def test_with_sharding_constraint_vmap_spmd_axis_name_error(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    def f(x):
      return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('x')))

    xs = jnp.arange(4 * 16.).reshape(4, 16)
    with self.assertRaisesRegex(ValueError, "spmd_axis_name"):
      jax.vmap(f, spmd_axis_name='x')(xs)

  def test_cache_bug(self):
    devices = list(jax.devices())
    if len(devices) < 2:
      raise unittest.SkipTest("Test requires 2 devices")

    def under_jvp(f):
      return jax.jvp(f, (), ())

    x0 = jnp.zeros(1, device=devices[0])
    x1 = jnp.zeros(1, device=devices[1])

    # comments describe how caches worked under the old `_most_recent_pjit_call_executable` system
    under_jvp(lambda: jnp.sin(x0)) # cpp_pjit miss, pjit_call_impl miss
    jnp.sin(x1)        # cpp_pjit miss, pjit_call_impl miss
    ans1 = jnp.sin(x0) # cpp_pjit miss, pjit_call_impl hit. Bad cpp_pjit entry created
    ans2 = jnp.sin(x0) # cpp_pjit hit with bad cache entry
    assert(ans1.devices() == ans2.devices())

  def test_zero_literal_equality(self):
    # This test verifies that we don't accidentally conflate positive and
    # negative zeros when deduplicating literals in the IR.
    f = jax.jit(lambda x: (x / np.float32(-0.0), x / np.float32(0.0)))
    a, b = f(np.float32(1.0))
    self.assertEqual(a, -np.inf)
    self.assertEqual(b, np.inf)
    ir = f.lower(np.float32(1.0)).as_text()
    self.assertIn("stablehlo.constant dense<0.000000e+00>", ir)
    self.assertIn("stablehlo.constant dense<-0.000000e+00>", ir)

  def test_device_put_copy_donate(self):
    x = np.arange(1000)
    y = jax.device_put(x, device=jax.devices()[0], may_alias=False, donate=False)
    z = jax.device_put(y, device=jax.devices()[0], may_alias=False, donate=False)
    a = jax.jit(lambda y: y * 2, donate_argnums=0)(y)
    self.assertDeleted(y)
    self.assertNotDeleted(z)
    self.assertArraysEqual(a, x * 2)


@jtu.pytest_mark_if_available('multiaccelerator')
class CustomPartitionerTest(jtu.JaxTestCase):

  def skip_if_custom_partitioning_not_supported(self):
    if jtu.is_cloud_tpu():
      raise unittest.SkipTest("Custom partitioning is not supported on libtpu.")

  @jtu.skip_on_devices('cpu')  # Collectives don't seem to work on CPU.
  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_custom_partitioner(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(precision, mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      result_sharding = result_shape[0].sharding
      self.assertEqual(arg_shardings[0], result_sharding)
      self.assertEqual(P('x', None), result_sharding.spec)
      self.assertEqual(P('y', None), arg_shardings[1].spec)

      def lower_fn(x, y):
        axis_name = arg_shardings[1].spec[0][0]
        i = jax.lax.axis_index(axis_name)
        # Use offset i * 0 instead of 0 to ensure that the two offsets have the
        # same dtype regardless the value of config.enable_x64.
        z = jax.lax.psum(
            jax.lax.dynamic_slice(x, (i * 0, i * 8), (8, 8)) @ y, (axis_name)
        )
        return z, z * z

      return mesh, lower_fn, (result_sharding, result_sharding), arg_shardings

    def infer_sharding_from_operands(precision, mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      x_shard, y_shard = arg_shardings
      x_shape, y_shape = arg_shapes
      x_names = tuple(x_shard.spec) + tuple(
          None for _ in range(len(x_shape.shape) - len(x_shard.spec)))
      y_names = tuple(y_shard.spec) + tuple(
          None for _ in range(len(y_shape.shape) - len(y_shard.spec)))
      z_shard = NamedSharding(y_shard.mesh, P(*(x_names[:-1] + y_names[1:])))
      return z_shard, z_shard

    @partial(custom_partitioning, static_argnums=(2,))
    def f(x, y, precision=None):
      z = jnp.matmul(x, y, precision=precision)
      return z, z * z

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=SdyShardingRule(operand_mappings=(('i', 'j'), ('j', 'k')), result_mappings=(('i', 'k'), ('i', 'k'))))

    pjit_f = pjit(f, in_shardings=(P('x'), P('y')), out_shardings=P('x'))
    x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
    y = np.asarray(np.random.randint(0, 20, (16, 32)), dtype=np.float32)
    result1 = jax.jit(f)(x, y)
    result2 = f(x, y)
    result0 = pjit_f(x, y)
    self.assertArraysEqual(result0, result1)
    self.assertArraysEqual(result1, result2)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_custom_partitioner_propagate_user_sharding(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      return (
          mesh,
          lower_fn,
          arg_shapes[0].sharding,
          (arg_shapes[0].sharding,),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return arg_shapes[0].sharding

    def propagate_user_sharding(mesh, user_shape):
      return user_shape.sharding

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        propagate_user_sharding=propagate_user_sharding,
        sharding_rule='i j -> i j',
    )

    def f2(a):
      return a + f(a)

    pjit_f = pjit(f2, in_shardings=(P(None, 'x')), out_shardings=P('x'))
    x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
    self.assertArraysEqual(x + x, pjit_f(x))

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_custom_partitioner_sharding_override(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      y_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(y_shard.mesh, P(None)),
          (NamedSharding(y_shard.mesh, P(None)),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      y_shard = arg_shapes[0].sharding
      return NamedSharding(y_shard.mesh, P('x'))

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=SdyShardingRule(operand_mappings=((BATCHING, 'i'),), result_mappings=((BATCHING, 'i'),)))

    pjit_f = pjit(f, in_shardings=(P(None, 'x')), out_shardings=P('x'))
    x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
    self.assertArraysEqual(x, pjit_f(x))

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_custom_partitioner_invalid_sharding(self):
    self.skip_if_custom_partitioning_not_supported()
    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      y_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(y_shard.mesh, P(None)),
          (NamedSharding(y_shard.mesh, P(None, 'x')),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      y_shard = arg_shapes[0].sharding
      return NamedSharding(y_shard.mesh, P('x'))

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='i j -> i j',
    )

    pjit_f = pjit(f, in_shardings=(P(None, 'x')), out_shardings=P('x'))
    x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)

    with self.assertRaisesRegex(Exception, 'Mismatch in result shapes.'):
      pjit_f(x).block_until_ready()

  @jtu.with_mesh([('x', 4)])
  def test_custom_partitioner_jit_annotated_function(self):
    """Test correct lowering of function with a @jax.jit annotated callee.

    Annotating a callee with @jax.jit results in a module with a HLO CallOp.
    This test is makes sure that the custom partitioner lowering supports
    CallOps.
    """

    self.skip_if_custom_partitioning_not_supported()

    @custom_partitioning
    def f(x):
      return x

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        @jax.jit
        def g(y):
          return y

        return g(x)

      x_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(x_shard.mesh, P('x')),
          (NamedSharding(x_shard.mesh, P('x')),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      x_shard = arg_shapes[0].sharding
      return NamedSharding(x_shard.mesh, P('x'))

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='i -> i',
    )

    jit_f = jax.jit(f)
    x = np.asarray(np.random.randint(0, 20, (32,)), dtype=np.float32)
    pjit_f = pjit(jit_f, in_shardings=(P('x')), out_shardings=P('x'))
    self.assertArraysEqual(x, pjit_f(x))

  @jtu.with_mesh([('x', 4)])
  def test_custom_partitioner_with_scan(self):
    self.skip_if_custom_partitioning_not_supported()

    # This is a reproducer from https://github.com/jax-ml/jax/issues/20864.

    @custom_partitioning
    def f(x):
      return jnp.sum(x)

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(xs):
        def f(carry, x):
          return carry + jax.lax.psum(jnp.sum(x), axis_name='x'), None

        carry, _ = jax.lax.scan(f, 0, xs)
        return carry

      result_shardings = jax.tree.map(lambda x: x.sharding, result_shape)
      arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
      return mesh, lower_fn, result_shardings, arg_shardings

    f.def_partition(
        partition,
        infer_sharding_from_operands=lambda mesh, *_: NamedSharding(mesh, P()),
        propagate_user_sharding=lambda _, user_shape: user_shape.sharding,
        sharding_rule='i j -> ')  # Result is a scalar.

    pjit_f = pjit(f, in_shardings=P(None, 'x'))
    xs = jnp.ones([32, 16])
    self.assertEqual(pjit_f(xs), xs.sum())

  def test_custom_partitioning_no_mesh_context(self):
    self.skip_if_custom_partitioning_not_supported()

    @custom_partitioning
    def f(x):
      return x

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        @jax.jit
        def g(y):
          return y

        return g(x)

      x_shard = arg_shapes[0].sharding
      return (
          mesh,
          lower_fn,
          NamedSharding(x_shard.mesh, P('x')),
          (NamedSharding(x_shard.mesh, P('x')),),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      x_shard = arg_shapes[0].sharding
      return NamedSharding(x_shard.mesh, P('x'))

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='i -> i',
    )

    mesh = jtu.create_mesh((4,), ('x',))
    x = np.asarray(np.random.randint(0, 20, (32,)), dtype=np.float32)
    s = NamedSharding(mesh, P('x'))

    jit_f = jax.jit(f, in_shardings=s, out_shardings=s)
    self.assertArraysEqual(x, jit_f(x))

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_custom_partitioner_pytree_inputs(self):
    self.skip_if_custom_partitioning_not_supported()

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(xs):
        x, y, z = xs
        return x + y + z

      return (
          mesh,
          lower_fn,
          arg_shapes[0][0].sharding,
          jax.tree.map(lambda x: x.sharding, arg_shapes),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return arg_shapes[0][0].sharding

    def propagate_user_sharding(mesh, user_shape):
      return user_shape.sharding

    @custom_partitioning
    def f(xs):
      x, y, z = xs
      return x + y + z

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        propagate_user_sharding=propagate_user_sharding,
        sharding_rule='i j, i j, i j -> i j',
    )

    def f2(a):
      return a + f((a, a, a))

    pjit_f = pjit(f2, in_shardings=(P(None, 'x')), out_shardings=P('x'))
    x = np.asarray(np.random.randint(0, 20, (32, 16)), dtype=np.float32)
    self.assertArraysEqual(x * 4, pjit_f(x))


@jtu.pytest_mark_if_available('multiaccelerator')
class AutoShardingPjitTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    ('2d_array', (4, 2), (4, 2), ('x', 'y')),
    # TODO(b/226977360): Support 3D mesh shape for example (2, 2, 2).
    ('3d_array', (1, 4, 2), (2, 4, 8, 4), ('x', 'y', 'z')),
    ('1d_array', (8,), (8, 2), ('x')),
  )
  def test_pjit_arr_auto_sharding_array(self, mesh_shape, global_input_shape,
                                        mesh_axis_names):
    if config.use_shardy_partitioner.value:
      self.skipTest('Must register auto partitioner for Shardy')
    global_mesh = jtu.create_mesh(mesh_shape, mesh_axis_names)
    input_data = np.arange(
        math.prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    f = jax.jit(lambda x: x, in_shardings=AUTO(global_mesh),
                out_shardings=AUTO(global_mesh))

    inp = core.ShapedArray(input_data.shape, input_data.dtype)
    compiled = f.lower(inp).compile()
    inputs = [create_array(global_input_shape, global_mesh, ip, input_data)[0]
              for ip in compiled.input_shardings[0]]
    out = compiled(*inputs)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out._value, input_data)

  def test_xla_arr_sharding_mismatch(self):
    if config.use_shardy_partitioner.value:
      self.skipTest('Must register auto partitioner for Shardy')
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    global_input_shape = (6, 2)
    input_data = np.arange(
        math.prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with global_mesh:
      f = pjit(lambda x: x, in_shardings=AUTO(global_mesh),
               out_shardings=AUTO(global_mesh))
      inp = core.ShapedArray(input_data.shape, input_data.dtype)
      compiled = f.lower(inp).compile()

      different_pspec = (
          P('y', 'x')
          if compiled.input_shardings[0][0].is_equivalent_to(
              NamedSharding(global_mesh, P('x', 'y')), len(global_input_shape)
          )
          else P('x', 'y')
      )
      arr, _ = create_array(global_input_shape, global_mesh, different_pspec,
                            input_data)
      with self.assertRaisesRegex(
          ValueError,
          r"Compiled object called with input sharding\(s\) does not match the "
          r"sharding\(s\) the computation was compiled with.*\n.*for arg x"):
        compiled(arr)

  def test_gda_auto_shardings_len(self):
    if config.use_shardy_partitioner.value:
      self.skipTest('Must register auto partitioner for Shardy')
    global_mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    global_input_shape = (4, 2)
    input_data = np.arange(
        math.prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with global_mesh:
      f = pjit(lambda x, y, z: (x, y, z), in_shardings=AUTO(global_mesh),
               out_shardings=AUTO(global_mesh))
      inp = core.ShapedArray(input_data.shape, input_data.dtype)
      compiled = f.lower(inp, inp, inp).compile()
      self.assertLen(compiled.output_shardings, 3)
      self.assertLen(compiled.input_shardings[0], 3)

  @parameterized.named_parameters(
    ('3d_array', (1, 1, 2), ('x', 'y', 'z'), P(('x', 'y', 'z'))),
    ('2d_array', (4, 2), ('x', 'y'), P('y', 'x')),
    ('1d_array', (8,), ('x'), P('x')),
  )
  def test_jit_arr_partial_auto_sharding_array(
      self, mesh_shape, mesh_axis_names, pspec):
    if config.use_shardy_partitioner.value:
      self.skipTest('Must register auto partitioner for Shardy')
    mesh = jtu.create_mesh(mesh_shape, mesh_axis_names)
    global_input_shape = (8, 4)
    input_data = np.arange(
        math.prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)
    inp_s = NamedSharding(mesh, pspec)
    f = jax.jit(
        lambda x, y: (x, y),
        in_shardings=(inp_s, AUTO(mesh)),
        out_shardings=AUTO(mesh))

    inp = core.ShapedArray(input_data.shape, input_data.dtype)
    compiled = f.lower(inp, inp).compile()
    inputs = [create_array(global_input_shape, mesh, ip, input_data)[0]
              for ip in compiled.input_shardings[0]]
    self.assertEqual(compiled.input_shardings[0][0], inp_s)
    out1, out2 = compiled(*inputs)
    for o in [out1, out2]:
      self.assertIsInstance(o, array.ArrayImpl)
      self.assertArraysEqual(o._value, input_data)

  def test_jit_different_mesh_in_auto(self):
    mesh1 = jtu.create_mesh((4,), ('x',))
    dev = jax.devices()
    mesh2 = jax.sharding.Mesh([dev[0], dev[3], dev[2], dev[1]], 'x')
    f = jax.jit(lambda x, y: (x, y),
                in_shardings=(NamedSharding(mesh2, P('x')), AUTO(mesh1)))
    inp = jax.ShapeDtypeStruct((8, 2), np.float32)
    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation"):
      f.lower(inp, inp).compile()

  @parameterized.named_parameters(
    ('2d_array', (4, 2), ('x', 'y')),
    ('1d_array', (8,), ('x')),
  )
  def test_jit_auto_sharding_partial_tuple_input_shardings(
      self, mesh_shape, mesh_axis_names):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest('Parameters are tupled only on TPU if >2000 parameters')
    if config.use_shardy_partitioner.value:
      self.skipTest('Must register auto partitioner for Shardy')

    mesh = jtu.create_mesh(mesh_shape, mesh_axis_names)
    global_input_shape = (8, 4)
    input_data = np.arange(
        math.prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)
    input_sharding = NamedSharding(mesh, P(mesh_axis_names)) # sharded
    input_sharding_annotations = [AUTO(mesh)] * 2001
    output_sharding = NamedSharding(mesh, P()) # replicated
    output_sharding_annotations = [AUTO(mesh)] * 2001
    for i in range(1000):
      input_sharding_annotations[2*i] = input_sharding
      output_sharding_annotations[2*i] = output_sharding

    jit_tuple_identity_fn = jax.jit(
        lambda *x: x,
        in_shardings=input_sharding_annotations,
        out_shardings=tuple(output_sharding_annotations))

    inp = core.ShapedArray(input_data.shape, input_data.dtype)
    compiled = jit_tuple_identity_fn.lower(*([inp] * 2001)).compile()


    # Check sharding preservation for even numbered inputs.
    for i in range(1000):
      self.assertEqual(compiled.input_shardings[0][2*i], input_sharding)
      self.assertEqual(compiled.output_shardings[2*i], output_sharding)

  @unittest.skip('The error is not raised yet. Enable this back once we raise '
                 'the error in pjit again.')
  def test_pjit_array_error(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    input_data = np.arange(
        math.prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with global_mesh:
      f = pjit(lambda x: x, in_shardings=AUTO(global_mesh),
               out_shardings=AUTO(global_mesh))

      inp = core.ShapedArray(input_data.shape, input_data.dtype)
      compiled = f.lower(inp).compile()
      inputs = [create_array(global_input_shape, global_mesh, ip, input_data)[0]
                for ip in compiled.input_shardings[0]]
      with self.assertRaisesRegex(
          ValueError,
          ('Passing sharding on pjit and on args while using the '
            'auto spmd partitioner is not allowed. Please call the '
            'compiled object on the inputs.')):
        f(*inputs)


@jtu.pytest_mark_if_available('multiaccelerator')
class ArrayPjitTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    ('fully_sharded_output', P('x', 'y'), (2, 4)),
    ('fully_replicated_output', P(None), (8, 8)),
  )
  def test_pjit_array_single_output(self, out_axis_resources, shard_shape):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, input_data = create_array(global_input_shape, global_mesh, mesh_axes)

    f = pjit(lambda x: x @ x.T, out_shardings=NamedSharding(
        global_mesh, out_axis_resources))
    expected_matrix_mul = input_data @ input_data.T

    out = f(input_array)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertTrue(out._committed)
    self.assertEqual(out.shape, (8, 8))
    self.assertEqual(out.addressable_shards[0].data.shape, shard_shape)
    for s in out.addressable_shards:
      self.assertLen(s.data.devices(), 1)
      self.assertArraysEqual(s.data, expected_matrix_mul[s.index])
    self.assertArraysEqual(out._value, expected_matrix_mul)

  @parameterized.named_parameters(
    ('fully_sharded_output', P('x', 'y'), (2, 4)),
    ('fully_replicated_output', P(None), (8, 8)),
  )
  def test_pjit_array_single_output_with_mesh_context_manager(
      self, out_axis_resources, shard_shape):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, input_data = create_array(global_input_shape, global_mesh, mesh_axes)

    with global_mesh:
      f = pjit(lambda x: x @ x.T, out_shardings=NamedSharding(
          global_mesh, out_axis_resources))
      expected_matrix_mul = input_data @ input_data.T

      out = f(input_array)
      self.assertIsInstance(out, array.ArrayImpl)
      self.assertEqual(out.shape, (8, 8))
      self.assertEqual(out.addressable_shards[0].data.shape, shard_shape)
      for s in out.addressable_shards:
        self.assertLen(s.data.devices(), 1)
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])
      self.assertArraysEqual(out._value, expected_matrix_mul)

  def test_numpy_array_input_assume_fully_replicated(self):
    input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    input_data = np.arange(
        math.prod(input_shape)).reshape(input_shape)

    f = pjit(lambda x: x,
              out_shardings=NamedSharding(global_mesh, P('x', 'y')))
    out = f(input_data)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, input_data)
    for s in out.addressable_shards:
      self.assertEqual(s.data.shape, (2, 1))
      self.assertArraysEqual(s.data, input_data[s.index])

  def test_numpy_array_input(self):
    input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    input_data = np.arange(
        math.prod(input_shape), dtype=np.float32).reshape(input_shape)
    with global_mesh:
      f = pjit(
          lambda x: x,
          in_shardings=NamedSharding(global_mesh, P(None)),
          out_shardings=NamedSharding(global_mesh, P('x', 'y')),
      )
      out = f(input_data)
      self.assertIsInstance(out, array.ArrayImpl)
      for s in out.addressable_shards:
        self.assertEqual(s.data.shape, (2, 1))
        self.assertArraysEqual(s.data, input_data[s.index])
      self.assertArraysEqual(out._value, input_data)

  def test_unspecified_out_axis_resources(self):

    def _checks(out, input_data):
      self.assertIsInstance(out, array.ArrayImpl)
      self.assertIsInstance(out.sharding, NamedSharding)
      self.assertEqual(out.shape, (8, 2))
      self.assertEqual(out.addressable_shards[0].data.shape, (2, 1))
      for s in out.addressable_shards:
        self.assertLen(s.data.devices(), 1)
        self.assertArraysEqual(s.data, input_data[s.index])
      self.assertArraysEqual(out._value, input_data)

    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, input_data = create_array(global_input_shape, global_mesh, mesh_axes)

    f = pjit(lambda x: x * 2)

    out = f(input_array)
    _checks(out, input_data * 2)

    out2 = f(out)
    _checks(out2, input_data * 4)

  @parameterized.named_parameters(
    ('mesh1', (4, 2), (2, 8), (2, 2), (1, 2), (8, 2)),
    ('mesh2', (2, 2), (4, 8), (4, 2), (2, 2), (8, 2)),
    ('mesh3', (2, 1), (4, 8), (4, 2), (4, 2), (8, 2)),
  )
  def test_pjit_array_multi_input_multi_output(self, mesh_shape, s1_shape,
                                               s2_shape, s3_shape, s4_shape):
    global_mesh = jtu.create_mesh(mesh_shape, ('x', 'y'))
    global_input_shape = (8, 2)

    spec1 = P('x', 'y')
    a1, input_data = create_array(global_input_shape, global_mesh, spec1)
    spec2 = P('x')
    a2, _ = create_array(global_input_shape, global_mesh, spec2)
    spec3 = P(('x', 'y'))
    a3, _ = create_array(global_input_shape, global_mesh, spec3)
    spec4 = P(None)
    a4, _ = create_array(global_input_shape, global_mesh, spec4)

    @pjit
    def f(tree):
      return tree
    out_tree = f((a1 @ a1.T, (a2, (a3 * 2, a4))))
    (out1, out2, out3, out4), _ = jax.tree.flatten(out_tree)

    self.assertIsInstance(out1, array.ArrayImpl)
    self.assertEqual(out1.shape, (8, 8))
    self.assertEqual(out1.addressable_shards[0].data.shape, s1_shape)
    for s in out1.addressable_shards:
      self.assertArraysEqual(
          s.data, (input_data @ input_data.T)[s.index])

    self.assertIsInstance(out2, array.ArrayImpl)
    self.assertEqual(out2.shape, (8, 2))
    self.assertEqual(out2.addressable_shards[0].data.shape, s2_shape)
    for s in out2.addressable_shards:
      self.assertArraysEqual(s.data, input_data[s.index])

    self.assertIsInstance(out3, array.ArrayImpl)
    self.assertEqual(out3.shape, (8, 2))
    self.assertEqual(out3.addressable_shards[0].data.shape, s3_shape)
    for s in out3.addressable_shards:
      self.assertArraysEqual(s.data, (input_data * 2)[s.index])

    self.assertIsInstance(out4, array.ArrayImpl)
    self.assertEqual(out4.shape, (8, 2))
    self.assertEqual(out4.addressable_shards[0].data.shape, s4_shape)
    for s in out4.addressable_shards:
      self.assertArraysEqual(s.data, input_data)

  def test_sds_full_like(self):
    # https://github.com/jax-ml/jax/issues/20390
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    x = jax.ShapeDtypeStruct((4, 4), jnp.float32, sharding=s)
    y = jnp.zeros_like(x)
    z = jnp.zeros_like(x, device=y.sharding)

    self.assertEqual(x.sharding, s)
    self.assertEqual(y.sharding, s)
    self.assertEqual(z.sharding, s)

  def test_in_axis_resources_mismatch_error(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, _ = create_array(global_input_shape, global_mesh, mesh_axes)

    with global_mesh:
      f = pjit(lambda x: x,
                in_shardings=NamedSharding(global_mesh, P('x')))
      err_msg = re.compile(
          "Sharding passed to pjit does not match the sharding on the "
          r"respective arg.*arg shape.*\[8,2\]", re.M | re.S)
      with self.assertRaisesRegex(ValueError, err_msg):
        f(input_array)

  def test_in_axis_resources_same_as_array_sharding(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, _ = create_array(global_input_shape, global_mesh, mesh_axes)

    with global_mesh:
      out = pjit(
          lambda x: x,
          in_shardings=NamedSharding(global_mesh, P('x' ,'y')))(input_array)
      self.assertIsInstance(out, array.ArrayImpl)

  def test_no_input_output(self):
    def f():
      pass
    pjit(f)

  def test_array_device_assignment_mismatch_with_mesh(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, _ = create_array(
        global_input_shape, jtu.create_mesh((2, 2), ('x', 'y')),
        mesh_axes)

    with global_mesh:
      with self.assertRaisesRegex(
          ValueError, "Received incompatible devices for jitted computation"):
        pjit(lambda x: x)(input_array)

  def test_array_lower_compile(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))

    a1, input_data = create_array(global_input_shape, global_mesh, P('x', 'y'))
    a2, _ = create_array(global_input_shape, global_mesh, P('x'))

    aval = core.ShapedArray(global_input_shape, np.float32)

    with global_mesh:
      f = pjit(
          lambda x, y, z, a, b, c: (x @ y.T, y, z, a, b, c),
          in_shardings=NamedSharding(global_mesh, P('x' ,'y')))
      compiled = f.lower(aval, aval, aval, aval, aval, aval).compile()
      out, *_ = compiled(a1, a1, a1, a1, a1, a1)
      self.assertIsInstance(out, array.ArrayImpl)
      self.assertArraysEqual(out._value, input_data @ input_data.T)

      with self.assertRaisesRegex(
          ValueError,
          r"Compiled object called with input sharding.*does not match the "
          r"sharding.*the computation was compiled with. "
          "Here are.*mismatches.*"):
        compiled(a2, a2, a2, a2, a2, a2)

    with global_mesh:
      f = pjit(lambda a: a, in_shardings=NamedSharding(global_mesh, P('x' ,'y')))
      abstract_inp = {'x': aval, 'y': {'y1': aval}}
      inp1 = {'x': a1, 'y': {'y1': a1}}
      compiled = f.lower(abstract_inp).compile()
      compiled(inp1)
      inp2 = {'x': a2, 'y': {'y1': a2}}
      with self.assertRaisesRegex(
          ValueError,
          r"Compiled object called with input sharding.*does not match the "
          r"sharding.*the computation was compiled with. "
          "Here are the.*mismatches"):
        compiled(inp2)

  def test_globally_sharded_key_array_result_8x4_single_device(self):
    input_shape = (8, 4)
    seeds = jnp.arange(
        math.prod(input_shape), dtype=np.uint32).reshape(input_shape)

    @pjit
    def make_keys(seeds):
      make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
      return make_key(seeds)

    out = make_keys(seeds)
    self.assertTrue(jax.dtypes.issubdtype(out.dtype, jax.dtypes.prng_key))
    self.assertEqual(out.shape, input_shape)
    jax.random.key_data(out)  # doesn't crash

  def test_globally_sharded_key_array_8x4_multi_device_with_out_sharding(self):
    input_shape = (8, 4)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    @partial(pjit, out_shardings=NamedSharding(mesh, P('x', 'y')))
    def make_keys(seeds):
      make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
      return make_key(seeds)

    out = make_keys(seeds)
    self.assertTrue(jax.dtypes.issubdtype(out.dtype, jax.dtypes.prng_key))
    self.assertEqual(out.shape, input_shape)
    jax.random.key_data(out)  # doesn't crash

  def test_globally_sharded_key_array_8x4_multi_device(self):
    input_shape = (8, 4)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    @pjit
    def make_keys(seeds):
      make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
      return make_key(seeds)

    out = make_keys(seeds)
    self.assertTrue(jax.dtypes.issubdtype(out.dtype, jax.dtypes.prng_key))
    self.assertEqual(out.shape, input_shape)
    jax.random.key_data(out)  # doesn't crash

  def test_array_device_assignment_mismatch_out_shardings(self):
    input_shape = (8, 2)
    m1 = jtu.create_mesh((4, 2), ('x', 'y'))
    m2 = jtu.create_mesh((2, 2), ('x', 'y'))
    spec = P('x', 'y')

    a1 = jnp.arange(math.prod(input_shape)).reshape(input_shape)

    with m1:
      with self.assertRaisesRegex(
          ValueError, "Received incompatible devices for jitted computation"):
        pjit(lambda x, y: (x, y),
              out_shardings=(NamedSharding(m1, spec),
                             NamedSharding(m2, spec)))(a1, a1)

  def test_array_device_assignment_mismatch_in_and_out_shardings(self):
    input_shape = (8, 2)
    m1 = jtu.create_mesh((4, 2), ('x', 'y'))
    m2 = jtu.create_mesh((2, 2), ('x', 'y'))
    spec = P('x', 'y')

    a1 = jnp.arange(math.prod(input_shape)).reshape(input_shape)

    with m1:
      with self.assertRaisesRegex(
          ValueError, "Received incompatible devices for jitted computation"):
        pjit(
            lambda x, y: (x, y),
            in_shardings=NamedSharding(m2, spec),
            out_shardings=NamedSharding(m1, spec),
        )(a1, a1)

  def test_mixed_inputs(self):
    input_shape = (8, 2)
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    a1, input_data = create_array(input_shape, global_mesh, spec)

    with global_mesh:
      f = pjit(lambda x, y: (x, y),
                in_shardings=NamedSharding(global_mesh, P(None)))
      with self.assertRaisesRegex(
          ValueError,
          ('Sharding passed to pjit does not match the sharding on the '
            'respective arg')):
        f(input_data, a1)

  def test_pjit_array_same_sharding_aot(self):
    global_mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    a1, _ = create_array(input_shape, global_mesh, P(None,))
    with global_mesh:
      f = pjit(lambda x: x, in_shardings=NamedSharding(global_mesh, P(None,)))
      compiled = f.lower(core.ShapedArray(input_shape, jnp.float32)).compile()
      compiled(a1)  # no error

  @jtu.thread_unsafe_test()  # cache_info isn't thread-safe
  def test_pjit_single_device_sharding_add(self):
    a = np.array([1, 2, 3], dtype=jnp.float32)
    b = np.array([4, 5, 6], dtype=jnp.float32)

    @pjit
    def add(x, y):
      return x + y

    out = add(a, b)
    cache_info1 = _pjit_lower_cached.cache_info()
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, a + b)
    self.assertFalse(out._committed)

    out2 = add(out, out)
    cache_info2 = _pjit_lower_cached.cache_info()
    self.assertIsInstance(out2, array.ArrayImpl)
    self.assertArraysEqual(out2, 2 * (a + b))
    self.assertFalse(out2._committed)

    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)

    c = jax.device_put(a, jax.devices()[0])
    out3 = add(c, c)
    cache_info3 = _pjit_lower_cached.cache_info()
    self.assertArraysEqual(out3, 2 * c)
    self.assertTrue(out3._committed)

    self.assertEqual(cache_info3.hits, cache_info2.hits)
    self.assertEqual(cache_info3.misses, cache_info2.misses + 1)

    out4 = add(out3, out3)
    self.assertArraysEqual(out4, 4 * c)
    self.assertTrue(out4._committed)

  def test_pjit_single_device_sharding_mul(self):
    a = jnp.arange(16).reshape((8, 2))

    @pjit
    def mul(x):
      return x @ x.T

    out = mul(a)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, a @ a.T)

  def test_pjit_single_device_sharding_cache(self):
    a = jnp.arange(16).reshape((8, 2))
    f = pjit(lambda x: x)

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = f(a)
      _ = f(out)
    self.assertEqual(count(), 1)

  def test_pjit_different_device_recompilation(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest('Requires 2 or more devices.')

    val1 = jnp.array([1, 2, 3], dtype=jnp.float32)
    a = jax.device_put(val1, jax.devices()[0])

    val2 = jnp.array([4, 5, 6], dtype=jnp.float32)
    b = jax.device_put(val2, jax.devices()[1])

    f = pjit(lambda x: x)

    with jtu.count_jit_compilation_cache_miss() as count:
      out1 = f(a)
      out2 = f(b)
    self.assertEqual(count(), 2)

    self.assertArraysEqual(out1, val1)
    self.assertArraysEqual(out2, val2)

  def test_grad_of_pjit_single_device_sharding(self):
    a = jnp.array(16, dtype=jnp.float32)
    f = lambda x: x * 3
    out = jax.grad(pjit(f))(a)
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, jax.grad(f)(a))

  def test_autodiff_with_single_device_sharding(self):
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    f = pjit(lambda x: x.sum(1) * h.sum())
    g = pjit(lambda x: f(jnp.sin(x * 4 + 2)))
    jtu.check_grads(g, (jnp.arange(16.).reshape((4, 4)) / 100,), order=2)

  def test_fast_path_array(self):
    devices = jax.devices()
    if len(devices) < 8:
      raise unittest.SkipTest("Test requires 8 global devices.")
    mesh_devices = np.array([[devices[0], devices[2]],
                             [devices[3], devices[1]],
                             [devices[4], devices[6]],
                             [devices[7], devices[5]]])
    shape = (8, 2)
    mesh = jax.sharding.Mesh(mesh_devices, ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    inp_data = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    # Explicitly put on the ordering of devices which does not match the mesh
    # ordering to make sure we reorder them in the constructor and the output
    # is correct.
    local_devices = jax.local_devices()[:8] # Take 8 local devices
    di_map = s.devices_indices_map(shape)
    bufs = [jax.device_put(inp_data[di_map[d]], d) for d in local_devices]
    arr = array.ArrayImpl(core.ShapedArray(shape, np.float32), s, bufs, committed=True)

    f = pjit(lambda x: x, out_shardings=s)
    out = f(arr)
    self.assertTrue(out.sharding.is_equivalent_to(arr.sharding, arr.ndim))
    self.assertArraysEqual(out, inp_data)
    out2 = f(out)
    self.assertTrue(out2.sharding.is_equivalent_to(out.sharding, out.ndim))
    self.assertArraysEqual(out2, inp_data)

  def test_array_enabled_non_empty_mesh_with_pspec(self):
    arr = jnp.array([1, 2, 3])
    with self.assertRaisesRegex(
        RuntimeError,
        r'jit requires a non-empty mesh if you are passing `PartitionSpec`s or'
        r' `None` to in_shardings.*'):
      pjit(lambda x: x, in_shardings=P('x'))(arr)

    with self.assertRaisesRegex(
        TypeError,
        "in_shardings leaf specifications are expected to be PartitionSpec "
        "instances or None, but got x"):
      pjit(lambda x: x, in_shardings='x')

  def test_pjit_uncommitted_array_reshard(self):
    arr = jnp.array([[1, 2, 3]])
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    with mesh:
      out = pjit(lambda x: x)(arr)
      self.assertArraysEqual(out, arr)
      self.assertLen(out.addressable_shards, 8)

  def test_pjit_uncommitted_array_in_axis_resources_reshard(self):
    arr = jnp.arange(16).reshape(8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    with mesh:
      out = pjit(lambda x: x, in_shardings=P('x', 'y'))(arr)
      self.assertArraysEqual(out, arr)
      self.assertLen(out.addressable_shards, 8)
      for s in out.addressable_shards:
        self.assertArraysEqual(s.data, arr[s.index])
        self.assertEqual(s.replica_id, 0)

  def test_pjit_uncommitted_array_and_committed_array(self):
    shape = (8, 2)
    uarr = jnp.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    carr, inp_data = create_array(shape, mesh, P('x', 'y'))
    with mesh:
      out1, out2 = pjit(lambda x, y: (x, y))(uarr, carr)
      self.assertArraysEqual(out1, inp_data)
      self.assertArraysEqual(out2, inp_data)
      self.assertLen(out1.addressable_shards, 8)
      self.assertLen(out2.addressable_shards, 8)

      mul_out = pjit(lambda x, y: x @ y.T)(uarr, carr)
      self.assertEqual(mul_out.shape, (8, 8))
      self.assertLen(mul_out.addressable_shards, 8)

    with jtu.create_mesh((2, 2), ('x', 'y')):
      with self.assertRaisesRegex(
          ValueError,
          "Received incompatible devices for jitted computation"):
        pjit(lambda x, y: (x, y))(uarr, carr)

  def test_pjit_uncommitted_array_multi_devices(self):
    shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    arr = array.ArrayImpl(
        core.ShapedArray(shape, np.int32), NamedSharding(mesh, P(None)),
        [jax.device_put(inp, d) for d in mesh.devices.flat], committed=False)
    with self.assertRaisesRegex(
        NotImplementedError,
        "Having uncommitted Array sharded on multiple devices is not supported."):
      pjit(lambda x: x)(arr)

  def test_pjit_committed_array_different_devices(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices')
    a = jax.device_put(np.array([1, 2, 3]), jax.devices()[0])
    b = jax.device_put(np.array([4, 5, 6]), jax.devices()[1])
    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation. Got argument "
        r"x of.*\<lambda\> with shape int.*\[3\] and device ids \[0\].*and "
        r"argument y of.*\<lambda\> with shape int.*\[3\] and device ids \[1\].*"):
      pjit(lambda x, y: (x, y))(a, b)

  def test_pjit_committed_array_different_devices_variadic_args(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices')
    a = jax.device_put(np.array([1, 2, 3]), jax.devices()[0])
    b = jax.device_put(np.array([4, 5, 6]), jax.devices()[1])
    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation. Got argument "
        r"x\[0\] of.*\<lambda\> with shape int.*\[3\] and device ids \[0\].*and "
        r"argument x\[1\] of.*\<lambda\> with shape int.*\[3\] and device ids "
        r"\[1\].*"):
      pjit(lambda *x: x)(a, b)

  def test_jit_no_forwarding(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @partial(jax.jit, donate_argnums=(0,))
    def f(x):
      return x, x * 2

    x = jax.device_put(jnp.zeros(64, dtype="int32"), NamedSharding(mesh, P()))
    jaxpr = jax.make_jaxpr(f)(x)
    y = core.jaxpr_as_fun(jaxpr)(x)
    self.assertTrue(x.is_deleted())
    self.assertFalse(y[0].is_deleted())
    self.assertFalse(y[1].is_deleted())

  def test_pjit_pytree_inp_device_assignment_mismatch(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    a = jax.device_put(np.array([1, 2, 3]), jax.devices()[0])
    b = jax.device_put(np.array([4, 5, 6]), jax.devices()[1])
    c = jax.device_put(np.arange(16).reshape(8, 2),
                       NamedSharding(mesh, P('x', 'y')))

    msg = ("Received incompatible devices for jitted computation. Got "
           r"argument {} of.*<lambda> with shape int.*\[3\] and device ids "
           r"\[0\].*and argument {} of.*<lambda> with shape int.*\[8,2\] and "
           r"device ids.*")

    with self.assertRaisesRegex(
        ValueError, msg.format(r'tuple_inp\[0\]', r'tuple_inp\[1\]\[0\]')):
      pjit(lambda tuple_inp: tuple_inp)((a, (c, (b))))

    with self.assertRaisesRegex(
        ValueError, msg.format(r"dict_inp\['a'\]\['b'\]\['c'\]",
                               r"dict_inp\['a'\]\['b'\]\['g'\]")):
      inp = {'d': a, 'z': a, 'a': {'f': a, 'y': b, 'b': {'g': c, 'c': a}}}
      pjit(lambda dict_inp: dict_inp)(inp)

  def test_same_out_sharding_id(self):
    shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    arr, inp_data = create_array(shape, mesh, P('x', 'y'))

    f = pjit(lambda x: x)
    out1 = f(arr)
    self.assertArraysEqual(out1, inp_data)
    out1_sharding_id = id(out1.sharding)

    out2 = f(out1)
    self.assertArraysEqual(out2, inp_data)
    out2_sharding_id = id(out2.sharding)

    out3 = f(out2)
    self.assertArraysEqual(out3, inp_data)
    out3_sharding_id = id(out3.sharding)

    self.assertEqual(out1_sharding_id, out2_sharding_id)
    self.assertEqual(out1_sharding_id, out3_sharding_id)
    self.assertEqual(out2_sharding_id, out3_sharding_id)

  @jtu.thread_unsafe_test()  # cache_info isn't thread-safe
  def test_out_sharding_indices_id_cache_hit(self):
    shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    arr, _ = create_array(shape, mesh, P('x', 'y'))

    f = pjit(lambda x: x)
    out1 = f(arr)
    self.assertIsInstance(out1.sharding, NamedSharding)
    out1.sharding.devices_indices_map(shape)
    cache_info1 = common_devices_indices_map.cache_info()

    out2 = f(out1)
    self.assertIsInstance(out2.sharding, NamedSharding)
    out2.sharding.devices_indices_map(shape)
    cache_info2 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)

    out3 = f(out2)
    self.assertIsInstance(out3.sharding, NamedSharding)
    out3.sharding.devices_indices_map(shape)
    cache_info3 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info3.hits, cache_info2.hits + 1)

  def test_aot_compile_in_tree_mismatch(self):
    @jax.jit
    def f(tree):
      return tree

    tree1 = {'a': {'c': 5, 'd': 6}}
    tree2 = {'a': 1, 'c': {'b': 5, 'e': 7}}
    with self.assertRaisesRegex(
        TypeError,
        'Function compiled with input pytree does not match the input pytree it'
        ' was called with'):
      f.lower(tree1).compile()(tree2)

  @jax.enable_custom_prng()
  def test_device_put_sharding_prng(self):
    mesh = jtu.create_mesh((8,), ('x',))
    s = NamedSharding(mesh, P('x'))

    x = jax.random.split(jax.random.PRNGKey(0), len(jax.devices()))
    y = jax.device_put(x, s)
    self.assertTrue(jax.dtypes.issubdtype(y.dtype, jax.dtypes.prng_key))
    self.assertEqual(y.sharding, s)

    s1 = SingleDeviceSharding(jax.devices()[1])
    z = jax.device_put(x, s1)
    self.assertTrue(jax.dtypes.issubdtype(z.dtype, jax.dtypes.prng_key))
    self.assertEqual(z.sharding, s1)

    out_p = jax.pmap(lambda x: x)(np.arange(jax.device_count()))
    a = jax.device_put(x, out_p.sharding)
    self.assertTrue(jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key))
    self.assertEqual(a.sharding, out_p.sharding)

    if config.use_shardy_partitioner.value:
      # OpSharding is not supported in shardy.
      return

    op = xc.OpSharding()
    op.type = xc.OpSharding.Type.OTHER
    op.tile_assignment_dimensions = [8]
    op.tile_assignment_devices = [0, 1, 2, 3, 4, 5, 6, 7]
    gs = GSPMDSharding(tuple(mesh.devices.flat), op)
    b = jax.device_put(x, gs)
    self.assertTrue(jax.dtypes.issubdtype(b.dtype, jax.dtypes.prng_key))
    self.assertEqual(b.sharding, gs)

  def test_device_put_on_different_sharding(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))

    x = jnp.arange(8).reshape(4, 2)
    s1 = NamedSharding(mesh, P('x'))
    a = jax.device_put(x, s1)
    self.assertEqual(a.sharding, s1)

    s2 = NamedSharding(mesh, P('x', 'y'))
    b = jax.device_put(a, s2)
    self.assertEqual(b.sharding, s2)

  def test_with_sharding_constraint_jit(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    @partial(jax.jit, static_argnums=(0, 1))
    def sharded_zeros(shape, pspec):
      out = jnp.zeros(shape, jnp.bfloat16)
      return jax.lax.with_sharding_constraint(out, NamedSharding(mesh, pspec))

    out = sharded_zeros((4096, 3072), P('x', 'y'))
    out_s = NamedSharding(mesh, P('x', 'y'))
    self.assertTrue(op_shardings.are_op_shardings_equal(
        out.sharding._to_xla_hlo_sharding(out.ndim),
        out_s._to_xla_hlo_sharding(out.ndim)))

  def test_with_sharding_constraint_pjit(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    @partial(pjit, static_argnums=(0, 1))
    def sharded_zeros(shape, pspec):
      out = jnp.zeros(shape, jnp.bfloat16)
      return jax.lax.with_sharding_constraint(out, NamedSharding(mesh, pspec))

    out = sharded_zeros((4096, 3072), P('x', 'y'))
    out_s = NamedSharding(mesh, P('x', 'y'))
    self.assertTrue(op_shardings.are_op_shardings_equal(
        out.sharding._to_xla_hlo_sharding(out.ndim),
        out_s._to_xla_hlo_sharding(out.ndim)))

  def test_jit_with_sharding_constraint_committed_inp_error(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    s = NamedSharding(mesh, P('x', 'y'))

    @jax.jit
    def sharded_inp(inp):
      return jax.lax.with_sharding_constraint(
          inp, NamedSharding(mesh, P('x', 'y')))

    committed_inp = jax.device_put(jnp.zeros((8, 2), jnp.bfloat16), jax.devices()[0])
    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation. Got argument "
        r"inp of.*sharded_inp with shape bfloat16\[8,2\] and device ids \[0\].*"
        r"sharding_constraint inside jit with device ids.*"):
      sharded_inp(committed_inp)

    @pjit
    def my_nested_pjit(inp1, inp2, inp3):
      @partial(pjit, in_shardings=(s, s, s),
               out_shardings=(s, s, s))
      def f(x, y, z):
        return x * 2, y * 2, z * 2
      return f(inp1, inp2, inp3)
    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation. Got argument "
        r"inp1 of.*my_nested_pjit with shape bfloat16\[8,2\] and device ids \[0\].*"
        r"pjit inside jit with device ids.*"):
      my_nested_pjit(committed_inp, committed_inp, committed_inp)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def test_jit_device_with_sharding_constraint_error(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    @partial(jax.jit, static_argnums=(0, 1), device=jax.devices()[0])
    def sharded_zeros(shape, pspec):
      out = jnp.zeros(shape, jnp.bfloat16)
      return jax.lax.with_sharding_constraint(out, NamedSharding(mesh, pspec))

    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation. Got explicit "
        r"output sharding with device ids \[0\].*sharding_constraint inside "
        r"jit with device ids.*"):
      sharded_zeros((4096, 3072), P('x', 'y'))

  def test_concurrent_pjit(self):
    global_mesh = jtu.create_mesh((1,), ('x',))
    sharding = NamedSharding(global_mesh, P('x',))
    n = 10
    with global_mesh:
      fs = [pjit(lambda x, i: x + i, static_argnums=1) for _ in range(n)]

      def _invoke_with_mesh_twice(arg_tuple):
        f, x, i = arg_tuple
        with global_mesh:
          f(x, i)
          return f(x, i)

      xs = [
          array.make_array_from_callback(
              (i,), sharding, lambda idx: np.arange(i, dtype=np.float32))
          for i in range(n)
      ]
      with concurrent.futures.ThreadPoolExecutor() as executor:
        ys = executor.map(_invoke_with_mesh_twice,
                          [(fs[i], x, i) for i, x in enumerate(xs)])
      for i, x, y in zip(range(n), xs, ys):
        self.assertAllClose(x + i, y)

  def test_trivial_computation(self):
    shape = (8, 2)
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    inp_data = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(inp_data, s)
    out = pjit(lambda x: x)(arr)
    self.assertArraysEqual(out, inp_data)

  def test_trivial_computation_with_sharded_const(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    const = jax.device_put(np.arange(16).reshape(8, 2),
                           NamedSharding(mesh, P('x', 'y')))
    with mesh:
      out = pjit(lambda: const)()
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, np.arange(16).reshape(8, 2))

  def test_trivial_computation_with_sharded_const_using_transposed_mesh(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    const = jax.device_put(np.arange(16).reshape(8, 2),
                           NamedSharding(mesh, P('x', 'y')))
    mesh2 = jtu.create_mesh((1, 2), ('x', 'y'))
    with mesh2:
      out = pjit(lambda: const)()
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertArraysEqual(out, np.arange(16).reshape(8, 2))

  def test_trivial_computation_with_replicated_literal(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    with mesh:
      out = pjit(lambda: 1)()
    self.assertEqual(out.sharding, NamedSharding(mesh, P()))
    self.assertIsInstance(out, array.ArrayImpl)
    self.assertEqual(out, 1)

  def test_multi_device_pjit_mul(self):
    shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    inp_data = np.arange(math.prod(shape)).reshape(shape)
    arr1 = jax.device_put(inp_data, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(inp_data, NamedSharding(mesh, P(None, 'y')))

    out1, out2 = pjit(lambda x, y: (x @ x.T, y * 2))(arr1, arr2)

    self.assertArraysEqual(out1, inp_data @ inp_data.T)
    self.assertEqual(out1.shape, (8, 8))
    self.assertArraysEqual(out2, inp_data * 2)
    self.assertEqual(out2.shape, (8, 2))

  def test_single_device_pjit_cpp_dispatch(self):
    shape = (8, 2)
    mesh = jtu.create_mesh((1,), ('x',))
    inp_data = np.arange(math.prod(shape)).reshape(shape)

    f = pjit(lambda x: x @ x.T, in_shardings=None, out_shardings=None)
    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(10):
        arr1 = jax.device_put(
            inp_data, jax.sharding.NamedSharding(mesh, P('x')))
        with mesh:
          f(arr1)
    self.assertEqual(count(), 1)

  def test_single_device_add_single_compile(self):
    f1 = pjit(lambda x, y: x + y)
    a = jax.device_put(jnp.array([1, 2, 3], dtype=jnp.float32),
                       jax.devices()[0])
    b = jax.device_put(jnp.array([4, 5, 6], dtype=jnp.float32),
                       jax.devices()[0])

    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(2):
        f1(a, b)
    self.assertEqual(count(), 1)

  def test_global_array_to_host_local_array_already_host_local(self):
    inp_shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    pspec = P('x', 'y')

    arr, _ = create_array(inp_shape, mesh, pspec)
    out = multihost_utils.global_array_to_host_local_array(arr, mesh, pspec)
    self.assertEqual(id(arr), id(out))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileWithStaticArguments(self):
    @partial(pjit,
             in_shardings=P(('x', 'y'),),
             out_shardings=P(('x', 'y'),), static_argnums=0)
    def f(c, x):
      return x if c == 0 else x + 1

    shape = (8, 8)
    x = jnp.arange(math.prod(shape)).reshape(shape)
    exe = f.lower(1, x).compile()

    self.assertAllClose(exe(x), x + 1, check_dtypes=False)

  def test_vmap_of_jvp_pjit_no_axis_resources(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    pjit_inp1 = jax.device_put(
        jnp.arange(8.), jax.sharding.NamedSharding(mesh, P('x')))
    pjit_inp2 = jax.device_put(
        jnp.arange(8.), jax.sharding.NamedSharding(mesh, P(('x', 'y'))))

    def f_(x, n):
      if n == 0:
        return x * 2.
      return jax.jit(partial(f_, n=n-1))(x - 1)
    f = jax.jit(partial(f_, n=5))
    jit_out1, jit_out2 = jax.vmap(lambda xs, ts: jax.jvp(f, xs, ts))(
        (jnp.arange(8.),), (jnp.arange(8.),))

    def g_(x, n):
      if n == 0:
        return x * 2.
      return pjit(partial(g_, n=n - 1))(x - 1)
    g = pjit(partial(g_, n=5))
    pjit_out1, pjit_out2 = jax.vmap(lambda xs, ts: jax.jvp(g, xs, ts))(
        (pjit_inp1,), (pjit_inp2,))

    self.assertArraysEqual(pjit_out1, jit_out1)
    self.assertArraysEqual(pjit_out2, jit_out2)

  def test_vmap_of_jvp_pjit_no_axis_resources_2d(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    f_inp = jnp.arange(8.).reshape(2, 2, 2)

    # g_inp is sharded with P(None, 'x') because f_inp is sharded with P('x')
    # and then `f` will get vmapped and pjit's batching rule will insert a
    # replicated axis for the batched dimension converting it into P(None, 'x')
    g_inp = jax.device_put(f_inp,
                           jax.sharding.NamedSharding(mesh, P(None, 'x')))

    # Reference pjit with axis_resources
    def f_(x, n):
      if n == 0:
        return x * 2.
      return pjit(
          partial(f_, n=n - 1), in_shardings=P('x'), out_shardings=P('x')
      )(x - 1)
    f = pjit(partial(f_, n=5), in_shardings=P('x'), out_shardings=P('x'))
    with mesh:
      f_out1, f_out2 = jax.vmap(lambda xs, ts: jax.jvp(f, xs, ts))(
          (f_inp,), (f_inp,))

    # pjit with no axis_resources
    def g_(x, n):
      if n == 0:
        return x * 2.
      return pjit(partial(g_, n=n - 1))(x - 1)
    g = pjit(partial(g_, n=5))
    g_out1, g_out2 = jax.vmap(lambda xs, ts: jax.jvp(g, xs, ts))(
        (g_inp,), (g_inp,))

    self.assertArraysEqual(f_out1, g_out1)
    self.assertArraysEqual(f_out2, g_out2)
    self.assertEqual(f_out1.sharding, g_out1.sharding)
    self.assertEqual(f_out2.sharding, g_out2.sharding)

  def test_pjit_on_different_default_device_with_uncommitted_inputs(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >= 2 devices')

    @pjit
    def f(x, y):
      return x + y

    a = jnp.array([1, 2, 3], dtype=jnp.float32)
    self.assertFalse(a._committed)
    out = f(a, a)
    self.assertFalse(out._committed)
    self.assertEqual(out.devices(), {jax.devices()[0]})
    self.assertArraysEqual(out, a * 2)

    with jax.default_device(jax.devices()[1]):
      b = jnp.array([4, 5, 6], dtype=jnp.float32)
      self.assertFalse(b._committed)
      out2 = f(b, b)
      self.assertFalse(out2._committed)
      self.assertEqual(out2.devices(), {jax.devices()[1]})
      self.assertArraysEqual(out2, b * 2)

  def test_pjit_with_static_argnames(self):

    def f(x: str) -> int:
      assert x == 'foo'
      return 1

    f_nums = pjit(f, static_argnums=0)
    assert f_nums('foo') == 1
    assert f_nums(x='foo') == 1

    f_names = pjit(f, static_argnames='x')
    assert f_names('foo') == 1
    assert f_names(x='foo') == 1

  def test_pjit_with_static_argnames_cpp_dispatch(self):
    def f(y, **kwargs):
      self.assertEqual(kwargs, {'x': 'foo'})
      return y * y

    y = jnp.arange(8.)
    with jtu.count_pjit_cpp_cache_miss() as count:
      f_names = pjit(f, static_argnames='x')
      f_names(y, x='foo')
      f_names(y, x='foo')
    self.assertEqual(count(), 1)

  def test_new_static_argnum_on_keyword_arguments(self):
    f = pjit(lambda x: x, static_argnums=0)
    y = f(x=4)
    assert y == 4

  def test_new_static_argnum_with_default_arguments(self):
    f = pjit(lambda x=4: x, static_argnums=0)
    y = f()
    assert y == 4

  def test_pjit_different_default_device(self):
    if jax.device_count() <= 1:
      self.skipTest('Test requires more >1 device.')

    system_default_device = list(jnp.add(1, 1).devices())[0]
    test_device = jax.devices()[-1]

    f = pjit(lambda x: x + 1)

    f(1)
    with jax.default_device(system_default_device):
      f(1)
    with jax.default_device(test_device):
      f(1)

    with jtu.count_pjit_cpp_cache_miss() as count:
      f(1)

      with jax.default_device(system_default_device):
        f(1)

      with jax.default_device(test_device):
        f(1)

      with jax.default_device(test_device):
        with jax.default_device(system_default_device):
          f(1)

    # The count here is 0 because before `count_pjit_cpp_cache_miss`, `f` was
    # called with `system_default_device` and `test_device` so it was added
    # to the cache. Subsequent calls hit the C++ cache.
    self.assertEqual(count(), 0)

  def test_pjit_with_mismatched_static_argnames(self):
    x_is_tracer, y_is_tracer = False, False
    def f(x, y):
      assert isinstance(x, core.Tracer) == x_is_tracer
      assert isinstance(y, core.Tracer) == y_is_tracer
      return 1

    # If both static_argnums and static_argnames are provided, they are allowed
    # to disagree and `jit` will respect the user's choices.
    f_nums = pjit(f, static_argnums=1, static_argnames=())
    x_is_tracer, y_is_tracer = True, False
    assert f_nums(2, 3) == 1
    x_is_tracer, y_is_tracer = True, True
    assert f_nums(1, y=2) == 1

    f_names = pjit(f, static_argnums=(), static_argnames='y')
    x_is_tracer, y_is_tracer = True, True
    assert f_names(2, 3) == 1
    x_is_tracer, y_is_tracer = True, False
    assert f_names(1, y=3) == 1

    f_mixed = pjit(f, static_argnums=(1,), static_argnames='x')
    x_is_tracer, y_is_tracer = True, False
    assert f_mixed(2, 3) == 1
    x_is_tracer, y_is_tracer = True, True
    assert f_mixed(1, y=3) == 1
    x_is_tracer, y_is_tracer = False, True
    assert f_mixed(x=2, y=3) == 1

  @jtu.thread_unsafe_test()  # cache_info isn't thread-safe
  def test_pjit_kwargs(self):
    a = jnp.arange(8.)
    b = jnp.arange(4.)
    c = jnp.arange(2.)

    @pjit
    def f(x, y, z):
      return x, y, z

    o1, o2, o3 = f(a, y=b, z=c)
    cache_info1 = pxla._cached_lowering_to_hlo.cache_info()
    self.assertArraysEqual(o1, a)
    self.assertArraysEqual(o2, b)
    self.assertArraysEqual(o3, c)

    o4, o5, o6 = f(x=a, y=b, z=c)
    cache_info2 = pxla._cached_lowering_to_hlo.cache_info()
    self.assertArraysEqual(o4, a)
    self.assertArraysEqual(o5, b)
    self.assertArraysEqual(o6, c)

    self.assertEqual(cache_info2.hits, cache_info1.hits)
    self.assertEqual(cache_info2.misses, cache_info1.misses + 1)

    o7, o8, o9 = f(a, b, c)
    cache_info3 = pxla._cached_lowering_to_hlo.cache_info()
    self.assertArraysEqual(o7, a)
    self.assertArraysEqual(o8, b)
    self.assertArraysEqual(o9, c)

    self.assertEqual(cache_info3.hits, cache_info2.hits)
    self.assertEqual(cache_info3.misses, cache_info2.misses + 1)

  def test_pjit_kwargs_axis_resources_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "pjit does not support kwargs when in_shardings is specified."):
      pjit(lambda x: x,
           in_shardings=SingleDeviceSharding(jax.devices()[0]))(x=jnp.arange(8.))

  def test_pjit_keep_unused_true(self):
    @partial(pjit, keep_unused=True)
    def f(x, y, z, a, b, c):  # pylint: disable=unused-argument
      return c @ c.T

    inp = jnp.arange(4)
    unused_inp = jnp.arange(8)

    out = f(unused_inp, unused_inp, unused_inp, unused_inp, unused_inp, inp)
    # Run it again to take the C++ dispatch.
    out_again = f(unused_inp, unused_inp, unused_inp, unused_inp, unused_inp, inp)

    self.assertArraysEqual(out, inp @ inp.T)
    self.assertArraysEqual(out_again, inp @ inp.T)

    compiled = f.lower(
        unused_inp, unused_inp, unused_inp, unused_inp, unused_inp, inp).compile()
    self.assertEqual(compiled._executable._kept_var_idx, {0, 1, 2, 3, 4, 5})
    self.assertLen(compiled._executable.in_avals, 6)

  def test_pjit_keep_unused_default_false(self):
    @pjit
    def f(x, y, z, a, b, c):  # pylint: disable=unused-argument
      return c @ c.T

    inp = jax.device_put(jnp.arange(4), jax.devices()[0])
    unused_inp = jax.device_put(jnp.arange(8), jax.devices()[0])

    out = f(unused_inp, unused_inp, unused_inp, unused_inp, unused_inp, inp)
    # Run it again to take the C++ dispatch.
    out_again = f(unused_inp, unused_inp, unused_inp, unused_inp, unused_inp, inp)

    self.assertArraysEqual(out, inp @ inp.T)
    self.assertArraysEqual(out_again, inp @ inp.T)

    compiled = f.lower(
        unused_inp, unused_inp, unused_inp, unused_inp, unused_inp, inp).compile()
    self.assertEqual(compiled._executable._kept_var_idx, {5})
    self.assertLen(compiled._executable.in_avals, 1)

  def test_pjit_relayout_multi_slice(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    @jax.jit
    def mul(x):
      return x @ x.T

    x = jnp.arange(8).reshape(4, 2)
    y = jax.device_put(x, jax.sharding.NamedSharding(mesh, P('x', 'y')))
    compiled = mul.lower(jax.ShapeDtypeStruct(
        y.shape, y.dtype, sharding=y.sharding)).compile()
    out = compiled(y)
    self.assertArraysEqual(out, x @ x.T)

  def test_pjit_with_device_arg(self):
    def mul(x):
      return x @ x.T

    def _check(out, expected_device, expected_out):
      self.assertEqual(out.devices(), {expected_device})
      self.assertLen(out.sharding.device_set, 1)
      self.assertArraysEqual(out, expected_out @ expected_out.T)

    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      f = pjit(mul, device=jax.devices()[1])

    x = jnp.arange(8).reshape(4, 2)
    f_out = f(x)
    f_out2 = f(f_out)
    _check(f_out, jax.devices()[1], x)
    _check(f_out2, jax.devices()[1], f_out)

    y = jax.device_put(x, jax.sharding.NamedSharding(mesh, P('x', 'y')))
    out2 = f(y)
    _check(out2, jax.devices()[1], y)

    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      h = pjit(mul, device=jax.devices()[-1])
    h_out = h(y)
    _check(h_out, jax.devices()[-1], y)

    # AOT test
    compiled = f.lower(core.ShapedArray(y.shape, y.dtype)).compile()
    out3 = compiled(y)
    _check(out3, jax.devices()[1], y)

  def test_pjit_with_device_arg_input_from_another_pjit(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    inp = np.arange(8).reshape(4, 2)

    y = jax.device_put(inp, jax.sharding.NamedSharding(mesh, P('x', 'y')))
    out = pjit(lambda x: x * 2)(y)

    expected_device = jax.devices()[2]
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      final_out = pjit(lambda x: x * 3, device=expected_device)(out)

    self.assertEqual(final_out.devices(), {expected_device})
    self.assertLen(final_out.sharding.device_set, 1)
    self.assertArraysEqual(final_out, inp * 6)

  @jtu.run_on_devices("tpu")
  def test_pjit_with_backend_arg(self):
    def _check(out, expected_device, expected_out):
      self.assertEqual(out.devices(), {expected_device})
      self.assertLen(out.sharding.device_set, 1)
      self.assertArraysEqual(out, expected_out)

    x = jnp.arange(8)
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      g = pjit(lambda x: x, backend='tpu')
    g_out = g(x)
    _check(g_out, jax.devices()[0], x)

    compiled = g.lower(core.ShapedArray(x.shape, x.dtype)).compile()
    out4 = compiled(x)
    _check(out4, jax.devices()[0], x)

  def test_autodiff_with_device_arg(self):
    if jax.device_count() <= 1:
      self.skipTest('Test requires more >1 device.')
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      f = pjit(lambda x: x.sum(1) * h.sum(), device=jax.devices()[1])
      g = pjit(lambda x: f(jnp.sin(x * 4 + 2)), device=jax.devices()[1])
    jtu.check_grads(g, (jnp.arange(16.).reshape((4, 4)) / 100,), order=2)

  def test_pjit_device_backend_axis_resources_error(self):
    s = SingleDeviceSharding(jax.devices()[0])
    with self.assertRaisesRegex(
        ValueError,
        'If backend or device is specified on jit, then '
        'in_shardings should not be specified.'):
      with jtu.ignore_warning(category=DeprecationWarning,
                              message="backend and device argument"):
        pjit(lambda x: x, in_shardings=s, backend='cpu')

    with self.assertRaisesRegex(
        ValueError,
        'If backend or device is specified on jit, then '
        'out_shardings should not be specified.'):
      with jtu.ignore_warning(category=DeprecationWarning,
                              message="backend and device argument"):
        pjit(lambda x: x, out_shardings=s, device=jax.devices()[0])

  def test_check_arg_error(self):
    sds = jax.ShapeDtypeStruct((4, 2), np.int32)
    inp = np.arange(8).reshape(4, 2)

    with self.assertRaisesRegex(
        TypeError,
        r"Argument 'x\['b'\]\['c'\]' of shape int32\[4,2\] of "
        "type.*ShapeDtypeStruct.*is not a valid JAX type."):
      jax.jit(lambda x: x)({'a': inp, 'b': {'c': sds}})

  def test_pjit_device_backend_both_error(self):
    with self.assertRaisesRegex(
        ValueError, "can't specify both a device and a backend for jit"):
      with jtu.ignore_warning(category=DeprecationWarning,
                              message="backend and device argument"):
        pjit(lambda x: x, device=jax.devices()[0], backend='cpu')

  def test_pjit_mesh_with_device_or_backend_error(self):
    mesh = jtu.create_mesh((1,), ('x',))
    with mesh:
      with self.assertRaisesRegex(
          ValueError,
          "Mesh context manager should not be used with jit when backend or "
          "device is also specified as an argument to jit."):
        with jtu.ignore_warning(category=DeprecationWarning,
                                message="backend and device argument"):
          pjit(lambda x: x, device=jax.devices()[0])(jnp.arange(8))

  def test_pjit_inline(self):
    @partial(pjit, inline=False)
    def f(x):
      return x * 2

    jaxpr = jax.make_jaxpr(f)(3)
    self.assertIn('pjit', str(jaxpr))

    @partial(pjit, inline=True)
    def g(x):
      return x * 2

    jaxpr = jax.make_jaxpr(g)(3)
    self.assertNotIn('pjit', str(jaxpr))

  def test_pjit_inline_literal(self):
    # https://github.com/jax-ml/jax/issues/27545
    def bar(x):
      return jnp.array(1)

    def foo(x):
      x = pjit(bar, inline=True)(x)
      self.assertEqual(x.shape, ())

    pjit(foo)(0)  # doesn't crash

  def test_pmap_in_axis_resources_error(self):
    pmap_out = jax.pmap(lambda x: x)(jnp.arange(jax.device_count()))
    self.assertIsInstance(pmap_out.sharding, jax.sharding.PmapSharding)

    with self.assertRaisesRegex(
        ValueError,
        r"One of in_shardings.*got sharding.*which is not allowed."):
      pjit(lambda x: x, in_shardings=pmap_out.sharding)

    with self.assertRaisesRegex(
        ValueError,
        r"One of out_shardings.*got sharding.*which is not allowed."):
      pjit(lambda x: x, out_shardings=pmap_out.sharding)

  def test_pmap_sharding_input_to_pjit_single_device(self):
    pmap_out = jax.pmap(lambda x: x)(jnp.arange(jax.device_count()))
    self.assertIsInstance(pmap_out.sharding, jax.sharding.PmapSharding)
    self.assertLen(pmap_out.devices(), jax.device_count())

    out = pjit(lambda x: x * 3)(pmap_out)
    self.assertArraysEqual(out, pmap_out * 3)
    # Even though pmap out is on jax.device_count() number of devices, the
    # output will be 1 device since it will be resharded.
    self.assertLen(out.devices(), 1)

  def test_pmap_sharding_input_to_pjit_multi_device(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    pmap_out = jax.pmap(lambda x: x)(jnp.arange(jax.device_count()))
    self.assertIsInstance(pmap_out.sharding, jax.sharding.PmapSharding)

    inp2 = jnp.arange(4)
    with mesh:
      out1, out2 = pjit(lambda x, y: (x * 2, y * 2))(pmap_out, inp2)

    self.assertArraysEqual(out1, pmap_out * 2)
    self.assertArraysEqual(out2, inp2 * 2)
    self.assertLen(out1.devices(), 4)
    self.assertLen(out2.devices(), 4)
    self.assertTrue(op_shardings.is_op_sharding_replicated(
        out1.sharding._to_xla_hlo_sharding(pmap_out.ndim)))
    self.assertTrue(op_shardings.is_op_sharding_replicated(
        out2.sharding._to_xla_hlo_sharding(inp2.ndim)))

  def test_pmap_sharding_input_pjit_in_axis_resources(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))

    pmap_out = jax.pmap(lambda x: x)(jnp.arange(jax.device_count()))
    self.assertIsInstance(pmap_out.sharding, jax.sharding.PmapSharding)

    out = pjit(lambda x: x * 2, in_shardings=NamedSharding(mesh, P('x')))(pmap_out)
    self.assertArraysEqual(out, pmap_out * 2)
    self.assertLen(out.devices(), 4)

  def test_nested_pjit_closing_over_tracer(self):
    @pjit
    def f(x):
      y = jnp.float32(2) * x

      @pjit
      def g(z):
        return jax.pmap(lambda x: x[jnp.newaxis] * y)(z)

      return g(x)

    f(np.arange(1., dtype='float32').reshape((1, 1)))  # doesn't crash
    # Second call is to trigger C++ dispatch.
    f(np.arange(1., dtype='float32').reshape((1, 1)))  # doesn't crash

  def test_aot_nested_pjit_closing_over_const_top_level(self):
    const = jnp.arange(8.)

    @pjit
    def f(x):
      return const * 2 + x

    inp = jnp.arange(8.)
    compiled = f.lower(inp).compile()
    self.assertArraysEqual(compiled(inp), const * 2 + inp)

  def test_nested_pjit_closing_over_const_top_level_and_tracer(self):
    const = jnp.arange(8.)

    @pjit
    def f(x):
      y = jnp.arange(8., 16.) * x + const

      @pjit
      def g(z):
        return z + y * 2 + const

      return g(x)

    f(jnp.arange(8.))  # doesn't crash
    # Second call is to trigger C++ dispatch.
    f(jnp.arange(8.))  # doesn't crash

  def test_nested_pjit_closing_over_top_level_const(self):
    const = jnp.arange(8.)

    @pjit
    def f(x):

      @pjit
      def g(z):
        return z + const

      return g(x)

    inp = jnp.arange(8., 16.)
    f(inp)  # doesn't crash
    # Second call is to trigger C++ dispatch.
    f(inp)  # doesn't crash

  def test_pjit_sin_nested(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))

    @pjit
    def f(x):
      return jnp.sin(x)

    with mesh:
      inp = jnp.arange(8.)
      out = f(inp)
      self.assertArraysAllClose(out, np.sin(inp))
      self.assertLen(out.devices(), 8)

  def test_jit_with_mesh_context_manager(self):
    mesh = jtu.create_mesh((1,), ('x',))
    with self.assertRaisesRegex(
        RuntimeError,
        "jax.jit only supports `Sharding`s being passed to "
        "in_shardings"):
      with mesh:
        jax.jit(lambda x: x, in_shardings=P('x'),
                out_shardings=P('x'))(jnp.arange(8))

  def test_pjit_nested_uncommitted_output(self):
    @pjit
    def f(x):
      @pjit
      def g(y):
        return y * 2
      return g(x)

    out = f(jnp.arange(8))
    self.assertFalse(out._committed)
    self.assertArraysEqual(out, np.arange(8) * 2)

  def test_pjit_disable_jit(self):
    sideeffect = []

    def f(x):
      sideeffect.append(None)
      return x + 1

    f = jax.jit(f)
    for _ in range(2):
      f(1)
      self.assertLen(sideeffect, 1)

    with jax.disable_jit():
      f(1)
    self.assertLen(sideeffect, 2)

  def test_pmap_pjit_axis_index(self):
    @partial(jax.pmap, axis_name='data')
    def _pmapped_fun(inputs):
      del inputs
      return jax.lax.axis_index('data')

    inputs = jnp.zeros(shape=[jax.device_count()])
    with jtu.ignore_warning(
        message=".*Using jit-of-pmap can lead to inefficient data movement"):
      pjit(_pmapped_fun)(inputs)  # doesn't crash
      jax.jit(_pmapped_fun)(inputs)  # doesn't crash

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_cache_miss_explanations_sharding_mismatch(self):
    mesh = jtu.create_mesh((2,), ('x',))
    s = NamedSharding(mesh, P('x'))

    @jax.jit
    def f(x, y):
      return x * y

    np_inp = np.arange(8, dtype=np.float32)
    x = np_inp
    y = jax.device_put(np_inp, s)
    f(x, y)

    expected_log_len = 1 if not is_persistent_cache_enabled() else 3

    # sharding change
    with config.explain_cache_misses(True):
      x_ = jax.device_put(np_inp, s)
      with self.assertLogs(level='WARNING') as cm:
        f(x_, y)
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn("different input types", msg)
    self.assertIn("at x, now f32[8]({Auto: ('x',)}) and before f32[8]({})", msg)

  def test_pjit_function_cache_cpp(self):
    def f(x):
      return x * 2

    inp = jnp.arange(3.)

    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(10):
        pjit(f)(inp)
    self.assertEqual(count(), 1)

  @jtu.thread_unsafe_test()  # count_pjit_cpp_cache_miss is not thread-safe
  def test_pjit_no_global_cache_hit_axis_resources(self):
    mesh = jtu.create_mesh((1,), ('x',))
    s = NamedSharding(mesh, P('x'))
    inp = jnp.arange(8.0)

    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(10):
        pjit(lambda x: x * 2, in_shardings=s, out_shardings=s)(inp)
    self.assertEqual(count(), 10)

    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(10):
        with jtu.ignore_warning(category=DeprecationWarning,
                                message="backend and device argument"):
          pjit(lambda x: x * 2, device=jax.devices()[0])(inp)
    self.assertEqual(count(), 10)

    pf = pjit(lambda x: x * 2, in_shardings=s, out_shardings=s)
    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(10):
        pf(inp)
    self.assertEqual(count(), 1)

    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      pf1 = pjit(lambda x: x * 2, device=jax.devices()[0])
    with jtu.count_pjit_cpp_cache_miss() as count:
      for _ in range(10):
        pf1(inp)
    self.assertEqual(count(), 1)

  def test_with_sharding_constraint_spmd_axis_name(self):
    mesh = jtu.create_mesh((2, 2, 2), ('replica', 'data', 'mdl'))
    shape = (8, 4, 2, 2)
    x = jnp.arange(math.prod(shape)).reshape(shape)

    def f(inp):
      sharding = NamedSharding(mesh, P('data', None, None))
      return with_sharding_constraint(inp, sharding)

    out = jax.vmap(jax.jit(f), spmd_axis_name='mdl')(x)
    ns, _ = op_shardings.get_num_ways_dim_sharded(
        out.sharding._to_xla_hlo_sharding(out.ndim))
    self.assertListEqual(ns, [2, 2, 1, 1])

    def apply_with_scan(x):
      x, _ = jax.lax.scan(lambda x, _: (f(x), None), x, None, length=1)
      return x

    out2 = jax.vmap(apply_with_scan, spmd_axis_name='mdl')(x)
    ns2, _ = op_shardings.get_num_ways_dim_sharded(
        out2.sharding._to_xla_hlo_sharding(out2.ndim))
    self.assertListEqual(ns2, [2, 2, 1, 1])

  def test_device_put_sharding_nondivisible_sharding_error(self):
    mesh = jtu.create_mesh((2,), ('x',))
    s = NamedSharding(mesh, P('x'))

    x = jnp.ones((1,))
    with self.assertRaisesRegex(
        ValueError, 'implies that the global size of its dimension 0 should be '
                    'divisible by 2, but it is equal to 1 '):
      jax.device_put(x, s)

    y = jnp.ones((2,))
    with self.assertRaisesRegex(
        ValueError, 'implies that the global size of its dimension 0 should be '
                    'divisible by 2, but it is equal to 1 '):
      jax.device_put((y, x), s)

    with self.assertRaisesRegex(
        ValueError,
        "The sharded dimension must be equal to the number of "
        "devices passed to PmapSharding. Got sharded dimension 0 with value 1 "
        r"in shape \(1,\) and the number of devices=2"):
      s2 = jax.pmap(lambda x: x,
                    devices=list(mesh.devices.flat))(jnp.arange(2)).sharding
      jax.device_put(x, s2)

    jax.device_put(2., NamedSharding(mesh, P()))  # doesn't crash

  def test_with_sharding_constraint_with_two_meshes(self):
    if jax.device_count() < 4:
      self.skipTest("Requires more than 4 devices.")

    dev0 = jax.devices()[:2]
    mesh0 = jax.sharding.Mesh(dev0, ('x'))

    dev1 = jax.devices()[2:4]
    mesh1 = jax.sharding.Mesh(dev1, ('x'))

    def f(x):
      y = x * 2
      y = jax.lax.with_sharding_constraint(y, P('x'))
      return y + 2

    with mesh0:
      x = np.ones((32, 4))
      out0 = pjit(f)(x)
      self.assertListEqual(sorted([d.id for d in out0.devices()]),
                           [d.id for d in dev0])

    with mesh1:
      x = np.ones((32, 4))
      out1 = pjit(f)(x)
      self.assertListEqual(sorted([d.id for d in out1.devices()]),
                           [d.id for d in dev1])

  def test_device_assignment_mismatch_apply_primitive(self):
    if jax.device_count() < 2:
      self.skipTest("Requires >=2 devices.")
    arr = jax.device_put(np.arange(8), jax.devices()[0])
    arr2 = jax.device_put(np.arange(8), jax.devices()[1])
    with self.assertRaisesRegex(
        ValueError,
        "Received incompatible devices for jitted computation. Got argument.*"
        r"of concatenate with shape int.*\[8\].*and argument.*"):
      jnp.concatenate([arr, arr2])

  def test_device_put_grad(self):
    if jax.device_count() < 8:
      self.skipTest("Requires >=8 devices.")
    if jtu.is_device_tpu(5, 'e'):
      self.skipTest('TPU v5e does not support computations that run on a '
                    'non-singleton subset of cores.')
    if jtu.is_device_tpu(6, 'e'):
      self.skipTest('TPU v6e does not support computations that run on a '
                    'non-singleton subset of cores.')

    def _test(fun, inp, np_inp, in_s):
      out = fun(inp)
      self.assertArraysEqual(out, np.sum(np_inp ** 2 * 3))
      self.assertArraysEqual(
          [d.id for d in out.sharding._device_assignment], [4, 5, 6, 7])

      gout = jax.grad(fun)(inp)
      self.assertTrue(gout.sharding.is_equivalent_to(in_s, gout.ndim))
      self.assertArraysEqual(
          [d.id for d in gout.sharding._device_assignment], [0, 1, 2, 3])
      self.assertArraysEqual(gout, jax.grad(fun)(np_inp))

    mesh1 = jax.sharding.Mesh(jax.devices()[:4], 'x')
    mesh2 = jax.sharding.Mesh(jax.devices()[4:8], 'x')

    @pjit
    def stage1(x):
      return x ** 2

    @pjit
    def stage2(x):
      return x * 3

    def f(x):
      y = stage1(x)
      y = jax.device_put(y, device=NamedSharding(mesh2, P('x')))
      z = stage2(y)
      return jnp.sum(z)

    def g(x):
      y = stage1(x)
      y = jax.device_put(y, src=NamedSharding(mesh1, P('x')),
                         device=NamedSharding(mesh2, P('x')))
      z = stage2(y)
      return jnp.sum(z)

    np_inp = np.arange(4.)
    in_s = NamedSharding(mesh1, P('x'))
    arr = jax.device_put(np_inp, in_s)

    _test(f, arr, np_inp, in_s)

    _test(g, arr, np_inp, in_s)
    # Test second order autodiff with src argument specified in device_put.
    jtu.check_grads(g, (arr,), order=2)

  @jtu.thread_unsafe_test()  # cache_info isn't thread-safe
  def test_pjit_out_sharding_preserved(self):
    if config.use_shardy_partitioner.value:
      raise unittest.SkipTest("Shardy doesn't support PositionalSharding")
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    ns = NamedSharding(mesh, P('x'))
    ps = PositionalSharding(jax.devices()[:2]).reshape(2, 1)

    arr = jax.device_put(np.arange(8).reshape(8, 1), ns)
    arr2 = jax.device_put(np.arange(8).reshape(8, 1), ps)

    def mul(x):
      return x * 2

    f = pjit(mul, out_shardings=ns)
    f2 = pjit(mul, out_shardings=ps)

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = f(arr)
      cache_info1 = pxla._cached_compilation.cache_info()
      self.assertIsInstance(out.sharding, NamedSharding)

      out = f(arr)
      self.assertIsInstance(out.sharding, NamedSharding)
    self.assertEqual(count(), 1)

    with jtu.count_pjit_cpp_cache_miss() as count:
      out2 = f2(arr)
      cache_info2 = pxla._cached_compilation.cache_info()
      self.assertIsInstance(out2.sharding, PositionalSharding)

      out2 = f2(arr)
      self.assertIsInstance(out2.sharding, PositionalSharding)
    self.assertEqual(count(), 1)

    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)

    with jtu.count_jit_tracing_cache_miss() as tracing_count:
      out3 = jnp.squeeze(arr, axis=-1)
      self.assertIsInstance(out3.sharding, NamedSharding)

      out4 = jnp.squeeze(arr2, axis=-1)
      self.assertIsInstance(out4.sharding, PositionalSharding)
    self.assertEqual(tracing_count(), 2)

  @jtu.thread_unsafe_test()  # cache_info isn't thread-safe
  def test_cache_hit_pjit_lower_with_cpp_cache_miss(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    ns = NamedSharding(mesh, P('x'))
    np_arr = np.arange(8, dtype=np.float32).reshape(8, 1)
    arr = jax.device_put(np_arr, ns)

    def mul(x):
      return x * 2

    f = pjit(mul, in_shardings=ns, out_shardings=ns)

    with (jtu.count_pjit_cpp_cache_miss() as count,
          jtu.count_jit_and_pmap_lowerings() as lowering_count):
      out = f(arr)
      self.assertIsInstance(out.sharding, NamedSharding)

      out2 = f(np_arr)
      self.assertIsInstance(out2.sharding, NamedSharding)

    # Drops out of C++ cache i.e. cache miss
    self.assertEqual(count(), 2)
    self.assertEqual(lowering_count(), 2)

  def test_list_in_pspec(self):
    mesh = jtu.create_mesh((2,), ('x',))
    with mesh:
      out = with_sharding_constraint(jnp.arange(8), P(['x']))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

  def test_wsc_error_on_none(self):
    with self.assertRaisesRegex(
        ValueError,
        'One of with_sharding_constraint arguments got sharding None which is'
        ' not allowed'):
      with_sharding_constraint(jnp.arange(8), None)

  def test_sharding_on_output_with_vmap(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    ns = NamedSharding(mesh, P('x'))
    arr = jax.device_put(
        np.arange(16).reshape(8, 2), NamedSharding(mesh, P(None, 'x')))

    with jtu.count_jit_and_pmap_lowerings() as count:
      vf = jax.vmap(pjit(lambda x: x * 2, in_shardings=ns))
      out = vf(arr)
      self.assertIsInstance(out.sharding, NamedSharding)

      out2 = vf(out)
      self.assertIsInstance(out2.sharding, NamedSharding)

      out3 = vf(out2)
      self.assertIsInstance(out3.sharding, NamedSharding)
    self.assertEqual(count(), 1)

  @config.numpy_dtype_promotion('standard')
  def test_mutable_array_closed_over_multi_device(self):
    mesh = jtu.create_mesh((2,), ('x',))
    key_data = jax.random.key_data(jax.random.key(42))
    key_data_ref = core.mutable_array(key_data)
    output_sharding = NamedSharding(mesh, P('x'))

    @partial(jax.jit, out_shardings=output_sharding)
    def generate_random_numbers():
      key_val = key_data_ref[...]
      outputs = jnp.arange(8, dtype=jnp.float32) + key_val[0]
      return outputs

    generate_random_numbers()  # doesn't crash

  @jtu.thread_unsafe_test()  # cache_info isn't thread-safe
  def test_jit_mul_sum_sharding_preserved(self):
    if config.use_shardy_partitioner.value:
      raise unittest.SkipTest("Shardy doesn't support PositionalSharding")
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    ns = NamedSharding(mesh, P('x'))
    ps = PositionalSharding(jax.devices()[:2]).reshape(2, 1)

    arr = jax.device_put(np.arange(8).reshape(8, 1), ns)
    arr2 = jax.device_put(np.arange(8).reshape(8, 1), ps)

    f = jax.jit(lambda x: x * 2)

    with jtu.count_jit_compilation_cache_miss() as compilation_count:
      out = f(arr)
      self.assertIsInstance(out.sharding, NamedSharding)

      with jtu.count_pjit_cpp_cache_miss() as cpp_count:
        out2 = f(arr2)
        self.assertIsInstance(out2.sharding, PositionalSharding)

        # This will hit the cpp cache.
        out3 = f(out2)
        self.assertIsInstance(out3.sharding, PositionalSharding)

    self.assertEqual(compilation_count(), 2)
    self.assertEqual(cpp_count(), 1)

    out4 = jnp.sum(arr)
    self.assertIsInstance(out4.sharding, NamedSharding)

  def test_single_device_sharding_preserved(self):
    if jax.device_count() < 2:
      self.skipTest('Test requires >=2 devices')

    x = jnp.arange(8)

    # trivial computation
    out = jax.jit(lambda x: x)(x)
    self.assertIsInstance(out.sharding, SingleDeviceSharding)

    # trivial computation with committed inp
    y = jax.device_put(x, jax.devices()[1])
    out2 = jax.jit(lambda x: x)(y)
    self.assertIsInstance(out2.sharding, SingleDeviceSharding)
    self.assertEqual(out2.devices(), {jax.devices()[1]})

    out3 = jax.jit(lambda x: x * 2)(x)
    self.assertIsInstance(out3.sharding, SingleDeviceSharding)

    out4 = jax.jit(lambda x: x * 3,
                   out_shardings=SingleDeviceSharding(jax.devices()[1]))(x)
    self.assertIsInstance(out4.sharding, SingleDeviceSharding)
    self.assertEqual(out4.devices(), {jax.devices()[1]})

  def test_none_out_sharding(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    x = jnp.arange(8)
    with mesh:
      out = pjit(lambda x: x * 2, out_shardings=None)(x)
      self.assertEqual(out.sharding.mesh, mesh)
      self.assertIsInstance(out.sharding, NamedSharding)
      self.assertEqual(out.sharding.spec, P())

    x2 = jax.device_put(x, NamedSharding(mesh, P()))
    out2 = pjit(lambda x: x * 2)(x2)
    self.assertIsInstance(out2.sharding, NamedSharding)
    self.assertEqual(out2.sharding.mesh, mesh)
    self.assertEqual(out2.sharding.spec, P())

  def test_sharding_preserved_apply_primitive(self):
    if config.use_shardy_partitioner.value:
      raise unittest.SkipTest("Shardy doesn't support PositionalSharding")
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    ns = NamedSharding(mesh, P('x'))

    arr = jax.device_put(np.arange(8).reshape(8, 1), ns)

    out = jnp.copy(arr)
    self.assertIsInstance(out.sharding, NamedSharding)

    ps = PositionalSharding(jax.devices()[:2]).reshape(2, 1)
    arr2 = jax.device_put(np.arange(8).reshape(8, 1), ps)
    out2 = jnp.copy(arr2)
    self.assertIsInstance(out2.sharding, PositionalSharding)

    arr3 = jnp.arange(8)
    out3 = jnp.copy(arr3)
    self.assertIsInstance(out3.sharding, SingleDeviceSharding)

    arr4 = jax.device_put(jnp.arange(8), jax.devices()[1])
    out4 = jnp.copy(arr4)
    self.assertIsInstance(out4.sharding, SingleDeviceSharding)
    self.assertEqual(out4.devices(), {jax.devices()[1]})

  def test_same_named_sharding_pspec_on_eager_ops(self):
    mesh = jtu.create_mesh((1, 8, 1), ('x', 'y', 'z'))
    sharding = jax.sharding.NamedSharding(mesh, P('x', 'y', 'z'))
    x = jax.device_put(jnp.arange(32).reshape(1, -1, 1), sharding)
    y = x + 1
    self.assertEqual(x.sharding, y.sharding)

  def test_different_named_sharding_object_replicated(self):
    mesh = jtu.create_mesh((1, 2), ('x', 'y'))
    sharding = jax.sharding.NamedSharding(mesh, P('x'))
    x = jax.device_put(np.arange(16).reshape(8, 2), sharding)
    y = jnp.sum(x)
    self.assertNotEqual(x.sharding, y.sharding)

  def test_vmap_pjit_single_device(self):
    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      jf = pjit(lambda x: x, device=jax.devices()[0])
    out = jax.vmap(jf)(jnp.ones((3,)))  # doesn't crash
    self.assertIsInstance(out.sharding, SingleDeviceSharding)

  def test_to_gspmd_sharding_cache_with_and_without_device(self):
    mesh = jtu.create_mesh((2,), ('x',))
    np_inp = jnp.arange(4)

    def identity(x):
      return x

    # Fill up the to_gspmd_sharding cache so that the next jit will miss it.
    out = jax.jit(identity,
                  in_shardings=SingleDeviceSharding(jax.devices()[0]))(np_inp)
    self.assertEqual(out.devices(), {jax.devices()[0]})
    self.assertArraysEqual(out, np_inp)

    with jtu.ignore_warning(category=DeprecationWarning,
                            message="backend and device argument"):
      out2 = jax.jit(identity, device=jax.devices()[0])(
          jax.device_put(np_inp, NamedSharding(mesh, P('x'))))
    self.assertEqual(out2.devices(), {jax.devices()[0]})
    self.assertArraysEqual(out2, np_inp)

  def test_jit_submhlo_cached(self):
    @jax.jit
    def nest(x):
      return x * 2

    @jax.jit
    def top(x):
      y = nest(x)
      z = nest(y)
      a = nest(z)
      b = nest(a)
      return b

    with jtu.count_subjaxpr_to_hlo_conversion(fun_name='nest') as count:
      top(jnp.arange(8))

    # The count should be 1 because `nest`'s lowering to MHLO should be cached.
    self.assertEqual(count(), 1)

  def test_wsc_eager(self):
    mesh = jtu.create_mesh((2,), ('x',))
    np_inp = np.arange(8)
    inp = jax.device_put(np_inp, NamedSharding(mesh, P()))
    out = with_sharding_constraint(inp, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    for s in out.addressable_shards:
      self.assertArraysEqual(s.data, np_inp[s.index])

  def test_wsc_eager_no_resharding(self):
    mesh = jtu.create_mesh((2,), ('x',))
    np_inp = np.arange(8)
    inp = jax.device_put(np_inp, NamedSharding(mesh, P('x')))
    out = with_sharding_constraint(inp, NamedSharding(mesh, P('x')))
    self.assertEqual(id(out), id(inp))

  def test_wsc_eager_different_order_devices(self):
    mesh1 = jtu.create_mesh((2,), ('x',))
    mesh2 = jax.sharding.Mesh([jax.devices()[1], jax.devices()[0]], 'x')
    inp = jax.device_put(np.arange(8), NamedSharding(mesh1, P()))
    with self.assertRaisesRegex(
        ValueError, "Received incompatible devices for jitted computation"):
      with_sharding_constraint(inp, NamedSharding(mesh2, P('x')))

  def test_jaxpr_as_fun_fast_path(self):
    @jax.jit
    def f(x):
      return x * 2
    inp = jax.device_put(jnp.arange(8), jax.devices()[0])
    jaxpr = jax.make_jaxpr(f)(inp)

    with jtu.count_pjit_cpp_cache_miss() as count:
      out1 = core.jaxpr_as_fun(jaxpr)(inp)
      out2 = core.jaxpr_as_fun(jaxpr)(inp)
    self.assertEqual(count(), 1)
    self.assertArraysEqual(out1[0], inp * 2)
    self.assertArraysEqual(out2[0], inp * 2)

  @jtu.run_on_devices('tpu', 'gpu')
  def test_aot_device_implicit_transfer(self):
    mesh = jtu.create_mesh((1,), 'x')
    np_inp = np.arange(8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P()))

    @jax.jit
    def f(x):
      return x * 2

    compiled = f.lower(arr).compile()

    cpu_dev = jax.devices('cpu')[0]
    with jax.default_device(cpu_dev):
      cpu_arr = jnp.arange(8)
      self.assertEqual(cpu_arr.sharding, SingleDeviceSharding(cpu_dev))
      self.assertFalse(cpu_arr._committed)

    out = compiled(cpu_arr)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P()))
    self.assertEqual(out.sharding.memory_kind, 'device')

  def test_jit_static_argnames_non_interned(self):
    def do_nothing(foobar: int):
      return foobar

    argname = "foobar"
    # Has the side effect of ensuring argname is not interned.
    argname = str(json.loads(json.dumps(argname)))
    jax.jit(do_nothing, static_argnames=[argname])(foobar=2)  # doesn't crash

  def test_most_recent_executable_outer_inner_cache(self):
    x = np.zeros((20, 20), dtype=jnp.float64)

    def trace_to_jaxpr(x):
      jnp.pad(x, [(0, 1), (0, 0)], mode= 'wrap')
      jnp.pad(x, [(0, 0), (1, 0)], mode= 'constant',
              constant_values= ((0.0, 0.0), (0.0, 0.0)))

    jaxpr = jax.make_jaxpr(trace_to_jaxpr)(x)
    jax._src.core.jaxpr_as_fun(jaxpr)(x)

    jnp.pad(x, [(0, 1), (0, 0)], mode= 'wrap')
    jnp.pad(x, [(0, 1), (0, 0)], mode= 'wrap')  # doesn't crash

  def test_shape_dtype_struct_as_const_error(self):
    const = jax.ShapeDtypeStruct((8,), jnp.int32)
    with self.assertRaisesRegex(TypeError,
                                r"Argument.*is not a valid JAX type"):
      jax.jit(lambda x: (x, const))(jnp.arange(8))

  def test_jit_out_shardings_none(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    inp = jax.device_put(np_inp, s)
    out = jax.jit(lambda x: x * 2, out_shardings=None)(inp)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, s)

  def test_jit_in_shardings_none(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    inp = jax.device_put(np_inp, s)

    out = jax.jit(lambda x: x * 2, in_shardings=None)(inp)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, s)

    out2 = jax.jit(lambda x: x * 2, in_shardings=None)(np_inp)
    self.assertArraysEqual(out2, np_inp * 2)
    self.assertEqual(out2.sharding, SingleDeviceSharding(jax.devices()[0]))

  def test_device_put_in_jit_default_mem_kind_no_op(self):
    mesh = jtu.create_mesh((2,), 'x')
    np_inp = np.arange(8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x')))

    @jax.jit
    def f(x):
      y = x * 2
      return jax.device_put(y, NamedSharding(mesh, P()))

    lowered_text = f.lower(arr).as_text()
    self.assertNotIn('@Sharding', lowered_text)
    self.assertNotIn('@annotate_device_placement', lowered_text)

  def test_jit_both_shardings_none(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    inp = jax.device_put(np_inp, s)

    out = jax.jit(lambda x: x * 2, in_shardings=None, out_shardings=None)(inp)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, s)

    out2 = jax.jit(lambda x: x * 2, in_shardings=None, out_shardings=None)(np_inp)
    self.assertArraysEqual(out2, np_inp * 2)
    self.assertEqual(out2.sharding, SingleDeviceSharding(jax.devices()[0]))

  def test_jit_lower_shape_dtype_struct_sharding_none(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))

    lower_inp1 = jax.ShapeDtypeStruct((8, 2), np.int32, sharding=s)
    # Will be considered as uncommitted and resharded over all the devices of
    # the mesh.
    lower_inp2 = jax.ShapeDtypeStruct((8, 2), np.int32)

    compiled = jax.jit(lambda x, y: (x * 2, y * 2)).lower(
        lower_inp1, lower_inp2).compile()

    np_inp = np.arange(16, dtype=np.int32).reshape(8, 2)
    inp = jax.device_put(np_inp, s)
    out1, out2 = compiled(inp, np_inp)

    self.assertArraysEqual(out1, np_inp * 2)
    self.assertArraysEqual(out2, np_inp * 2)
    self.assertTupleEqual(out1.sharding._device_assignment,
                          s.mesh._flat_devices_tuple)
    self.assertTupleEqual(out2.sharding._device_assignment,
                          s.mesh._flat_devices_tuple)

  def test_vmap_spmd_axis_name_error(self):
    s = SingleDeviceSharding(jax.devices()[0])

    def f(inp):
      return with_sharding_constraint(inp, s)

    arr = jax.device_put(np.arange(8), s)
    with self.assertRaisesRegex(
        ValueError,
        'If you are using spmd_axis_name parameter of jax.vmap, please'
        ' make sure to run your jitted function inside the mesh context'
        ' manager.*SingleDeviceSharding'):
      jax.jit(jax.vmap(f, spmd_axis_name='x'))(arr)

  def test_no_output_multiple_devices(self):
    mesh = jtu.create_mesh((2,), ('x',))

    @pjit
    def f():
      return

    with mesh:
      f()  # doesn't crash

  def test_lowering_cache_hit_different_devices(self):
    if jax.device_count() < 4:
      self.skipTest('Requires >=4 devices')

    mesh1 = jax.sharding.Mesh(jax.devices()[:2], 'x')
    mesh2 = jax.sharding.Mesh(jax.devices()[2:4], 'x')

    @jax.jit
    def f(x):
      return x * 2

    def g(a):
      a = jax.device_put(a, NamedSharding(mesh1, P('x')))
      out_a = f(a)  # lowering cached

      # same num_devices but different devices.
      b = jax.device_put(out_a, NamedSharding(mesh2, P('x')))
      f(b)  # lowering cache *hit*

    with jtu.count_jit_and_pmap_lowerings() as count:
      g(np.arange(8))
    self.assertEqual(count(), 1)

  def test_lowering_cache_miss_different_devices_and_sharding(self):
    if jax.device_count() < 4:
      self.skipTest('Requires >=4 devices')

    mesh1 = jax.sharding.Mesh(jax.devices()[:2], 'x')
    mesh2 = jax.sharding.Mesh(jax.devices()[2:4], 'y')

    @jax.jit
    def f(x):
      return x * 2

    def g(a):
      a = jax.device_put(a, NamedSharding(mesh1, P('x')))
      out_a = f(a)  # lowering cached

      # same num_devices but different devices and sharding
      b = jax.device_put(out_a, NamedSharding(mesh2, P()))
      f(b)  # lowering cache *miss*

    with jtu.count_jit_and_pmap_lowerings() as count:
      g(np.arange(8))
    self.assertEqual(count(), 2)

  def test_single_device_named_sharding_preserved(self):
    mesh = jax.sharding.Mesh([jax.devices()[0]], 'x')
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(8)
    inp = jax.device_put(np_inp, s)

    out = jax.jit(lambda x: x)(inp)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, np_inp)

  def test_mpmd_device_put_fast_path(self):
    if jax.device_count() < 4:
      self.skipTest('Needs >= 4 devices')

    dev_count = jax.device_count()
    mesh1 = jax.sharding.Mesh(jax.devices()[:dev_count//2], 'x')
    mesh2 = jax.sharding.Mesh(jax.devices()[dev_count//2:], 'x')
    inp = np.arange(8)
    arr1 = jax.device_put(inp, NamedSharding(mesh1, P('x')))

    # This is to prevent changes to shard_arg_handler of Array which checks for
    # indices to take the fast path for resharding. Changes made to the handler
    # to check for shardings instead of indices will cause this test to fail and
    # that is expected.
    with jtu.count_device_put_fast_path_hit() as count:
      out = jax.device_put(arr1, NamedSharding(mesh2, P('x')))
    self.assertEqual(count(), 1)
    self.assertTupleEqual(out.sharding._device_assignment,
                          mesh2._flat_devices_tuple)
    self.assertArraysEqual(out, inp)

  def test_prng_sharding_propagation(self):
    input_shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    @jax.jit
    def make_keys(seeds):
      make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
      key = make_key(seeds)
      return key.T

    out = make_keys(seeds)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))

    base_array = jax.random.key_data(out)
    self.assertEqual(base_array.shape, (2, 8, 2))
    self.assertEqual(base_array.sharding, NamedSharding(mesh, P('y', 'x', None)))

    lowered_text = make_keys.lower(seeds).as_text()
    if config.use_shardy_partitioner.value:
      self.assertIn('<@empty_mesh, [{?}, {?}, {}]>', lowered_text)
    else:
      self.assertIn('unspecified_dims=[0,1]', lowered_text)

  def test_prng_sharding_propagation_with_nested_jit(self):
    input_shape = (8, 2)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    @jax.jit
    def make_keys(seeds):
      @partial(jax.jit, out_shardings=NamedSharding(mesh, P('y')))
      def f():
        make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
        return make_key(seeds)
      x = f()
      return x.T

    out = make_keys(seeds)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y')))

    base_array = jax.random.key_data(out)
    self.assertEqual(base_array.shape, (2, 8, 2))
    self.assertEqual(base_array.sharding, NamedSharding(mesh, P(None, 'y', None)))

    lowered_text = make_keys.lower(seeds).as_text()
    if config.use_shardy_partitioner.value:
      self.assertIn('<@empty_mesh, [{?}, {?}, {}]>', lowered_text)
    else:
      self.assertIn('unspecified_dims=[0,1]', lowered_text)

  def test_partial_sharded_prng_key_inp(self):
    input_shape = (8, 2, 2)
    mesh = jtu.create_mesh((2, 2, 2), ('x', 'y', 'z'))
    spec = P('x', 'y', None)

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    @jax.jit
    def make_keys(seeds):
      make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
      key = make_key(seeds)
      return key.T

    make_keys(seeds)
    out = make_keys(seeds)  # cpp dispatch
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    base_array = jax.random.key_data(out)
    self.assertEqual(base_array.shape, (2, 2, 8, 2))
    self.assertEqual(base_array.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    lowered_text = make_keys.lower(seeds).as_text()
    if config.use_shardy_partitioner.value:
      self.assertIn('<@empty_mesh, [{?}, {?}, {?}, {}]>', lowered_text)
    else:
      self.assertIn('unspecified_dims=[0,1,2]', lowered_text)

  def test_wsc_with_scalar(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P())
    out = jax.lax.with_sharding_constraint(1., s)
    self.assertArraysEqual(out, 1.)
    self.assertEqual(out.sharding, s)

  def test_jit_partially_specified_shardings(self):

    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    s2 = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np_inp, s)
    arr2 = jax.device_put(np_inp, s2)

    @partial(jax.jit, in_shardings=(s, None, s2, UNSPECIFIED, UNSPECIFIED),
             out_shardings=(s2, None, None, s, None))
    def f(x, y, z, a, b):
      return x * 2, y @ y.T, z ** 2, a * 3, b.T

    out1, out2, out3, out4, out5 = f(arr, np_inp, arr2, np_inp, arr)
    self.assertArraysEqual(out1, np_inp * 2)
    self.assertArraysEqual(out2, np_inp @ np_inp.T)
    self.assertArraysEqual(out3, np_inp ** 2)
    self.assertArraysEqual(out4, np_inp * 3)
    self.assertArraysEqual(out5, np_inp.T)

  def test_input_shardings_aot(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x')))

    @jax.jit
    def f(x, y):
      return x * 2, y.T

    arg_shardings, _ = f.lower(arr, np_inp).compile().input_shardings
    for s in arg_shardings:
      self.assertIsInstance(s, NamedSharding)

  def test_parameter_tupled_jit(self):
    if not jtu.test_device_matches(["tpu"]):
      self.skipTest('Parameters are tupled only on TPU if >2000 parameters')

    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x'))

    @jax.jit
    def f(*args):
      return args * 2

    inp = np.arange(8)
    arr = jax.device_put(inp, s)
    inps = [arr, *[inp] * 2001]
    f(inps)  # doesn't crash

  def test_spmd_preserves_input_sharding_vmap_grad(self):
    if config.use_shardy_partitioner.value:
      self.skipTest("Shardy doesn't support PositionalSharding")
    # https://github.com/jax-ml/jax/issues/20710
    n_devices = jax.device_count()
    sharding = PositionalSharding(jax.devices())

    def model(params, x):
      return x @ params

    feature_dim = 3
    batch_size_total = 8

    # Get example data
    x = jnp.ones((batch_size_total, feature_dim))
    params = jnp.ones(feature_dim)

    # Shard data, replicate params
    x = jax.device_put(x, sharding.reshape(n_devices, 1))
    params = jax.device_put(params, sharding.replicate(axis=0))

    model(params, x)  # doesn't crash

    jax.vmap(model, in_axes=(None, 0))(params, x)  # doesn't crash

    jax.grad(lambda p: model(p, x).sum())(params)  # doesn't crash

    jax.vmap(jax.grad(model), in_axes=(None, 0))(params, x)  # doesn't crash

  def test_jit_token_input(self):
    x = jnp.arange(8)
    token = jax.lax.create_token(None)
    device = jax.devices()[0]
    x = jax.device_put(x, device=device)
    out1, out2 = jax.jit(lambda x, t: (x, t))(x, token)
    self.assertArraysEqual(out1, x)
    self.assertIsInstance(out2, core.Token)

  def test_uneven_sharding_wsc(self):
    mesh = jtu.create_mesh(
        (2, 1, 1, 1, 1), ('data', 'expert', 'fsdp', 'seq', 'model')
    )

    @jax.jit
    def fn(key):
      x = jnp.arange(113003)
      x = with_sharding_constraint(x, P('data'))
      y = jnp.arange(65536)
      y = with_sharding_constraint(y.reshape(-1), P('data'))
      z = jnp.concatenate([x, y], axis=0)
      z = with_sharding_constraint(z, P('data'))
      return x, y, z

    with mesh:
      x, y, z = fn(jax.random.key(42))

    expected_x = np.arange(113003)
    expected_y = np.arange(65536)
    expected_z = np.concatenate([x, y], axis=0)

    self.assertArraysEqual(expected_x.max(), x.max())
    self.assertArraysEqual(expected_y.max(), y.max())
    self.assertArraysEqual(expected_z.max(), z.max())

  def test_threefry_partitionable_context_within_jit(self):
    with jax.threefry_partitionable(False):
      def f(x):
        return x + jax.random.randint(jax.random.key(72), (), 0, 10)

      def g(x):
        with jax.threefry_partitionable(True):  # False by default
          return x + jax.random.randint(jax.random.key(72), (), 0, 10)

      h = jax.jit(g)

      self.assertNotEqual(f(1), g(1))
      self.assertEqual(g(1), h(1))

  def test_wsc_vmap_unconstrained_spmd_axis_name(self):
    def get_wsc_eqn_sharding(jaxpr):
      for eqn in jaxpr.eqns:
        if str(eqn.primitive) == 'sharding_constraint':
          return eqn.params['sharding'], eqn.params['unconstrained_dims']
      for s in core.subjaxprs(jaxpr):
        return get_wsc_eqn_sharding(s)

    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    inp = jnp.ones((10, 10))

    def a_function(x):
      return with_sharding_constraint(x, NamedSharding(mesh, P(P.UNCONSTRAINED)))

    def vmap_the_function_spmd(y):
      return jax.vmap(a_function, spmd_axis_name='x')(y)

    f1 = jax.jit(vmap_the_function_spmd)
    f1(inp)  # doesn't crash
    jaxpr1 = jax.make_jaxpr(f1)(inp)
    s1, u1 = get_wsc_eqn_sharding(jaxpr1)
    self.assertEqual(s1.spec, P('x', P.UNCONSTRAINED))
    self.assertEqual(u1, {1})

    def vmap_the_function_no_spmd(y):
      return jax.vmap(a_function)(y)

    f2 = jax.jit(vmap_the_function_no_spmd)
    f2(inp)  # doesn't crash
    jaxpr2 = jax.make_jaxpr(f2)(inp)
    s2, u2 = get_wsc_eqn_sharding(jaxpr2)
    self.assertEqual(s2.spec, P(P.UNCONSTRAINED, P.UNCONSTRAINED))
    self.assertEqual(u2, {0, 1})

  def test_aot_sharding_dce(self):
    inp = np.arange(8)

    @jax.jit
    def f(x, y):
      return x

    input_shardings, _ = f.lower(inp, inp).compile().input_shardings
    self.assertLen(input_shardings, 2)

  def test_aot_out_info(self):
    inp = np.arange(8, dtype=np.int32)
    out_info = jax.jit(lambda x: x).lower((inp, inp)).out_info
    self.assertEqual(out_info[0].shape, (8,))
    self.assertEqual(out_info[1].shape, (8,))
    self.assertEqual(out_info[0].dtype, np.int32)
    self.assertEqual(out_info[1].dtype, np.int32)
    self.assertEqual(out_info[0].sharding, None)
    self.assertEqual(out_info[1].sharding, None)

  def test_jit_trace(self):
    def f(x):
      return x * 2

    traced = jax.jit(f).trace(jnp.arange(8, dtype=jnp.int32))
    self.assertLen(traced.jaxpr.eqns, 1)
    self.assertEqual(jax.tree.structure(traced.out_info).num_leaves, 1)
    self.assertEqual(traced.out_info.shape, (8,))
    self.assertEqual(traced.out_info.dtype, jnp.int32)
    # one for args, one for kwargs (though kwargs is empty)
    self.assertLen(traced.in_avals, 2)
    self.assertLen(traced.in_avals[0], 1)
    self.assertLen(traced.in_avals[1], 0)  # empty kwarg

  def test_in_out_shardings_unconstrained_error(self):
    mesh = jtu.create_mesh((1,), ('x',))

    with self.assertRaisesRegex(
        ValueError, "Unconstrained dims are not allowed"):
      jax.jit(lambda x: x,
              in_shardings=NamedSharding(mesh, P(P.UNCONSTRAINED, 'x')))

  def test_empty_io_callback_under_shard_map(self):
    mesh = jtu.create_mesh((4,), 'i')

    def empty_callback(x):
      return

    def _f(x, y):
      jax.experimental.io_callback(
          empty_callback, (), x, ordered=True)
      return x + y[..., jnp.newaxis]

    f = jax.jit(shard_map(
        _f, mesh=mesh, in_specs=(P(None, 'i'), P(None)),
        out_specs=P(None, 'i')))
    f(jnp.zeros((2, 16)), jnp.ones(2))

  def test_empty_io_callback_under_shard_map_reshard_to_singledev(self):
    if config.use_shardy_partitioner.value:
      self.skipTest("TODO(b/384938613): Failing under shardy.")
    mesh = jtu.create_mesh((4,), 'i')

    def empty_callback(x):
      return

    def _f(x, y):
      jax.experimental.io_callback(
          empty_callback, (), x, ordered=True)
      return x + y[..., jnp.newaxis]

    f = jax.jit(shard_map(
        _f, mesh=mesh, in_specs=(P(None, 'i'), P(None)),
        out_specs=P(None, 'i')))
    f(jnp.zeros((2, 16)), jnp.ones(2))

    _f(jnp.zeros((2, 16)), jnp.ones(2))

    jax.jit(_f)(jnp.zeros((2, 16)), jnp.ones(2))

    jax.effects_barrier()

  def test_jit_trace_lower_and_compile(self):
    def f(x):
      return x * 2

    lowered = jax.jit(f).trace(jnp.arange(8)).lower()
    self.assertEqual(lowered.args_info[0][0].shape, (8,))

    compiled = lowered.compile()
    out = compiled(jnp.arange(8))
    self.assertArraysEqual(out, np.arange(8) * 2)

    # fast-forward
    lowered2 = jax.jit(f).lower(jnp.arange(8))
    self.assertEqual(lowered2.args_info[0][0].shape, (8,))

    compiled2 = lowered2.compile()
    out2 = compiled2(jnp.arange(8))
    self.assertArraysEqual(out2, np.arange(8) * 2)

  def test_nullary_out_sharding_partial(self):
    mesh = jtu.create_mesh((jax.device_count(),), 'x')

    @partial(jax.jit, out_shardings=(None, NamedSharding(mesh, P())))
    def init():
      tensor = jnp.zeros(shape=(1,))
      other_tensor = jnp.zeros(shape=(1,))
      return tensor, other_tensor

    out = init()
    self.assertIsInstance(out[0].sharding, NamedSharding)
    self.assertIsInstance(out[1].sharding, NamedSharding)

  def test_device_put_efficient_reshard_single_host(self):
    if jax.device_count() < 4:
      self.skipTest('Requires >= 4 devices')

    dev = jax.devices()
    mesh1 = Mesh(np.array([dev[0], dev[1], dev[2], dev[3]]).reshape(2, 2),
                 ('x', 'y'))
    mesh2 = Mesh(np.array([dev[3], dev[2], dev[1], dev[0]]).reshape(2, 2),
                 ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    s1 = NamedSharding(mesh1, P('x', 'y'))
    s2 = NamedSharding(mesh2, P('x'))

    x_s1 = jax.device_put(np_inp, s1)

    with jax.transfer_guard('disallow_explicit'):
      out = jax.device_put(x_s1, s2)
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.sharding, s2)

  @parameterized.named_parameters(
      ("8_2", (8, 2)),
      ("8_384", (8, 384)),
  )
  def test_device_put_efficient_reshard_complex_mesh(self, shape):
    if jax.device_count() < 8:
      self.skipTest('Requires >= 8 devices')

    dev = jax.devices()
    mesh1 = jax.sharding.Mesh(
        np.asarray(dev).reshape([1, 2, 2, 2]),
        ('replica', 'data', 'seq', 'model'))
    mesh2 = jax.sharding.Mesh(
        np.asarray(jax.devices())
        .reshape([1, 1, 2, 2, 2, 1])
        .swapaxes(2, 3)
        .reshape([1, 1, 4, 2, 1]),
        ('replica', 'data', 'seq', 'model_q', 'model_kv'))

    np_inp = jnp.arange(math.prod(shape)).reshape(shape)
    s1 = NamedSharding(mesh1, P('model'))
    s2 = NamedSharding(mesh2, P())

    x_s1 = jax.device_put(np_inp, s1)
    # Reshard!
    out = jax.device_put(x_s1, s2)
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.sharding, s2)

    s3 = NamedSharding(mesh2, P('model_q'))
    x_s3 = jax.device_put(np_inp, s3)
    # Reshard to iota device assignment!
    out2 = jax.device_put(x_s3, s1)
    self.assertArraysEqual(out2, np_inp)
    self.assertEqual(out2.sharding, s1)

  def test_device_put_donate_pytree(self):
    shape1 = (8, 2)
    shape2 = (8, 384)
    if jax.device_count() < 8:
      self.skipTest('Requires >= 8 devices')

    dev = jax.devices()
    mesh1 = jax.sharding.Mesh(
        np.asarray(dev).reshape([1, 2, 2, 2]),
        ('replica', 'data', 'seq', 'model'))
    mesh2 = jax.sharding.Mesh(
        np.asarray(jax.devices())
        .reshape([1, 1, 2, 2, 2, 1])
        .swapaxes(2, 3)
        .reshape([1, 1, 4, 2, 1]),
        ('replica', 'data', 'seq', 'model_q', 'model_kv'))

    np_inp1 = jnp.arange(math.prod(shape1)).reshape(shape1)
    np_inp2 = jnp.arange(math.prod(shape2)).reshape(shape2)
    s1 = NamedSharding(mesh1, P('model'))
    s2 = NamedSharding(mesh2, P('model_q'))

    x1 = jax.device_put(np_inp1, s1)
    x2 = jax.device_put(np_inp2, s1)
    # Reshard!
    out1, out2 = jax.device_put((x1, x2), s2, donate=(True, False))
    self.assertArraysEqual(out1, np_inp1)
    self.assertArraysEqual(out2, np_inp2)
    self.assertEqual(out1.sharding, s2)
    self.assertEqual(out2.sharding, s2)
    self.assertTrue(x1.is_deleted())
    self.assertFalse(x2.is_deleted())

  def test_convert_element_type_sharding(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    inp = np.arange(16).reshape(8, 2)

    out = lax_internal._convert_element_type(
        inp, new_dtype=np.float32, weak_type=False, sharding=s)
    self.assertArraysEqual(out, inp.astype('float32'))
    self.assertEqual(out.dtype, np.float32)
    self.assertEqual(out.sharding, s)

  def test_jnp_array_sharding(self):
    if jax.device_count() < 4:
      self.skipTest('Requires >=4 devices')
    mesh = jax.make_mesh((2, 2), ('x', 'y'), devices=jax.devices()[:4])
    s = NamedSharding(mesh, P('x', 'y'))
    inp = np.arange(16).reshape(8, 2)

    out = jnp.array(inp, device=s)
    self.assertArraysEqual(out, inp)
    self.assertEqual(out.sharding, s)

  def test_jnp_array_inside_jit_sharding(self):
    if jax.device_count() < 4:
      self.skipTest('Requires >=4 devices')
    mesh = jax.make_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    inp = np.arange(16).reshape(8, 2)

    @jax.jit
    def f():
      return jnp.array(inp, dtype=np.float32, device=s)

    out = f()
    self.assertArraysEqual(out, inp.astype('float32'))
    self.assertEqual(out.sharding, s)
    self.assertEqual(out.dtype, np.float32)

    @jax.jit
    def g(x):
      return jnp.array(x, dtype=np.float32, device=s)

    out2 = g(inp)
    self.assertArraysEqual(out2, inp.astype('float32'))
    self.assertEqual(out2.sharding, s)
    self.assertEqual(out2.dtype, np.float32)

  def test_make_mesh_non_int_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "`axis_shapes` passed to `make_mesh` should be a sequence of ints"):
      jax.make_mesh(((4,), 4), ('x', 'y'))

    jax.make_mesh((1, np.int32(1), np.int64(1)), ('x', 'y', 'z'))  # doesn't crash

  def test_jnp_array_reshard_error(self):
    if jax.device_count() < 2:
      self.skipTest('Requires >=2 devices')
    arr = jax.device_put(np.arange(8), jax.devices()[0])
    with self.assertRaisesRegex(ValueError, "Received incompatible devices.*"):
      jnp.array(arr, device=jax.devices()[1])

  def test_jnp_array_sharded_array_no_op(self):
    inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(inp, jax.devices()[0])

    out = lax_internal._convert_element_type(
        arr, sharding=SingleDeviceSharding(jax.devices()[0]))
    self.assertArraysEqual(out, inp)
    self.assertEqual(out.unsafe_buffer_pointer(), arr.unsafe_buffer_pointer())

  def test_wsc_named_sharding_nullary(self):
    mesh = jtu.create_mesh((2,), ('x',))
    s = NamedSharding(mesh, P())

    @jax.jit
    def f():
      return jax.lax.with_sharding_constraint(jnp.arange(8), s)

    out = f()
    self.assertEqual(out.sharding, s)

  @jtu.run_on_devices('tpu', 'gpu')
  def test_aot_device_mismatch(self):
    mesh = jtu.create_mesh((1,), 'x')
    np_inp = np.arange(8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P()))

    @jax.jit
    def f(x):
      return x * 2

    compiled = f.lower(arr).compile()

    cpu_arr = jax.device_put(np_inp, jax.devices('cpu')[0])
    with self.assertRaisesRegex(
        ValueError,
        "Compiled object called with input sharding.*does not match"):
      compiled(cpu_arr)

  def test_different_devices_wsc_abstract_mesh_cache_hit(self):
    if jax.device_count() < 4:
      self.skipTest('Requires >=4 devices')

    mesh1 = jax.sharding.Mesh(jax.devices()[:2], 'x')
    mesh2 = jax.sharding.Mesh(jax.devices()[2:4], 'x')

    @jax.jit
    def f(x):
      x = with_sharding_constraint(
          x, NamedSharding(mesh1.abstract_mesh, P('x')))
      return jax.lax.sin(x)

    with (
        jtu.count_jit_tracing_cache_miss() as tracing_count,
        jtu.count_jit_and_pmap_lowerings() as lowering_count,
        jtu.count_jit_compilation_cache_miss() as compilation_count,
    ):
      a = jax.device_put(np.arange(8.), NamedSharding(mesh1, P()))
      out_a = f(a)  # tracing and lowering cached

      # same num_devices but different devices.
      b = jax.device_put(out_a, NamedSharding(mesh2, P()))
      f(b)  # tracing and lowering cache *hit*

    self.assertEqual(tracing_count(), 1)
    self.assertEqual(lowering_count(), 1)
    self.assertEqual(compilation_count(), 2)  # 2 misses since devices differ.

  def test_wsc_abstract_mesh(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))

    abstract_mesh = mesh.abstract_mesh

    def f(x):
      x = with_sharding_constraint(x, NamedSharding(abstract_mesh, P('x')))
      return x * 2

    out = jax.jit(f)(arr)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

    out_eager = f(arr)
    self.assertArraysEqual(out_eager, np_inp * 2)
    self.assertEqual(out_eager.sharding, NamedSharding(mesh, P('x')))

  def test_wsc_sds_abstract_mesh(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P())
    abstract_mesh = mesh.abstract_mesh

    @jax.jit
    def f(x):
      x = with_sharding_constraint(x, NamedSharding(abstract_mesh, P('x')))
      return x * 2

    sds = jax.ShapeDtypeStruct((8, 2), np.float32, sharding=s)
    f.eval_shape(sds)  # doesn't crash

  def test_wsc_vmap_abstract_mesh(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    def f(x):
      x = with_sharding_constraint(x, NamedSharding(mesh.abstract_mesh, P('x')))
      return x * 2

    out = jax.jit(jax.vmap(f))(arr)  # doesn't crash
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'x')))

    out2 = jax.jit(jax.vmap(f, spmd_axis_name='y'))(arr)
    self.assertEqual(out2.sharding, NamedSharding(mesh, P('y', 'x')))

  def test_wsc_abstract_mesh_errors(self):
    mesh = jtu.create_mesh((2,), ('x',))
    np_inp = np.arange(8)
    abstract_mesh = mesh.abstract_mesh
    s_abs = NamedSharding(abstract_mesh, P('x'))

    with self.assertRaisesRegex(
        ValueError, ".*requires the input passed should be a `jax.Array`.*"):
      with_sharding_constraint(np_inp, s_abs)

    with self.assertRaisesRegex(
        TypeError, "The sharding on the input must be a `NamedSharding`"):
      with_sharding_constraint(jnp.arange(8), s_abs)

    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x')))
    abs_mesh2 = jtu.create_mesh((2,), 'y').abstract_mesh
    with self.assertRaisesRegex(
        ValueError,
        'Mesh shape of the input.*does not'
        ' match the mesh shape of the target sharding.*'):
      with_sharding_constraint(arr, NamedSharding(abs_mesh2, P('y')))

  def test_global_jit_cpp_cache_hit_out_shardings(self):
    mesh = jtu.create_mesh((2,), 'x')
    s = NamedSharding(mesh, P('x'))

    def f(x):
      return x * 2

    with jtu.count_pjit_cpp_cache_miss() as count:
      jax.jit(f, out_shardings=s)(np.arange(8))
      jax.jit(f, out_shardings=s)(np.arange(8))
    self.assertEqual(count(), 1)

  def test_input_shardings_single_device(self):
    @jax.jit
    def f(x):
      return x * 2

    ins, _ = f.lower(np.arange(8)).compile().input_shardings
    self.assertEqual(ins[0], SingleDeviceSharding(jax.devices()[0]))

  def test_abstract_mesh_lower(self):
    mesh = jtu.create_mesh((2,), 'x')
    mesh2 = jtu.create_mesh((1,), 'x')

    abstract_sds = jax.ShapeDtypeStruct(
        (8, 2), jnp.float32, sharding=NamedSharding(mesh.abstract_mesh, P('x')))
    abstract_sds2 = jax.ShapeDtypeStruct(
        (8, 2), jnp.float32, sharding=NamedSharding(mesh2.abstract_mesh, P('x')))

    @jax.jit
    def f(x):
      return x * 2

    lowered = f.trace(abstract_sds).lower(lowering_platforms=('tpu',))
    self.assertIn('num_partitions = 2', lowered.as_text())

    with self.assertRaisesRegex(
        RuntimeError, 'A jitted computation cannot contain AbstractMesh'):
      lowered.compile()

    @jax.jit
    def g(x, y):
      return x, y

    concrete_s = NamedSharding(mesh, P('x'))
    concrete_sds = jax.ShapeDtypeStruct((8,), jnp.float32, sharding=concrete_s)
    with self.assertRaisesRegex(
        ValueError,
        'AbstractMesh size: 1 does not match the device assignment size: 2'):
      g.lower(abstract_sds2, concrete_sds)

    with self.assertRaisesRegex(
        ValueError, "Passing lowering_platforms.*is required"):
      g.lower(abstract_sds, np.arange(8))

    lowered2 = g.trace(abstract_sds, np.arange(8)).lower(
        lowering_platforms=('tpu',))
    self.assertIn('num_partitions = 2', lowered2.as_text())
    with self.assertRaisesRegex(
        RuntimeError, 'A jitted computation cannot contain AbstractMesh'):
      lowered2.compile()

    lowered3 = g.lower(abstract_sds, concrete_sds)
    self.assertIn('num_partitions = 2', lowered3.as_text())
    with self.assertRaisesRegex(
        RuntimeError, 'A jitted computation cannot contain AbstractMesh'):
      lowered3.compile()

  def test_jit_out_shardings_unconstrained(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    out_s = NamedSharding(mesh, P(P.UNCONSTRAINED, P.UNCONSTRAINED))
    @partial(jax.jit, out_shardings=out_s)
    def f(x):
      return x * 2

    out = f(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, np_inp * 2)

    @partial(jax.jit, out_shardings=NamedSharding(mesh, P(P.UNCONSTRAINED, 'y')))
    def g(x):
      return x * 3

    out = g(arr)
    self.assertArraysEqual(out, np_inp * 3)
    self.assertEqual(out.sharding, s)
    lowered_text = g.lower(arr).as_text()
    if config.use_shardy_partitioner.value:
      self.assertIn('<@mesh, [{?}, {"y"}]>', lowered_text)
    else:
      self.assertIn("unspecified_dims=[0]", lowered_text)

  def test_prng_key_wsc(self):
    mesh = jtu.create_mesh((2,), 'x')

    @jax.jit
    def f(x):
      y = lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
      return y.T
    f(jax.random.key(0))  # doesn't crash

    @jax.jit
    def g(x):
      return lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
    g(jax.random.key(1))  # doesn't crash

  def test_prng_key_wsc_multi_axes_sharding(self):
    input_shape = (8, 4)
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    @jax.jit
    def make_keys(seeds):
      make_key = partial(prng.random_seed, impl=prng.threefry_prng_impl)
      return lax.with_sharding_constraint(
          make_key(seeds), NamedSharding(mesh, P('x', 'y')))

    out = make_keys(seeds)
    self.assertTrue(jax.dtypes.issubdtype(out.dtype, jax.dtypes.prng_key))
    self.assertEqual(out.shape, input_shape)
    jax.random.key_data(out)  # doesn't crash

  def test_sds_update(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s1 = jax.ShapeDtypeStruct((2, 2), jnp.int32)
    s1_u = s1.update(shape=(4, 2), dtype=np.float32)
    self.assertEqual(s1_u.shape, (4, 2))
    self.assertEqual(s1_u.dtype, np.float32)
    self.assertFalse(s1_u.weak_type)

    s2 = jax.ShapeDtypeStruct((2, 2), jnp.int32)
    s2_u = s2.update(shape=(4, 2), weak_type=True)
    self.assertEqual(s2_u.shape, (4, 2))
    self.assertEqual(s2_u.dtype, np.int32)
    self.assertTrue(s2_u.weak_type)

    s3 = jax.ShapeDtypeStruct((2, 2), jnp.int32,
                              sharding=NamedSharding(mesh, P()))
    s3_u = s3.update(sharding=NamedSharding(mesh, P('x')))
    self.assertEqual(s3_u.sharding, NamedSharding(mesh, P('x')))

    s32_u = s3.update(shape=(4, 2))
    self.assertEqual(s32_u.shape, (4, 2))
    self.assertEqual(s32_u.sharding, NamedSharding(mesh, P()))

    sh = NamedSharding(mesh, P())
    s4 = jax.ShapeDtypeStruct((2, 2), jnp.int32,
                              sharding=Format(DLL((0, 1)), sh))
    new_layout = Format(DLL((1, 0)), NamedSharding(mesh, P('x')))
    s4_u = s4.update(sharding=new_layout)
    self.assertEqual(s4_u.sharding, new_layout.sharding)
    self.assertEqual(s4_u.format, new_layout)

    with self.assertRaisesRegex(ValueError, "updating ShapeDtypeStruct"):
      s4.update(sharding=NamedSharding(mesh, P('x')))

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'), axis_types=(AxisType.Auto,) * 2)
  def test_sds_pspec_input(self, mesh):
    inp = jax.ShapeDtypeStruct((2, 2), np.float32, sharding=P('x'))
    lowered = jax.jit(lambda x: x * 2).lower(inp)
    self.assertIn('num_partitions = 2', lowered.as_text())

    np_inp = np.arange(4, dtype=np.float32).reshape(2, 2)
    arr = jax.device_put(np_inp, P('x'))
    out = lowered.compile()(arr)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

  def test_sds_pspec_no_mesh_ctx_error(self):
    with self.assertRaisesRegex(
        TypeError,
        'When specifying PartitionSpec to `ShapeDtypeStruct`, the context mesh'
        ' cannot be empty'):
      jax.ShapeDtypeStruct((2, 2), np.float32, sharding=P('x'))


def spec_regex(s):
  return str(s).replace(r"(", r"\(").replace(r")", r"\)")


class ShardingInTypesTest(jtu.JaxTestCase):

  def check_wsc_in_lowered(self, text):
    if config.use_shardy_partitioner.value:
      self.assertIn('sdy.sharding_constraint', text)
    else:
      self.assertIn('@Sharding', text)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_basic_mul(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      self.assertEqual(jax.typeof(x).sharding.spec, s.spec)
      x = x * 2
      self.assertEqual(jax.typeof(x).sharding.spec, s.spec)
      x = x * x
      self.assertEqual(jax.typeof(x).sharding.spec, s.spec)
      return x

    # Eager mode
    out = f(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp * 2) * (np_inp * 2))

    f = jax.jit(f)

    out = f(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp * 2) * (np_inp * 2))

    sds = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=s)
    lowered_text = f.lower(sds).as_text()
    if config.use_shardy_partitioner.value:
      self.assertEqual(lowered_text.count('sdy.sharding_constraint'), 3)
    else:
      self.assertEqual(lowered_text.count('@Sharding'), 3)

    @jax.jit
    def g(x):
      x = f(x)
      return jnp.sum(x)

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

    jax.jit(jax.grad(g)).lower(sds)  # doesn't crash

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_fully_replicated_array_mul(self, mesh):
    np_inp1 = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr1 = jax.device_put(np_inp1, s)

    np_inp2 = np.arange(2).reshape(1, 2)
    arr2 = jax.device_put(np_inp2, NamedSharding(mesh, P(None, None)))

    @jax.jit
    def f(x, y):
      self.assertEqual(x.aval.sharding.spec, s.spec)
      out = x * y
      self.assertEqual(out.aval.sharding.spec, s.spec)
      return out

    out = f(arr1, arr2)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp1 * np_inp2))

    out = f(arr1, jax.device_put(np_inp1, NamedSharding(mesh, P(('x',), ('y',)))))
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp1 * np_inp1))

    out = f(arr1, jax.device_put(np_inp2, NamedSharding(mesh, P())))
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp1 * np_inp2))

    @jax.jit
    def g(x, y):
      return x * y

    with self.assertRaisesRegex(
        core.ShardingTypeError,
        "mul got incompatible shardings for broadcasting"):
      g(arr1, jax.device_put(np_inp1, NamedSharding(mesh, P('y', 'x'))))

    with self.assertRaisesRegex(
        core.ShardingTypeError,
        "mul got incompatible shardings for broadcasting"):
      g(arr1, jax.device_put(np_inp1, NamedSharding(mesh, P(('x', 'y')))))

  @parameterized.named_parameters(
      ('x_y', P('x', None), P(None, 'y'), P('x', 'y'), None),
      ('x_None', P('x', None), P(None, None), P('x', None), None),
      ('contracting2', P('x', 'y'), P(None, None), P('x', None), 'all-gather'),
      ('fsdp', P('x', None), P('x', None), P('x', None), 'all-gather'),
      ('half_tp', P(None, 'y'), P(None, 'y'), P(None, 'y'), 'all-gather'),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dot_general(self, spec1, spec2, out_spec, collective_name, mesh):
    np_inp1 = np.arange(16.).reshape(8, 2)
    arr1 = jax.device_put(np_inp1, NamedSharding(mesh, spec1))
    arr2 = jax.device_put(np_inp1.T, NamedSharding(mesh, spec2))

    def f(x, y):
      out = x @ y
      self.assertEqual(out.aval.sharding.spec, out_spec)
      return out

    out = f(arr1, arr2)
    self.assertArraysEqual(out, np_inp1 @ np_inp1.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))

    f = jax.jit(f)

    out = f(arr1, arr2)
    self.assertArraysEqual(out, np_inp1 @ np_inp1.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))

    lowered = f.lower(arr1, arr2)
    self.check_wsc_in_lowered(lowered.as_text())

    compiled_text = lowered.compile().as_text()
    if collective_name is not None and compiled_text is not None:
      self.assertIn(collective_name, compiled_text)

    @jax.jit
    def g(x, y):
      out = f(x, y)
      return jnp.sum(out)

    out = jax.grad(g, argnums=(0, 1))(arr1, arr2)
    self.assertEqual(out[0].sharding, arr1.sharding)
    self.assertEqual(out[1].sharding, arr2.sharding)

    out = jax.jit(jax.grad(g, argnums=(0, 1)))(arr1, arr2)
    self.assertEqual(out[0].sharding, arr1.sharding)
    self.assertEqual(out[1].sharding, arr2.sharding)

  @parameterized.parameters([True, False])
  @jtu.with_explicit_mesh((4,), ('x',))
  def test_dot_general_out_sharding(self, use_jit, mesh):
    np_inp1 = np.arange(16.).reshape(8, 2)
    arr1 = jax.device_put(np_inp1, NamedSharding(mesh, P('x', None)))
    arr2 = jax.device_put(np_inp1.T, NamedSharding(mesh, P(None, 'x')))

    def f(x, y):
      out = jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', None))
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return jnp.sum(out)

    if use_jit:
      f = jax.jit(f)

    out = f(arr1, arr2)
    self.assertArraysEqual(out, np.sum(np_inp1 @ np_inp1.T))
    self.assertEqual(out.sharding, NamedSharding(mesh, P()))

    with self.assertRaisesRegex(
        ValueError,
        'PartitionSpec passed to einsum cannot contain axis names that are of'
        ' type Auto or Manual'):
      auto_axes(f, out_sharding=P())(arr1, arr2)

    out = jax.grad(f, argnums=(0, 1))(arr1, arr2)
    self.assertEqual(out[0].sharding, arr1.sharding)
    self.assertEqual(out[1].sharding, arr2.sharding)

    if use_jit:
      jitted_grad = jax.jit(jax.grad(f, argnums=(0, 1)))
      out = jitted_grad(arr1, arr2)
      self.assertEqual(out[0].sharding, arr1.sharding)
      self.assertEqual(out[1].sharding, arr2.sharding)

      jaxpr = jitted_grad.trace(arr1, arr2).jaxpr
      bwd_jaxpr = jaxpr.eqns[-1]
      expected_spec = [('broadcast_in_dim', P('x', None)),
                      ('dot_general', P('x', None)),
                      ('transpose', P(None, 'x')),
                      ('dot_general', P('x', None))]
      for eqn, spec in zip(bwd_jaxpr.params['jaxpr'].eqns, expected_spec):
        self.assertEqual(eqn.primitive.name, spec[0])
        self.assertEqual(eqn.outvars[0].aval.sharding.spec, spec[1])

  @parameterized.named_parameters(
      ('fail1', P('x', None), P(None, 'x'),
       "dot_general operation.*produces an illegally sharded result",
       core.ShardingTypeError),
      ('fail2', P('x', 'y'), P('x', 'y'),
       "dot_general requires contracting dimensions to have consistent sharding",
       core.ShardingTypeError),
      ('contracting1', P('x', 'y'), P('y', None),
       'Contracting dimensions are sharded', core.ShardingTypeError),
      ('other_half_tp', P(None, 'y'), P('y', None),
       'Contracting dimensions are sharded', core.ShardingTypeError),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dot_general_error(self, spec1, spec2, error_msg, error_type, mesh):
    np_inp1 = np.arange(16).reshape(8, 2)
    arr1 = jax.device_put(np_inp1, NamedSharding(mesh, spec1))
    arr2 = jax.device_put(np_inp1.T, NamedSharding(mesh, spec2))

    @jax.jit
    def f(x, y):
      return x @ y

    with self.assertRaisesRegex(error_type, error_msg):
      f(arr1, arr2)

  @jtu.with_explicit_mesh((2, 2, 1), ('x', 'y', 'z'))
  def test_dot_general_batch_error(self, mesh):
    arr1 = jax.device_put(np.ones((8, 4, 2)),
                          NamedSharding(mesh, P('x', 'y', 'z')))
    arr2 = jax.device_put(np.ones((8, 2, 4)),
                          NamedSharding(mesh, P('y', 'z', 'x')))
    with self.assertRaisesRegex(
        core.ShardingTypeError,
        'dot_general requires lhs batch dimensions and rhs batch dimensions to'
        ' have the consistent sharding'):
      jax.lax.dot_general(
          arr1, arr2, dimension_numbers=(([2], [1]), ([0], [0])))

    with self.assertRaisesRegex(
        core.ShardingTypeError,
        'dot_general requires lhs batch dimensions and rhs batch dimensions to'
        ' have the consistent sharding'):
      jnp.einsum('abc,acz->abz', arr1, arr2)

  @jtu.with_explicit_mesh((2, 2), ('model', 'data'))
  def test_aval_repr(self, mesh):
    mesh = mesh.abstract_mesh
    aval = core.ShapedArray((128, 64), np.float32,
                            sharding=NamedSharding(mesh, P('model', 'data')))
    self.assertEqual(aval.str_short(), 'float32[128@model,64@data]')

    aval = aval.update(sharding=NamedSharding(mesh, P('model', None)))
    self.assertEqual(aval.str_short(), 'float32[128@model,64]')

    aval = aval.update(sharding=NamedSharding(mesh, P(None, 'data')))
    self.assertEqual(aval.str_short(), 'float32[128,64@data]')

    aval = aval.update(sharding=NamedSharding(mesh, P(None, None)))
    self.assertEqual(aval.str_short(), 'float32[128,64]')

    aval = aval.update(sharding=NamedSharding(mesh, P(('model', 'data'), None)))
    self.assertEqual(aval.str_short(), 'float32[128@(model,data),64]')

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'))
  def test_jnp_ones_mesh_context_eager(self, mesh):
    s = NamedSharding(mesh, P('x', None))
    out = jnp.ones((8, 2), dtype=jnp.int32, device=s)
    self.assertEqual(out.sharding, s)

    s = NamedSharding(mesh, P('x', 'y'))
    out = jnp.ones((8, 2), dtype=jnp.int32, device=s)
    self.assertEqual(out.sharding, s)

  @parameterized.named_parameters(
      ('all', None, P('x', 'y'), P(), True),
      ('first', 0, P('x', 'y'), P('y'), True),
      ('second', 1, P('x', 'y'), P('x'), True),
      ('first2', 0, P(('x', 'y'), None), P(None), True),
      ('second2', 1, P(('x', 'y'), None), P(('x', 'y')), False),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce_sum(self, axis, in_spec, out_spec, reduce, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, in_spec)
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      self.assertEqual(x.aval.sharding.spec, s.spec)
      y = jnp.sum(x, axis=axis)
      self.assertEqual(y.aval.sharding.spec, out_spec)
      return y

    out = f(arr)
    self.assertArraysEqual(out, np.sum(np_inp, axis=axis))
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))

    lowered = f.lower(arr)
    self.check_wsc_in_lowered(lowered.as_text())

    compiled_text = lowered.compile().as_text()
    if reduce and compiled_text is not None:
      self.assertIn('all-reduce', compiled_text)

  @parameterized.named_parameters(
      ('all', None, P('x', 'y'), P(), True),
      ('first', 0, P('x', 'y'), P('y'), True),
      ('second', 1, P('x', 'y'), P('x'), True),
      ('first2', 0, P(('x', 'y'), None), P(None), True),
      ('second2', 1, P(('x', 'y'), None), P(('x', 'y')), False),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reduce_max(self, axis, in_spec, out_spec, reduce, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, in_spec)
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      self.assertEqual(x.aval.sharding.spec, s.spec)
      y = jnp.max(x, axis=axis)
      self.assertEqual(y.aval.sharding.spec, out_spec)
      return y

    out = f(arr)
    self.assertArraysEqual(out, np.max(np_inp, axis=axis))
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))

    lowered = f.lower(arr)
    self.check_wsc_in_lowered(lowered.as_text())

    compiled_text = lowered.compile().as_text()
    if reduce and compiled_text is not None:
      self.assertIn('all-reduce', compiled_text)

    @jax.jit
    def g(x):
      out = f(x)
      return jnp.mean(out)

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

  @parameterized.named_parameters(
      ('0', 0, P(None, 'x', 'y')),
      ('1', 1, P('x', None, 'y')),
      ('2', 2, P('x', 'y', None)),
      ('-1', -1, P('x', 'y', None)),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_broadcast_in_dim(self, axis, out_spec, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    out = jnp.expand_dims(arr, axis=axis)
    self.assertEqual(out.aval.sharding.spec, out_spec)

    @jax.jit
    def f(x):
      y = jnp.expand_dims(x, axis=axis)
      self.assertEqual(y.aval.sharding.spec, out_spec)
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))

    lowered_text = f.lower(arr).as_text()
    self.check_wsc_in_lowered(lowered_text)

  @parameterized.named_parameters(
      ('2', 2),
      ('3', 3),
      ('4', 4),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_integer_pow(self, pow, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x ** pow
      self.assertEqual(y.aval.sharding.spec, s.spec)
      return y

    out = f(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, np_inp ** pow)

    lowered_text = f.lower(arr).as_text()
    self.check_wsc_in_lowered(lowered_text)

  @jtu.with_explicit_mesh((1,), 'x')
  def test_broadcasting_nary_error(self, mesh):
    mesh2 = Mesh([jax.devices()[0]], 'y',
                 axis_types=(mesh_lib.AxisType.Explicit,))

    arr1 = jax.device_put(np.arange(8), NamedSharding(mesh, P()))
    arr2 = jax.device_put(np.arange(8), NamedSharding(mesh2, P()))

    @jax.jit
    def f(x, y):
      return x + y

    with self.assertRaisesRegex(
        ValueError, "For primitive.*context mesh.*aval mesh"):
      f(arr1, arr2)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_sin_unop(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = lax.sin(x)
      self.assertEqual(y.aval.sharding.spec, s.spec)
      return y

    out = f(arr)
    self.assertEqual(out.sharding, s)

    lowered_text = f.lower(arr).as_text()
    self.check_wsc_in_lowered(lowered_text)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_jnp_array(self, mesh):
    np_inp = np.arange(16, dtype=jnp.int32).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      assert x.dtype == jnp.int32
      y = jnp.array(x, dtype=jnp.float32)
      self.assertEqual(y.dtype, jnp.float32)
      self.assertEqual(y.aval.sharding.spec, s.spec)
      return y

    f(arr)

  @jtu.with_explicit_mesh((2, 2, 1), ('x', 'y', 'z'))
  def test_lax_transpose_rule(self, mesh):
    np_inp = np.arange(16).reshape(4, 2, 2)
    s = NamedSharding(mesh, P('x', 'y', 'z'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = jnp.transpose(x, (1, 2, 0))
      self.assertEqual(y.aval.sharding.spec, P('y', 'z', 'x'))
      return y

    out = f(arr)
    self.assertArraysEqual(out, np.transpose(arr, (1, 2, 0)))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'z', 'x')))

    lowered_text = f.lower(arr).as_text()
    self.check_wsc_in_lowered(lowered_text)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_broadcasted_iota_with_sharding(self, mesh):
    np_inp = np.arange(4)
    s = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = jax.nn.one_hot(x, 4)
      self.assertEqual(y.aval.sharding.spec, P('x', None))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def g(x):
      x = x * 2
      y = jax.lax.broadcasted_iota(x.dtype, (8, 2), 0, out_sharding=P('x', 'y'))
      self.assertEqual(y.aval.sharding.spec, P('x', 'y'))
      return x, y

    _, out = g(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_einsum_with_out_sharding(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    arr1 = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', 'x')))

    @jax.jit
    def f(x, y):
      out = jnp.einsum('xy,yz->xz', x, y,
                       out_sharding=NamedSharding(x.aval.sharding.mesh, P('x', None)))
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    out = f(arr1, arr2)
    self.assertArraysEqual(out, np_inp @ np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    lowered_text = f.lower(arr1, arr2).as_text()
    self.check_wsc_in_lowered(lowered_text)

    @jax.jit
    def g(x, y):
      out = jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', None))
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    arr3 = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr4 = jax.device_put(np_inp.T, NamedSharding(mesh, P('x', 'y')))
    out2 = g(arr3, arr4)
    self.assertArraysEqual(out2, np_inp @ np_inp.T)
    self.assertEqual(out2.sharding, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def h2(x, y):
      out = g(x, y)
      return jnp.sum(out)

    out = jax.grad(h2, argnums=(0, 1))(arr3, arr4)
    self.assertEqual(out[0].sharding, arr3.sharding)
    self.assertEqual(out[1].sharding, arr4.sharding)

    out = jax.jit(jax.grad(h2, argnums=(0, 1)))(arr3, arr4)
    self.assertEqual(out[0].sharding, arr3.sharding)
    self.assertEqual(out[1].sharding, arr4.sharding)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_einsum_inverse(self, mesh):
    np_inp = np.arange(64.)

    @jax.jit
    def h(x, y):
      spec = P('x', None, 'y', None)
      out = jnp.einsum('btd,dhq->bhtq', x, y, out_sharding=spec)
      self.assertEqual(out.aval.sharding.spec, spec)
      return out

    arr1 = jax.device_put(np_inp.reshape(8, 4, 2),
                          NamedSharding(mesh, P('x', 'y', None)))
    arr2 = jax.device_put(np_inp.reshape(2, 4, 8),
                          NamedSharding(mesh, P(None, 'x', 'y')))
    out = h(arr1, arr2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None, 'y', None)))

    lowered_text = h.lower(arr1, arr2).as_text()
    self.check_wsc_in_lowered(lowered_text)

    @jax.jit
    def h2(x, y):
      out = h(x, y)
      return jnp.sum(out)

    out = jax.grad(h2, argnums=(0, 1))(arr1, arr2)
    self.assertEqual(out[0].sharding, arr1.sharding)
    self.assertEqual(out[1].sharding, arr2.sharding)

    out = jax.jit(jax.grad(h2, argnums=(0, 1)))(arr1, arr2)
    self.assertEqual(out[0].sharding, arr1.sharding)
    self.assertEqual(out[1].sharding, arr2.sharding)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_fully_replicated_reshape(self, mesh):
    np_inp = np.arange(64).reshape(64, 1)
    arr = jax.device_put(np_inp, P(('x', 'y')))

    @jax.jit
    def f(x):
      x = reshard(x, P(None, None))
      return jax.lax.reshape(x, (2, 32, 1))

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, None, None)))
    self.assertArraysEqual(out, np_inp.reshape(2, 32, 1))

  @parameterized.parameters(
      (src_shape, dst_shape, src_spec, dst_spec, use_sharding_arg, fun)
      for fun in [jnp.reshape, jax.lax.reshape]
      for src_shape, dst_shape, src_spec, dst_spec, use_sharding_arg in [
        ((16, 1), (1, 16, 1), P('x', None), P(None, 'x', None),
         False),
        ((8, 2, 1), (1, 16, 1), P('x', None, None),
         P(None, 'x', None), True),
        ((8, 1), (1, 4, 2), P('x', None), P(None, None, 'x'),
         True),
        ((1, 4, 1, 6, 1), (1, 4, 6),
         P(None, 'x', None, None, None), P(None, 'x', None), False),
        ((4, 6), (4, 6), P(None, 'x'), P(None, 'x'), False),
        ((1024, 4096), (1024, 2048, 2, 1, 1, 1, 1),
         P('x', None), P('x', None, None, None, None, None, None), False),
        ((1024, 4096, 32), (1024, 2048, 2, 1, 1, 32),
         P('x', None, None), P('x', None, None, None, None, None), False),
        ((1024, 4096), (1024, 1, 1, 4096),
         P('x', None), P('x', None, None, None), False),
        ((1024, 4096), (1024, 1, 1, 4096),
         P(None, 'x'), P(None, None, None, 'x'), False),
        ((1024, 2048, 2, 1, 1, 1), (1024, 4096),
         P('x', None, None, None, None, None), P('x', None), False),
        ((1024, 2048, 2, 1, 1, 1), (1024, 4096),
         P(None, 'x', None, None, None, None), P(None, 'x'), False),
      ]
  )
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_reshape(self, src_shape, dst_shape, src_spec, dst_spec,
                   use_sharding_arg, fun, mesh):
    np_inp = np.arange(math.prod(src_shape),
                       dtype=np.float32).reshape(src_shape)
    arr = jax.device_put(np_inp, NamedSharding(mesh, src_spec))

    @partial(jax.jit, static_argnums=1)
    def f(x, new_sharding):
      y = fun(x, dst_shape, out_sharding=new_sharding)
      self.assertEqual(y.aval.sharding.spec, dst_spec)
      self.assertEqual(y.shape, dst_shape)
      y = y * 2
      self.assertEqual(y.aval.sharding.spec, dst_spec)
      return y

    new_s = dst_spec if use_sharding_arg else None
    out = f(arr, new_s)
    self.assertEqual(out.sharding, NamedSharding(mesh, dst_spec))
    self.assertArraysEqual(out, np_inp.reshape(dst_shape) * 2)

    lowered_text = f.lower(arr, new_s).as_text()
    self.check_wsc_in_lowered(lowered_text)

    def g(x):
      out = f(x, new_s)
      return jnp.square(jnp.sum(out))

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

  @parameterized.named_parameters(
      ('split_1', (4, 6, 8), (4, 2, 3, 8),
       P('x', None, 'y'), P('x', None, None, 'y'), ''
      ),
      ('split_2', (4, 6, 8), (4, 6, 2, 2, 2),
       P('x', None, None), P('x', None, None, None, None), ''
      ),
      ('split_3_error', (4, 6, 8), (4, 2, 3, 4, 2),
       P('x', None, None), P('x', None, None, None, None),
       'Splitting on more than 1 axis is not supported'
      ),
      ('split_4', (4, 6, 8), (4, 2, 3, 8),
       P('x', 'y', None), P('x', 'y', None, None), ''
      ),
      ('split_4_xy', (4, 12, 8), (4, 4, 3, 8),
       P(None, ('x', 'y'), None), P(None, ('x', 'y'), None, None), ''
      ),
      ('split_4_error', (4, 6, 8), (4, 3, 2, 8),
       P('x', 'y', None), None, 'This reshape is not supported'
      ),
      ('split_5_error', (4, 6, 8), (4, 4, 2, 6),
       P('x', None, None), None, 'This reshape is not supported'
      ),
      ('split_6_error', (4, 8, 9), (4, 2, 2, 3, 3, 2),
       P('x', None, None), None, 'This reshape is not supported'
      ),
      ('merge_1', (4, 2, 3, 8), (4, 6, 8),
       P('x', None, None, 'y'), P('x', None, 'y'), ''
      ),
      ('merge_2', (2, 2, 6, 8), (4, 6, 8),
       P(None, None, 'y', 'x'), P(None, 'y', 'x'), ''
      ),
      ('merge_3', (4, 6, 2, 2, 2), (4, 6, 8),
       P('x', None, None, None, None), P('x', None, None), ''
      ),
      ('merge_4', (4, 2, 3, 8), (4, 6, 8),
       P(None, 'y', None, 'x'), P(None, 'y', 'x'), ''
      ),
      ('merge_4_xy', (4, 4, 3, 8), (4, 12, 8),
       P(None, ('x', 'y'), None, None), P(None, ('x', 'y'), None), ''
      ),
      ('merge_4_error', (4, 2, 3, 2, 4), (4, 6, 8),
       P('x', None, None, None, None), P('x', None, None),
       'Merging on more than 1 axis is not supported'
      ),
      ('merge_5_error', (4, 2, 6, 8), (4, 12, 8),
       P(None, None, 'y', 'x'), None, 'This reshape is not supported'
      ),
      ('merge_6_error', (4, 2, 3, 8), (4, 8, 6),
       P(None, 'y', None, 'x'), None, 'This reshape is not supported'
      ),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshape_split_merge_one_axis(self, src_shape, dst_shape, src_spec,
                                        dst_spec, error_msg, mesh):
    np_inp = np.arange(math.prod(src_shape),
                       dtype=np.float32).reshape(src_shape)
    arr = jax.device_put(np_inp, NamedSharding(mesh, src_spec))

    @jax.jit
    def f(x):
      y = lax.reshape(x, dst_shape)
      y = y * 2
      self.assertEqual(y.aval.sharding.spec, dst_spec)
      return y

    if error_msg:
      with self.assertRaisesRegex(core.ShardingTypeError, error_msg):
        f(arr)
    else:
      out = f(arr)
      self.assertEqual(out.sharding, NamedSharding(mesh, dst_spec))
      self.assertArraysEqual(out, np_inp.reshape(dst_shape) * 2)

      lowered_text = f.lower(arr).as_text()
      self.check_wsc_in_lowered(lowered_text)

      def g(x):
        out = f(x)
        return jnp.square(jnp.sum(out))

      out = jax.jit(jax.grad(g))(arr)
      self.assertEqual(out.sharding, arr.sharding)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_select(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr1 = jax.device_put(np_inp, s)
    arr2 = jax.device_put(np_inp, s)

    @jax.jit
    def f(pred, on_true, on_false):
      y = lax.select(pred, on_true, on_false)
      self.assertEqual(y.aval.sharding.spec, s.spec)
      return y

    out = f(arr1 == arr2, arr1, arr2)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, arr1)

    lowered_text = f.lower(arr1 == arr2, arr1, arr2).as_text()
    self.check_wsc_in_lowered(lowered_text)

    arr3 = jax.device_put(np_inp, NamedSharding(mesh, P('y', 'x')))
    with self.assertRaisesRegex(
        core.ShardingTypeError, "select cases must have the same shardings"):
      f(arr1 == arr2, arr1, arr3)

  def test_explicit_mode_no_context_mesh(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    abstract_mesh = mesh.abstract_mesh
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', None)))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P(None, 'y')))

    @jax.jit
    def f(x, y):
      self.assertEqual(x.aval.sharding.spec, P('x', None))
      self.assertEqual(x.aval.sharding.mesh, abstract_mesh)
      self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
      self.assertEqual(y.aval.sharding.mesh, abstract_mesh)
      z = x @ y
      self.assertEqual(z.aval.sharding.spec, P('x', 'y'))
      self.assertEqual(z.aval.sharding.mesh, abstract_mesh)
      a = z * 2
      self.assertEqual(a.aval.sharding.spec, P('x', 'y'))
      self.assertEqual(a.aval.sharding.mesh, abstract_mesh)
      return a

    out = f(arr, arr2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  def test_auto_mode_no_context_mesh(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'),
                           axis_types=(AxisType.Auto,) * 2)
    abstract_mesh = mesh.abstract_mesh
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def f(x):
      self.assertEqual(x.aval.sharding.spec, P(None, None))
      self.assertEqual(x.aval.sharding.mesh, abstract_mesh)
      a = x * 2
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      self.assertEqual(a.aval.sharding.mesh, abstract_mesh)
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_mesh_cast_reshard_error(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = mesh_cast(x, NamedSharding(x.aval.sharding.mesh, P('x', None)))
      return y

    with self.assertRaisesRegex(
        ValueError,
        'mesh_cast should only be used when AxisType changes between the input'
        ' mesh and the target mesh'):
      f(arr)

    @jax.jit
    def g(x):
      return mesh_cast(x, P('x', None))

    with self.assertRaisesRegex(
        ValueError,
        'mesh_cast should only be used when AxisType changes between the input'
        ' mesh and the target mesh'):
      g(arr)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(AxisType.Explicit, AxisType.Auto))
  def test_mesh_cast_explicit_data_movement_error(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)
    full_user_mesh = mesh_lib.AbstractMesh(
        (2, 2), ('x', 'y'), axis_types=(AxisType.Explicit,) * 2)

    @jax.jit
    def f(x):
      return mesh_cast(x, NamedSharding(full_user_mesh, P('y', None)))

    with self.assertRaisesRegex(
        ValueError, 'Explicit data movement in mesh_cast is not allowed'):
      f(arr)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_shard_map_full_manual(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))

    def g(x, y):
      self.assertTrue(x.aval.sharding.mesh._are_all_axes_manual)
      self.assertTrue(y.aval.sharding.mesh._are_all_axes_manual)
      self.assertTrue(mesh_lib.get_abstract_mesh()._are_all_axes_manual)
      return x * y

    @jax.jit
    def f(x, y):
      z = shard_map(g, mesh=mesh,
                    in_specs=(x.aval.sharding.spec, y.aval.sharding.spec),
                    out_specs=P('x', 'y'))(x, y)
      self.assertEqual(z.aval.sharding.spec, P('x', 'y'))
      out = z * 2
      self.assertEqual(out.aval.sharding.spec, P('x', 'y'))
      return out

    out = f(arr, arr2)
    self.assertArraysEqual(out, (np_inp * np_inp) * 2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_shard_map_dot(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', 'x')))

    def g(x, y):
      self.assertTrue(x.aval.sharding.mesh._are_all_axes_manual)
      self.assertTrue(y.aval.sharding.mesh._are_all_axes_manual)
      self.assertTrue(mesh_lib.get_abstract_mesh()._are_all_axes_manual)
      allgatherd_y = jax.lax.all_gather(y, axis_name='x', axis=1, tiled=True)
      z = x @ allgatherd_y
      return jax.lax.psum(z, axis_name='y')

    @jax.jit
    def f(x, y):
      z = shard_map(g, mesh=mesh,
                    in_specs=(x.aval.sharding.spec, y.aval.sharding.spec),
                    out_specs=P('x', None))(x, y)
      self.assertEqual(z.aval.sharding.spec, P('x', None))
      out = z * 2
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    out = f(arr, arr2)
    self.assertArraysEqual(out, (np_inp @ np_inp.T) * 2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  def test_full_like_eager_non_concrete_sharding(self):
    s = NamedSharding(mesh_lib.AbstractMesh((2,), ('x',)), P('x'))
    arr = jax.ShapeDtypeStruct((8, 2), np.float32, sharding=s)
    out = jax.lax.full_like(arr, 0)
    # The sharding is single device because the sharding of input `arr`` to
    # full_like is not concrete.
    self.assertEqual(out.sharding, SingleDeviceSharding(jax.devices()[0]))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_slice(self, mesh):
    np_inp = np.arange(16.).reshape(4, 4)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def f(x):
      y = lax.slice(x, (0, 0), (4, 3))
      self.assertEqual(y.aval.sharding.spec, P('x', None))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))
    self.check_wsc_in_lowered(f.lower(arr).as_text())

    def g(x):
      out = f(x)
      return jnp.square(jnp.sum(out))

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

    with self.assertRaisesRegex(core.ShardingTypeError, "slicing on sharded dims"):
      f(jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y'))))

    with self.assertRaisesRegex(core.ShardingTypeError, "slicing on sharded dims"):
      f(jax.device_put(np_inp, NamedSharding(mesh, P(None, ('x', 'y')))))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_squeeze(self, mesh):
    np_inp = np.arange(16.).reshape(4, 4, 1)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', None, None)))

    @jax.jit
    def f(x):
      y = lax.squeeze(x, (2,))
      self.assertEqual(y.aval.sharding.spec, P('x', None))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))
    self.check_wsc_in_lowered(f.lower(arr).as_text())
    self.assertArraysEqual(out, np.squeeze(np_inp, axis=2))

    def g(x):
      out = f(x)
      return jnp.square(jnp.sum(out))

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_pad(self, mesh):
    np_inp = np.arange(8.)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x')))

    @partial(jax.jit, static_argnums=(1, 2))
    def f(x, padding_config, spec):
      y = lax.pad(x, 0., padding_config)
      self.assertEqual(y.aval.sharding.spec, spec)
      return y

    out = f(arr, ((2, 2, 0),), P('x'))
    self.assertArraysEqual(out, np.pad(np_inp, 2))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    self.check_wsc_in_lowered(f.lower(arr, ((2, 2, 0),), P('x')).as_text())

    out = f(arr, ((0, 0, 0),), P('x'))
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

    f(arr, ((0, 3, 1), ), P('x'))  # doesn't crash

    def g(x):
      out = f(x, ((2, 2, 0),), P('x'))
      return jnp.square(jnp.sum(out))

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

    with self.assertRaisesRegex(core.ShardingTypeError, "padding on sharded dims"):
      f(arr, ((2, 3, 0), ), None)

    with self.assertRaisesRegex(core.ShardingTypeError, "padding on sharded dims"):
      f(arr, ((0, 3, 0), ), None)

    with self.assertRaisesRegex(core.ShardingTypeError, "padding on sharded dims"):
      arr = jax.device_put(np_inp, NamedSharding(mesh, P(('x', 'y'))))
      f(arr, ((4, 4, 1),), None)

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'))
  def test_concatenate(self, mesh):
    np_inp = np.arange(16.).reshape(4, 4)
    s = NamedSharding(mesh, P('x', 'y'))
    arr1 = jax.device_put(np_inp, s)
    arr2 = jax.device_put(np.arange(4.).reshape(4, 1), s)

    @partial(jax.jit, static_argnums=2)
    def f(x, y, method='jnp'):
      if method == 'jnp':
        y = jnp.concatenate([x, y], axis=1)
      else:
        assert method == 'lax'
        y = lax.concatenate([x, y], dimension=1)
      self.assertEqual(y.aval.sharding.spec, P('x', 'y'))
      return y

    out = f(arr1, arr2)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, np.concatenate([arr1, arr2], axis=1))
    self.check_wsc_in_lowered(f.lower(arr1, arr2).as_text())

    out = f(arr1, arr2, method='lax')
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, np.concatenate([arr1, arr2], axis=1))

    with self.assertRaisesRegex(
        core.ShardingTypeError, "All operands should have the same sharding"):
      arr3 = jax.device_put(np.arange(4.).reshape(4, 1),
                            NamedSharding(mesh, P('x')))
      f(arr1, arr3)

    def g(x, y):
      out = f(x, y)
      return jnp.square(jnp.sum(out))

    out = jax.grad(g)(arr1, arr2)
    self.assertEqual(out.sharding, s)

    out = jax.jit(jax.grad(g))(arr1, arr2)
    self.assertEqual(out.sharding, s)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_scan(self, mesh):
    carry = jax.device_put(np.arange(16.).reshape(2, 8),
                           NamedSharding(mesh, P(None, 'x')))
    arr = jax.device_put(np.arange(128.).reshape(8, 8, 2),
                         NamedSharding(mesh, P(None, 'x', 'y')))

    @jax.jit
    def f(carry, xs):
      def g(carry, x):
        self.assertEqual(carry.aval.sharding.spec, P(None, 'x'))
        self.assertEqual(x.aval.sharding.spec, P('x', 'y'))
        y = jnp.einsum('xy,yz->xz', carry, x, out_sharding=P(None, 'y'))
        self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
        z = jax.nn.relu(y)
        self.assertEqual(z.aval.sharding.spec, P(None, 'y'))
        a = jnp.einsum('xy,yz->xz', z, x.T, out_sharding=P(None, 'x'))
        self.assertEqual(a.aval.sharding.spec, P(None, 'x'))
        return a, y
      return jax.lax.scan(g, carry, xs)

    activation, mean = f(carry, arr)
    self.assertEqual(activation.sharding, NamedSharding(mesh, P(None, 'x')))
    self.assertEqual(mean.sharding, NamedSharding(mesh, P(None, None, 'y')))

    f.lower(carry, arr).compile()(carry, arr)  # doesn't crash

    def g(carry, arr):
      out = f(carry, arr)
      return jnp.sum(out[0])
    out = jax.jit(jax.grad(g, argnums=(0, 1)))(carry, arr)
    self.assertEqual(out[0].sharding, carry.sharding)
    self.assertEqual(out[1].sharding, arr.sharding)

    with self.assertRaisesRegex(
        ValueError, "0th dimension of all xs should be replicated"):
      f(carry, jax.device_put(arr, NamedSharding(mesh, P('x', None, None))))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_argminmax(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      z = jnp.argmax(x, axis=0)
      self.assertEqual(z.aval.sharding.spec, P('y'))
      a = jnp.argmin(x, axis=1)
      self.assertEqual(a.aval.sharding.spec, P('x'))
      return z, a

    out1, out2 = f(arr)
    self.assertArraysEqual(out1, np.argmax(np_inp, axis=0))
    self.assertEqual(out1.sharding, NamedSharding(mesh, P('y')))
    self.assertArraysEqual(out2, np.argmin(np_inp, axis=1))
    self.assertEqual(out2.sharding, NamedSharding(mesh, P('x')))
    self.check_wsc_in_lowered(f.lower(arr).as_text())

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'), (mesh_lib.AxisType.Auto,) * 2)
  def test_only_auto(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def f(x, x2):
      y = x * 2
      self.assertEqual(y.aval.sharding.spec, P(None, None))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P(None, None))
      a = z @ x2
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      return a

    out = f(arr, arr.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  def test_auto_user(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'),
                           axis_types=(mesh_lib.AxisType.Auto,) * 2)
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x, x2):
      y = x * 2
      z = jnp.sin(y)
      a = z @ x2
      return a

    with jax.sharding.use_mesh(mesh):
      out = f(arr, arr.T)
      self.assertEqual(out.sharding, NamedSharding(mesh, P('x',)))
      lowered_text = f.lower(arr, arr.T).as_text()
      self.assertNotIn('unspecified_dims', lowered_text)

    mesh2 = jtu.create_mesh((2, 2), ('x', 'y'),
                            axis_types=(mesh_lib.AxisType.Explicit,
                                        mesh_lib.AxisType.Auto))
    with jax.sharding.use_mesh(mesh2):
      arr = jax.device_put(arr, NamedSharding(mesh2, P('x', 'y')))
      arr2 = jax.device_put(np_inp.T, NamedSharding(mesh2, P('y', None)))
      out = f(arr, arr2)
      self.assertEqual(out.sharding, NamedSharding(mesh2, P('x',)))
      lowered_text = f.lower(arr, arr2).as_text()
      if config.use_shardy_partitioner.value:
        self.assertTrue(lowered_text.count("{?}") == 5)
      else:
        self.assertTrue(lowered_text.count("unspecified_dims") == 5)

    mesh3 = jtu.create_mesh((2, 2), ('x', 'y'),
                            axis_types=(mesh_lib.AxisType.Auto,
                                        mesh_lib.AxisType.Explicit))
    with jax.sharding.use_mesh(mesh3):
      arr = jax.device_put(arr, NamedSharding(mesh3, P('x', 'y')))
      arr2 = jax.device_put(np_inp.T, NamedSharding(mesh3, P(None, 'x')))
      out = f(arr, arr2)
      self.assertEqual(out.sharding, NamedSharding(mesh3, P('x',)))
      lowered_text = f.lower(arr, arr2).as_text()
      if config.use_shardy_partitioner.value:
        self.assertTrue(lowered_text.count("{?}") == 5)
        self.assertIn('replicated={"y"}', lowered_text)
      else:
        self.assertTrue(lowered_text.count("unspecified_dims") == 4)

    with self.assertRaisesRegex(
        ValueError,
        "AxisTypes should be the same in a tuple subset of PartitionSpec"):
      NamedSharding(mesh2, P(('x', 'y')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_where_with_scalar(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    x = jax.device_put(np_inp, s)

    out = jnp.where(x > 0, x, 0)
    self.assertArraysEqual(out, x)
    self.assertEqual(out.sharding, s)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_full_user_to_full_auto(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x * 2
      with use_auto_axes('x', 'y'):
        y = mesh_cast(y, P(None, None))
        self.assertEqual(y.aval.sharding.spec, P(None, None))
        z = jnp.sin(y)
        self.assertEqual(z.aval.sharding.spec, P(None, None))
        a = z @ z.T
        self.assertEqual(a.aval.sharding.spec, P(None, None))
      a = mesh_cast(a, P('x', None))
      self.assertEqual(a.aval.sharding.spec, P('x', None))
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    jaxpr = f.trace(arr).jaxpr
    out2 = core.jaxpr_as_fun(jaxpr)(arr)
    self.assertEqual(out2[0].sharding, NamedSharding(mesh, P('x', None)))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(mesh_lib.AxisType.Auto,) * 2)
  def test_full_auto_to_full_user(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x * 2
      with use_explicit_axes('x', 'y'):
        y = mesh_cast(y, P(None, 'y'))
        self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
        z = jnp.sin(y)
        self.assertEqual(z.aval.sharding.spec, P(None, 'y'))
      a = mesh_cast(z, P(None, None))
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y')))

    jaxpr = f.trace(arr).jaxpr
    core.jaxpr_as_fun(jaxpr)(arr)  # doesn't crash

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_full_user_to_auto_user_mix(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x * 2
      with use_auto_axes('x'):
        y = mesh_cast(y, P(None, 'y'))
        self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
        z = jnp.sin(y)
        self.assertEqual(z.aval.sharding.spec, P(None, 'y'))
        a = jnp.einsum('xy,yz->xz', z, z.T, out_sharding=P(None, None))
        self.assertEqual(a.aval.sharding.spec, P(None, None))
      a = mesh_cast(a, P('x', None))
      self.assertEqual(a.aval.sharding.spec, P('x', None))
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    jaxpr = f.trace(arr).jaxpr
    out2 = core.jaxpr_as_fun(jaxpr)(arr)
    self.assertEqual(out2[0].sharding, NamedSharding(mesh, P('x', None)))

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'))
  def test_user_auto_mix_error(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x, y):
      x = x * 2
      with use_auto_axes('x'):
        z = x @ y
      return z

    with self.assertRaisesRegex(
        ValueError, "For primitive dot_general, context mesh.*aval mesh"):
      f(arr, arr.T)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_split(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @partial(jax.jit, static_argnums=(1, 2))
    def f(x, sizes=(4, 4), axis=0):
      ys = lax.split(x, sizes, axis=axis)
      self.assertEqual(ys[0].aval.sharding.spec, P('x', 'y'))
      self.assertEqual(ys[1].aval.sharding.spec, P('x', 'y'))
      return ys

    f(arr)
    self.check_wsc_in_lowered(f.lower(arr).as_text())

    with self.assertRaisesRegex(core.ShardingTypeError, "split on sharded dims"):
      f(arr, sizes=(1, 1), axis=1)

    def g(x):
      out = f(x)
      return jnp.square(jnp.sum(jnp.stack(out)))

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, s)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, s)

  @jtu.with_explicit_mesh((2,), 'x')
  def test_return_output_different_context(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      with use_auto_axes('x'):
        x = mesh_cast(x, P(None, None))
        return x

    self.assertDictEqual(arr.sharding.mesh._axis_types_dict,
                         {AxisType.Explicit: ('x',)})
    out = f(arr)
    self.assertArraysEqual(out, np_inp)
    self.assertDictEqual(out.sharding.mesh._axis_types_dict,
                         {AxisType.Auto: ('x',)})

  @jtu.with_explicit_mesh((2,), 'x')
  def test_device_put_use_mesh(self, mesh):
    out = jax.device_put(np.arange(8), P('x'))
    self.assertArraysEqual(out, np.arange(8))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

  def test_device_put_no_use_mesh_error(self):
    with self.assertRaisesRegex(
        ValueError,
        'Please set a mesh via `jax.sharding.use_mesh` if a PartitionSpec is'
        ' passed to device_put'):
      jax.device_put(np.arange(8), P('x'))

  @jtu.with_explicit_mesh((2,), 'x')
  def test_inputs_different_context(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np_inp, s)

    auto_mesh = jax.make_mesh((2,), 'x', axis_types=(AxisType.Auto,))
    with jax.sharding.use_mesh(auto_mesh):
      arr2 = jnp.ones(8)
    self.assertDictEqual(arr2.sharding.mesh._axis_types_dict,
                         {AxisType.Auto: ('x',)})

    @jax.jit
    def f(x, y):
      return x, y

    out1, out2 = f(arr, arr2)
    self.assertDictEqual(out1.sharding.mesh._axis_types_dict,
                         {AxisType.Explicit: ('x',)})
    self.assertDictEqual(out2.sharding.mesh._axis_types_dict,
                         {AxisType.Auto: ('x',)})

  @jtu.with_explicit_mesh((2,), 'x')
  def test_output_different_context_error(self, mesh):
    np_inp1 = np.arange(16).reshape(8, 2)
    arr1 = jax.device_put(np_inp1, NamedSharding(mesh, P('x', None)))
    arr2 = jax.device_put(np_inp1.T, NamedSharding(mesh, P(None, 'x')))
    auto_mesh = jax.make_mesh((2,), 'x', axis_types=(AxisType.Auto,)).abstract_mesh

    @jax.jit
    def f(x, y):
      out = jnp.einsum('xy,yz->xz', x, y,
                       out_sharding=NamedSharding(auto_mesh, P(None, None)))
      return out

    with self.assertRaisesRegex(
        ValueError, "Context mesh.*should match the mesh of sharding"):
      f(arr1, arr2)

    @jax.jit
    def g(x, y):
      with use_auto_axes('x'):
        out = jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', None))
      return out

    with self.assertRaisesRegex(
        ValueError, "PartitionSpec.*cannot contain axis names.*Auto"):
      g(arr1, arr2)

  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'),
                      axis_types=(AxisType.Explicit, AxisType.Explicit,
                                  AxisType.Auto))
  def test_out_sharding_mix_axis_types(self, mesh):
    np_inp = np.arange(16).reshape(4, 2, 2)
    s = NamedSharding(mesh, P('x', None, None))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x * 2
      self.assertEqual(y.aval.sharding.spec, P('x', None, None))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x',)))
    self.assertArraysEqual(out, np_inp * 2)

    lowered_text = f.lower(arr).as_text()
    if config.use_shardy_partitioner.value:
      self.assertTrue(lowered_text.count(
          '[{"x"}, {?}, {?}], replicated={"y"}') == 3)
    else:
      self.assertTrue(lowered_text.count("unspecified_dims=[1,2]") == 3)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_auto_mode_mix(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @partial(auto_axes, axes='x', out_sharding=P('x', None))
    def h(y):
      self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P(None, 'y'))
      a = jnp.einsum('xy,yz->xz', z, z.T, out_sharding=P(None, None))
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      return a

    @jax.jit
    def g(x):
      y = x * 2
      a = h(y)
      self.assertEqual(a.aval.sharding.spec, P('x', None))
      return a

    out = g(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    jaxpr = g.trace(arr).jaxpr
    out2 = core.jaxpr_as_fun(jaxpr)(arr)
    self.assertEqual(out2[0].sharding, NamedSharding(mesh, P('x', None)))

  @jtu.with_explicit_mesh((4,), ('x',))
  def test_concat_vmap(self, mesh):
    @jax.jit
    def _f(sharded_array, replicated_array):
      def _single_array(a, b):
        return jnp.concatenate([a, b], axis=-1)

      _first_vmap = jax.vmap(_single_array, in_axes=(None, 0))
      _second_vmap = jax.vmap(_first_vmap, in_axes=(0, None))
      return jax.vmap(_second_vmap, in_axes=(0, None))(sharded_array, replicated_array)

    np_inp = np.ones((4 * 4, 10, 5, 4))
    arr1 = jax.device_put(np_inp, NamedSharding(mesh, P('x')))
    arr2 = jax.device_put(
        jnp.ones((10, 5, 3)), NamedSharding(mesh, P()))

    out = _f(arr1, arr2)
    self.assertEqual(out.sharding,
                     NamedSharding(mesh, P('x', None, None, None, None)))

    out = _f(arr1, jnp.ones((10, 5, 3)))
    self.assertEqual(out.sharding,
                     NamedSharding(mesh, P('x', None, None, None, None)))

  def test_aval_spec_explicit_auto_complete(self):
    abstract_mesh = mesh_lib.AbstractMesh(
        (2,), 'x', axis_types=AxisType.Explicit)
    s = NamedSharding(abstract_mesh, P('x'))
    out = core.ShapedArray((8, 2), jnp.int32, sharding=s)
    self.assertEqual(out.sharding.spec, P('x', None))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(mesh_lib.AxisType.Auto,) * 2)
  def test_full_user_mode(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    # No axes specified means full visible mode.
    @partial(explicit_axes, in_sharding=P('x', 'y'))
    def h(y):
      self.assertEqual(y.aval.sharding.spec, P('x', 'y'))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P('x', 'y'))
      a = jnp.einsum('ab,bc->ac', z, z.T, out_sharding=P('x', None))
      self.assertEqual(a.aval.sharding.spec, P('x', None))
      return a

    @jax.jit
    def f(x):
      y = x * 2
      a = h(y)
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

    jaxpr = f.trace(arr).jaxpr
    core.jaxpr_as_fun(jaxpr)(arr)  # doesn't crash

  @jtu.with_explicit_mesh((4,), ('data',))
  def test_intermediate_einsum(self, mesh):
    shape1 = (8, 32, 1, 16)
    shape2 = (8, 32, 1, 8)
    np_inp1 = np.ones(math.prod(shape1)).reshape(shape1)
    np_inp2 = np.ones(math.prod(shape2)).reshape(shape2)

    s = NamedSharding(mesh, P('data'))
    arr1 = jax.device_put(np_inp1, s)
    arr2 = jax.device_put(np_inp1, s)
    arr3 = jax.device_put(np_inp2, s)

    @jax.jit
    def f(x, y, z):
      out = jnp.einsum('bthD, bthi, bthj->ijD', x, y, z,
                       out_sharding=P('data', None, None))
      self.assertEqual(out.shape, (16, 8, 16))
      self.assertEqual(out.aval.sharding.spec, P('data', None, None))
      return out

    out = f(arr1, arr2, arr3)
    self.assertEqual(out.shape, (16, 8, 16))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('data', None, None)))

  @jtu.with_explicit_mesh((4,), ('data',))
  def test_intermediate_einsum_auto_complete_spec(self, mesh):
    s = NamedSharding(mesh, P('data'))

    shape1 = (8, 32, 2*16)
    shape2 = (8, 32, 2, 8)
    shape3 = (8, 32, 2, 8)
    np_inp1 = np.ones(math.prod(shape1)).reshape(shape1)
    np_inp2 = np.ones(math.prod(shape2)).reshape(shape2)
    np_inp3 = np.ones(math.prod(shape3)).reshape(shape3)

    arr1 = jax.device_put(np_inp1, s)
    arr2 = jax.device_put(np_inp2, s)
    arr3 = jax.device_put(np_inp3, s)

    @jax.jit
    def f(x, y, z):
      x = jnp.reshape(x,  (8, 32, 2, 16))
      out = jnp.einsum('bthD, bthi, bthj->ijD', x, y, z,
                       out_sharding=P('data'))
      self.assertEqual(out.shape, (8, 8, 16))
      self.assertEqual(out.aval.sharding.spec, P('data', None, None))
      return out

    out = f(arr1, arr2, arr3)
    self.assertEqual(out.shape, (8, 8, 16))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('data', None, None)))

  def test_where_with_prng_sharded_inp(self):
    mesh = jax.sharding.Mesh(jax.devices(), axis_names=['batch'])
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('batch')
    )
    condition = jax.device_put(jnp.zeros([32, 1], dtype=jnp.bool), sharding)
    x = jax.device_put(
        jnp.broadcast_to(jax.random.key(0), [32, 32]),
        sharding,
    )

    def f(condition, x, y):
      condition = jnp.asarray(condition)
      self.assertTrue(x.aval.sharding.mesh._are_all_axes_auto)
      self.assertTrue(y.aval.sharding.mesh._are_all_axes_auto)
      x1 = jnp.asarray(x)
      self.assertEqual(x1.aval.sharding, x.aval.sharding)
      y1 = jnp.asarray(y)
      self.assertEqual(y1.aval.sharding, y.aval.sharding)
      return jnp.where(condition, x1, y1)

    f = jax.jit(f, in_shardings=(sharding, sharding, sharding))
    f(condition, x, x).block_until_ready()

  @jtu.with_explicit_mesh((4,), ('data',))
  def test_intermediate_einsum_conflict_error(self, mesh):
    shape1 = (8, 32, 1, 16)
    shape2 = (8, 32, 1, 8)
    np_inp1 = np.ones(math.prod(shape1)).reshape(shape1)
    np_inp2 = np.ones(math.prod(shape2)).reshape(shape2)

    arr1 = jax.device_put(
        np_inp1, NamedSharding(mesh, P(None, None, None, 'data')))
    arr2 = jax.device_put(np_inp1, NamedSharding(mesh, P('data')))
    arr3 = jax.device_put(np_inp2, NamedSharding(mesh, P('data')))

    @jax.jit
    def f(x, y, z):
      return jnp.einsum('bthD, bthi, bthj->ijD', x, y, z,
                        out_sharding=P('data', None, None))

    # Errors out on the intermediate einsum: `bthj,bthD->bthjD`
    # because of a conflict
    with self.assertRaisesRegex(
        core.ShardingTypeError,
        'dot_general operation.*produces an illegally sharded result'):
      f(arr1, arr2, arr3)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(mesh_lib.AxisType.Explicit,
                                  mesh_lib.AxisType.Auto))
  def test_mix_to_full_user_mode(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @partial(explicit_axes, axes='y', in_sharding=P('x', 'y'))
    def h(y):
      self.assertEqual(y.aval.sharding.spec, P('x', 'y'))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P('x', 'y'))
      a = jnp.einsum('ab,bc->ac', z, z.T, out_sharding=P('x', 'y'))
      self.assertEqual(a.aval.sharding.spec, P('x', 'y'))
      return a

    @jax.jit
    def f(x):
      y = x * 2
      a = h(y)
      self.assertEqual(a.aval.sharding.spec, P('x', None))
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(mesh_lib.AxisType.Auto,) * 2)
  def test_full_auto_to_partial_user(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @partial(explicit_axes, axes='y', in_sharding=P(None, 'y'))
    def h(y):
      self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P(None, 'y'))
      a = jnp.einsum('ab,bc->ac', z, z.T, out_sharding=P(None, 'y'))
      self.assertEqual(a.aval.sharding.spec, P(None, 'y'))
      return a

    @jax.jit
    def f(x):
      y = x * 2
      a = h(y)
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      return a

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_auto_gather_out_sharding(self, mesh):
    embed = jax.device_put(jnp.arange(128 * 8.).reshape(64, 16),
                           jax.NamedSharding(mesh, P(None, 'x')))
    tok = jax.device_put(jnp.arange(8 * 4).reshape(8, 4),
                         jax.NamedSharding(mesh, P('x', None)))

    @jax.jit
    def f(embed_vd, token_bt):
      out = embed_vd.at[token_bt].get(out_sharding=P('x', None, None))
      self.assertEqual(out.shape, (8, 4, 16))
      self.assertEqual(out.aval.sharding.spec, P('x', None, None))

      out2 = embed_vd.at[token_bt, :].get(out_sharding=P('x', None, None))
      self.assertEqual(out2.shape, (8, 4, 16))
      self.assertEqual(out2.aval.sharding.spec, P('x', None, None))

      out3 = embed_vd.at[token_bt, ...].get(out_sharding=P('x', None, None))
      self.assertEqual(out3.shape, (8, 4, 16))
      self.assertEqual(out3.aval.sharding.spec, P('x', None, None))
      return out

    out = f(embed, tok)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None, None)))

    lowered_text = f.lower(embed, tok).as_text()
    self.check_wsc_in_lowered(lowered_text)

    def g(x, y):
      out = f(x, y)
      return jnp.sum(out)

    out = jax.jit(jax.grad(g))(embed, tok)
    self.assertEqual(out.sharding, embed.sharding)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshard_error(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      y = reshard(x, P('x', None))
      self.assertEqual(y.aval.sharding.spec, P('x', None))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    f = jax.jit(f)

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    lowered_text = f.lower(arr).as_text()
    self.check_wsc_in_lowered(lowered_text)

    def g(x):
      y = f(x)
      return jnp.sum(y)

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

    def f_vmap(x):
      self.assertEqual(x.aval.sharding.spec, P('y'))
      y = reshard(x, P(None))
      self.assertEqual(y.aval.sharding.spec, P(None))
      return y

    out = jax.vmap(f_vmap)(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    out = jax.jit(jax.vmap(jax.jit(f_vmap)))(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    @jax.jit
    def h(x):
      with use_auto_axes('x'):
        return reshard(x, P('y', None))

    with self.assertRaisesRegex(
        ValueError, 'Mesh of the input.*does not equal.*target sharding'):
      h(arr)

  def test_auto_axes_top_level(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    np_inp = np.arange(16.).reshape(8, 2)
    arr1 = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', 'x')))

    @partial(auto_axes, out_sharding=P('x', None))
    def auto_matmul(arr1, arr2):
      return arr1 @ arr2

    @jax.jit
    def f(arr1, arr2):
      y = jnp.sin(arr1)
      z = auto_matmul(y, arr2)
      self.assertEqual(z.aval.sharding.spec, P('x', None))
      return z + 1

    with jax.sharding.use_mesh(mesh):
      out = f(arr1, arr2)
      self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  def test_explicit_axes_top_level(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'),
                           axis_types=(AxisType.Auto,) * 2)
    np_inp = np.arange(16.).reshape(8, 2)
    arr1 = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', 'x')))

    @partial(explicit_axes, in_sharding=(P('x', None), P('x', None)))
    def jax_matmul(arr1, arr2):
      out = arr1 @ arr2
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    @jax.jit
    def f(arr1, arr2):
      y = jnp.sin(arr1)
      z = jax_matmul(y, arr2)
      return z + 1

    with jax.sharding.use_mesh(mesh):
      out = f(arr1, arr2)
      self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

  def test_reshard_eager_mode(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    np_inp = np.arange(16.).reshape(8, 2)
    arr1 = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', 'x')))

    def matmul_reshard(arr1, arr2):
      arr2 = reshard(arr2, P('y', None))
      self.assertEqual(arr2.aval.sharding.spec, P('y', None))
      out = jnp.einsum('xy,yz->xz', arr1, arr2, out_sharding=P('x', 'y'))
      self.assertEqual(out.aval.sharding.spec, P('x', 'y'))
      return out

    with jax.sharding.use_mesh(mesh):
      matmul_reshard(arr1, arr2)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_full_auto_outside_jit(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x * 2
      self.assertEqual(y.aval.sharding.spec, P(None, None))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P(None, None))
      a = z @ z.T
      self.assertEqual(a.aval.sharding.spec, P(None, None))
      return a

    hf = auto_axes(f, axes=('x', 'y'), out_sharding=P('x', 'y'))
    out = hf(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(AxisType.Auto,) * 2)
  def test_full_visible_outside_jit(self, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = x * 2
      self.assertEqual(y.aval.sharding.spec, P('x', 'y'))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P('x', 'y'))
      return z

    hf = explicit_axes(f, axes=('x', 'y'), in_sharding=P('x', 'y'))
    out = hf(arr)  # doesn't crash
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  def test_compilation_cache_miss_when_devices_change(self):
    mesh1 = jtu.create_mesh((2, 2), ('x', 'y'))
    devs = jax.devices()[:4]
    mesh2 = Mesh(np.asarray(devs[::-1]).reshape(2, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)

    with jax.sharding.use_mesh(mesh1):
      arr1 = jax.device_put(np_inp, NamedSharding(mesh1, P('x', 'y')))
    with jax.sharding.use_mesh(mesh2):
      arr2 = jax.device_put(np_inp, NamedSharding(mesh2, P('x', 'y')))

    @jax.jit
    def f(x):
      return x

    with (jtu.count_jit_tracing_cache_miss() as tracing_count,
          jtu.count_jit_and_pmap_lowerings() as lowering_count,
          jtu.count_jit_compilation_cache_miss() as compilation_count,
          jtu.count_pjit_cpp_cache_miss() as cpp_cache_miss_count):
      with jax.sharding.use_mesh(mesh1):
        out1 = f(arr1)
      with jax.sharding.use_mesh(mesh2):
        out2 = f(arr2)

    self.assertEqual(tracing_count(), 1)
    self.assertEqual(lowering_count(), 1)
    self.assertEqual(compilation_count(), 2)
    self.assertEqual(cpp_cache_miss_count(), 2)

    self.assertTupleEqual(out1.sharding._device_assignment,
                          tuple(mesh1.devices.flat))
    self.assertTupleEqual(out2.sharding._device_assignment,
                          tuple(mesh2.devices.flat))

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'))
  def test_svd(self, mesh):
    np_inp = np.zeros([128, 128])
    arr = jax.device_put(np_inp, NamedSharding(mesh, P(None, None)))

    @jax.jit
    def f(x):
      return jnp.linalg.norm(x, 2)

    f(arr)  # doesn't crash

  def test_shaped_array_input_to_jit_no_sharding(self):
    # export_test.py has similar test but it's more complicated. This is a
    # simplified version of a part of that test.
    aval = core.ShapedArray((8,), jnp.int32)
    aval2 = core.ShapedArray((8,), jnp.int32)

    @jax.jit
    def f(x, y):
      return x * y

    lowered_text = f.lower(aval, aval2).as_text()
    self.assertNotIn("mhlo.sharding", lowered_text)

  @parameterized.parameters(True, False)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_mul_vmap(self, use_jit, mesh):
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      self.assertEqual(x.aval.sharding.spec, P(s.spec[1]))
      x = x * 2
      self.assertEqual(x.aval.sharding.spec, P(s.spec[1]))
      x = x * x
      self.assertEqual(x.aval.sharding.spec, P(s.spec[1]))
      return x

    if use_jit:
      f = jax.jit(f)

    f = jax.vmap(f)

    out = f(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp * 2) * (np_inp * 2))

    out = jax.jit(f)(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, (np_inp * 2) * (np_inp * 2))

    def g(x):
      return jnp.sum(f(x))

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

  @parameterized.parameters(True, False)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_dot_general_vmap(self, use_jit, mesh):
    np_inp1 = np.arange(16.).reshape(4, 2, 2)
    np_inp2 = np.arange(16.).reshape(2, 4, 2)
    arr1 = jax.device_put(np_inp1, NamedSharding(mesh, P('x', None, 'y')))
    arr2 = jax.device_put(np_inp2, NamedSharding(mesh, P(None, 'x', 'y')))

    def f(x, y):
      return jnp.einsum('xy,yz->xz', x, y, out_sharding=P(None, 'y'))

    if use_jit:
      f = jax.jit(f)

    f = jax.vmap(f, in_axes=(0, 1), out_axes=2)

    out = f(arr1, arr2)
    self.assertEqual(out.shape, (2, 2, 4))
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_reshape_vmap(self, mesh):
    np_inp = np.arange(16).reshape(2, 8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P(None, 'x')))

    def f(x):
      y = lax.reshape(x, (1, 2), out_sharding=P(None, 'y'))
      y = y * 2
      self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
      return y

    out = jax.jit(jax.vmap(f, in_axes=1))(arr)
    self.assertEqual(out.shape, (8, 1, 2))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None, 'y')))

  @parameterized.parameters(True, False)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_shit_vmap_error_check(self, use_jit, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', None)))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P(None, 'y')))

    def f(x, y):
      return x @ y

    if use_jit:
      f = jax.jit(f)

    with self.assertRaisesRegex(
        ValueError,
        "Mapped away dimension of inputs passed to vmap should be sharded "
        "the same"):
      jax.vmap(f, in_axes=(0, 1))(arr, arr2)

    with self.assertRaisesRegex(
        ValueError,
        'Mapped away dimension of inputs passed to vmap should be sharded the'
        ' same'):
      jax.jit(jax.vmap(f, in_axes=(0, 1)))(arr, arr2)

    with self.assertRaisesRegex(
        ValueError,
        "Only one of spmd_axis_name or arrays sharded on.*spmd_axis_name"):
      jax.vmap(f, spmd_axis_name='y')(arr, arr)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_unmapped_last_vmap(self, mesh):
    np_inp = np.arange(8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x',)))

    @partial(jax.vmap, out_axes=-1)
    def f(x):
      return jnp.zeros((4,))

    out = f(arr)
    self.assertEqual(out.shape, (4, 8))
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'x')))

  @jtu.with_explicit_mesh((2,), ('x',), axis_types=AxisType.Auto)
  def test_shmap_close_over(self, mesh):
    const = jnp.arange(8)
    def f():
      return const * 2

    shmap_f = shard_map(f, mesh=mesh, in_specs=(), out_specs=P('x'))
    shmap_f()  # doesn't crash
    jax.jit(shmap_f)()  # doesn't crash

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'),
                      axis_types=(AxisType.Auto,) * 2)
  def test_shmap_close_over_partial_auto(self, mesh):
    const = jnp.arange(8)
    def f():
      return const * 2

    shmap_f = shard_map(f, mesh=mesh, in_specs=(), out_specs=P('x'),
                        axis_names={'x'})
    f = jax.jit(shmap_f)
    out = f()
    self.assertArraysEqual(out, jnp.concatenate([const * 2, const * 2]))

    jaxpr = f.trace().jaxpr
    self.assertIn('mesh_cast', str(jaxpr))

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'))
  def test_wsc_error(self, mesh):
    s = NamedSharding(mesh, P('x'))
    with self.assertRaisesRegex(
        ValueError,
        "The spec of NamedSharding passed to with_sharding_constraint"):
      jax.lax.with_sharding_constraint(np.arange(8), s)

    s = NamedSharding(mesh, P(('x', 'y'), None))
    with self.assertRaisesRegex(
        ValueError,
        "The spec of NamedSharding passed to with_sharding_constraint"):
      jax.lax.with_sharding_constraint(np.arange(8).reshape(4, 2), s)

    with self.assertRaisesRegex(
        ValueError,
        'with_sharding_constraint cannot be used when all axes of the mesh are'
        ' of type `Explicit`'):
      jax.lax.with_sharding_constraint(np.arange(8), NamedSharding(mesh, P()))

    s = NamedSharding(Mesh(mesh.devices, mesh.axis_names,
                           axis_types=(AxisType.Explicit, AxisType.Auto)),
                      P('x', P.UNCONSTRAINED))
    with self.assertRaisesRegex(
        ValueError,
        "The spec of NamedSharding passed to with_sharding_constraint"):
      jax.lax.with_sharding_constraint(np.arange(8).reshape(4, 2), s)

    with self.assertRaisesRegex(
        ValueError,
        'PartitionSpec.*cannot contain `P.UNCONSTRAINED` when no mesh'
        ' axis_types are `Auto`'):
      NamedSharding(mesh, P(P.UNCONSTRAINED))

  def test_pspec_einsum_no_context_mesh(self):
    mesh = jtu.create_mesh((1, 1), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', None)))

    @jax.jit
    def f(x, y):
      return jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', 'y'))

    with self.assertRaisesRegex(
        ValueError,
        "Using PartitionSpec when.*not under a mesh context.*is not allowed"):
      f(arr, arr2)

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'),
                      axis_types=(AxisType.Auto,) * 2)
  def test_error_on_canonicalize_under_auto_mode(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P('x', 'y')))
    arr2 = jax.device_put(np_inp.T, NamedSharding(mesh, P('y', None)))

    @jax.jit
    def f(x, y):
      return jnp.einsum('xy,yz->xz', x, y,
                        out_sharding=NamedSharding(mesh, P('x', 'y')))

    with self.assertRaisesRegex(
        ValueError,
        "PartitionSpec passed to einsum cannot contain axis names.*Auto.*Manual"):
      f(arr, arr2)

  def test_broadcasted_iota_mix_axes(self):
    mesh = jtu.create_mesh(
        (2, 2, 2), ('x', 'y', 'z'),
        axis_types=(AxisType.Auto, AxisType.Explicit, AxisType.Explicit))
    yz_sharding = NamedSharding(mesh, P(('y', 'z')))

    @jax.jit
    def iota():
      out = jax.lax.broadcasted_iota(
          dtype=jnp.int32,
          shape=(16, 24),
          dimension=1,
          out_sharding=yz_sharding)
      self.assertEqual(out.aval.sharding.spec, P(('y', 'z'), None))
      return out

    with jax.sharding.use_mesh(mesh):
      out = iota()
      self.assertEqual(out.sharding, yz_sharding)

  @jtu.with_explicit_mesh((2, 2, 2), ('x', 'y', 'z'))
  def test_broadcast_to(self, mesh):
    x = np.arange(24).reshape((1, 24))
    x = jax.device_put(x, P(None, ('y', 'z')))

    @jax.jit
    def f(x):
      out = jnp.broadcast_to(x, (8, 24), out_sharding=P('x', ('y', 'z')))
      self.assertEqual(out.aval.sharding.spec, P('x', ('y', 'z')))
      return out

    out = f(x)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', ('y', 'z'))))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_cumsum(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P()))

    @jax.jit
    def f(x):
      return jnp.cumsum(x)

    out = f(arr)
    self.assertArraysEqual(out, np.cumsum(np_inp))
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None)))

    @jax.jit
    def f(x):
      x = jnp.expand_dims(x, 1)
      self.assertEqual(x.aval.sharding.spec, P('x', None))
      out = jnp.cumsum(x, axis=1)
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    arr2 = jax.device_put(np.arange(8), P('x'))
    out = f(arr2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  def test_device_put_under_use_mesh(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    x = jnp.zeros((4, 4), dtype=jnp.int32)
    x_np = np.zeros((4, 4), dtype=np.int32)
    s = NamedSharding(mesh, P('x', 'y'))
    with jax.sharding.use_mesh(mesh):
      y = jax.device_put(x, s)
      self.assertArraysEqual(y, x)
      self.assertEqual(y.sharding, s)

      y2 = jax.device_put(x_np, s)
      self.assertArraysEqual(y2, x_np)
      self.assertEqual(y2.sharding, s)

      s2 = NamedSharding(mesh, P('x'))
      z = jax.device_put(y, s2)
      self.assertArraysEqual(z, x)
      self.assertEqual(z.sharding, s2)

  @parameterized.parameters(True, False)
  def test_wsc_pspec_use_mesh(self, sharded_inp):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    np_inp = np.zeros((4, 4), dtype=np.int32)
    if sharded_inp:
      arr = jax.device_put(np_inp, NamedSharding(mesh, P()))
    else:
      arr = np_inp

    with jax.sharding.use_mesh(mesh):
      out = with_sharding_constraint(arr, P('x', 'y'))
      self.assertArraysEqual(out, np_inp)
      self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

    with jax.sharding.use_mesh(mesh):
      f = jax.jit(lambda x: with_sharding_constraint(x, P('x', 'y')))
      jaxpr = f.trace(arr).jaxpr
      self.assertIsInstance(jaxpr.eqns[0].params['sharding'].mesh,
                            mesh_lib.AbstractMesh)
      out = f(arr)
      self.assertArraysEqual(out, np_inp)
      self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

      arr2 = jax.device_put(np_inp, NamedSharding(mesh, P('x')))
      out2 = f(arr2)
      self.assertArraysEqual(out2, np_inp)
      self.assertEqual(out2.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2, 1), ('x', 'y'),
                      axis_types=(AxisType.Auto,) * 2)
  def test_axes_api_error_manual_to_auto_explicit(self, mesh):
    def g(x):
      return auto_axes(lambda a: a * 2, axes=('x', 'y'),
                       out_sharding=P('x', 'y'))(x)

    with self.assertRaisesRegex(
        NotImplementedError, "Going from `Manual`.*to.*`Auto`.*`Explicit`"):
      jax.jit(shard_map(g, mesh=mesh, in_specs=P('x', 'y'), out_specs=P('x', 'y'))
              )(np.arange(16).reshape(8, 2))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_auto_axes_numpy_array(self, mesh):
    @jax.jit
    def f(x):
      self.assertTrue(x.aval.sharding.mesh._are_all_axes_auto)
      return x * 2

    out = auto_axes(f, out_sharding=P('x'))(np.arange(8))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(out, np.arange(8) * 2)

  @jtu.sample_product(
    from_dtype=(['int4', 'uint4'] + jtu.dtypes.all_floating +
                jtu.dtypes.all_integer + jtu.dtypes.all_unsigned),
    to_dtype=(['int4', 'uint4'] + jtu.dtypes.all_floating +
              jtu.dtypes.all_integer + jtu.dtypes.all_unsigned),
    shape_and_spec=[((), P()), ((2,), P('x')), ((2, 4), P('x', 'y'))],
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_bitcast_convert_type(self, from_dtype, to_dtype, shape_and_spec,
                                mesh):
    shape, spec = shape_and_spec
    rng = jtu.rand_default(self.rng())
    nbits_in = dtypes.bit_width(from_dtype)
    nbits_out = dtypes.bit_width(to_dtype)
    if nbits_in < nbits_out:
      shape = (*shape, nbits_out // nbits_in)
      spec = P(*(*spec, None))
    args_maker = lambda: [jax.device_put(rng(shape, from_dtype),
                                         NamedSharding(mesh, spec))]

    if nbits_in == nbits_out:
      expected_shape = shape
      expected_spec = spec
    elif nbits_in < nbits_out:
      expected_shape = shape[:-1]
      expected_spec = P(*spec[:-1])
    else:
      expected_shape = (*shape, nbits_in // nbits_out)
      expected_spec = P(*spec, None)

    @jax.jit
    def f(x):
      out = lax.bitcast_convert_type(x, to_dtype)
      self.assertEqual(out.aval.shape, expected_shape)
      self.assertEqual(out.aval.sharding.spec, expected_spec)
      return out

    self._CompileAndCheck(f, args_maker)

    # Test the shape and dtype of the output. We avoid testing the values here
    # because the bitwise representation may vary from platform to platform.
    out = f(*args_maker())
    self.assertEqual(out.dtype, to_dtype)
    self.assertEqual(out.shape, expected_shape)
    self.assertEqual(out.sharding, NamedSharding(mesh, expected_spec))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_dynamic_slice(self, mesh):
    np_inp = np.arange(16., dtype=np.float32)
    s = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np_inp, s)

    @jax.jit
    def f(x):
      y = lax.dynamic_slice_in_dim(x, jnp.array(1, dtype=np.int32), 2)
      self.assertEqual(y.aval.sharding.spec, P('x'))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, s)

    def g(x):
      return jnp.sum(f(x))

    out = jax.jit(jax.grad(g))(arr)
    self.assertEqual(out.sharding, arr.sharding)

    out = jax.grad(g)(arr)
    self.assertEqual(out.sharding, arr.sharding)

  def test_auto_axes_computation_follows_data(self):
    mesh = jtu.create_mesh((2,), ('x',), axis_types=(AxisType.Explicit,))
    s = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np.arange(8), s)

    @jax.jit
    def f(x):
      return x * 2

    out = auto_axes(f, out_sharding=s)(arr)
    self.assertEqual(out.sharding, s)
    self.assertArraysEqual(out, arr * 2)

  def test_divisbility_aval_error(self):
    abstract_mesh = mesh_lib.AbstractMesh(
        (2,), ('x',), axis_types=AxisType.Explicit)
    s = NamedSharding(abstract_mesh, P('x'))
    with self.assertRaisesRegex(
        ValueError, 'does not evenly divide the dimension size'):
      core.ShapedArray((5, 2), jnp.int32, sharding=s)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_scan_unroll(self, mesh):
    np_inp = np.arange(64, dtype=jnp.float32).reshape(8, 8)
    arr = jax.device_put(np_inp, NamedSharding(mesh, P(None, 'y')))
    carry = jnp.ones((8,), dtype=jnp.float32)

    @jax.jit
    def f(carry, xs):
      def body(carry, x):
        return carry + x, x
      return jax.lax.scan(body, carry, xs, unroll=2)

    f(carry, arr)  # doesn't crash

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_reshard_with_np_array(self, mesh):
    out = reshard(np.arange(8), P('x'))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

    @jax.jit
    def f(x):
      return reshard(x, P('x'))
    out = f(np.arange(8))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

  @jtu.thread_unsafe_test()
  def test_set_mesh(self):
    mesh = jtu.create_mesh((2,), ('x',), axis_types=(AxisType.Explicit,))
    try:
      prev_mesh = jax.sharding.set_mesh(mesh)
      out = reshard(np.arange(8), P('x'))
      self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    finally:
      self.assertIsNone(prev_mesh)
      jax.sharding.set_mesh(prev_mesh)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_auto_axes_late_bind(self, mesh):
    @auto_axes
    def f(x):
      return x * 2

    out = f(np.arange(8), out_sharding=P('x'))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(out, np.arange(8) * 2)

  @jtu.with_explicit_mesh((2,), ('x',), axis_types=AxisType.Auto)
  def test_explicit_axes_late_bind(self, mesh):
    @explicit_axes
    def f(x):
      return x * 2

    out = f(np.arange(8), in_sharding=P('x'))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(out, np.arange(8) * 2)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_rng_bit_generator(self, mesh):
    def f(key):
      out = lax.rng_bit_generator(key, shape=(4, 8), out_sharding=P('x'))
      self.assertEqual(out[0].aval.sharding.spec, P(None))
      self.assertEqual(out[1].aval.sharding.spec, P('x', None))
      return out

    key = np.array((1, 2, 3, 4)).astype(np.uint32)
    out1 = f(key)
    jit_f = jax.jit(f)
    out2 = jit_f(key)
    self.assertEqual(out1[0].shape, (4,))
    self.assertEqual(out1[1].shape, (4, 8))
    self.assertEqual(out2[0].sharding, NamedSharding(mesh, P()))
    self.assertEqual(out2[1].sharding, NamedSharding(mesh, P('x', None)))
    self.assertEqual(out1[0].sharding, out2[0].sharding)
    self.assertEqual(out1[1].sharding, out2[1].sharding)
    self.assertArraysEqual(out1[0], out2[0])
    self.assertArraysEqual(out1[1], out2[1])

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_fold_in(self, mesh):
    key = jax.random.key(72)
    key = jax.device_put(key, NamedSharding(mesh, P()))

    @jax.jit
    def f(key):
      f1 = jax.random.fold_in(key, 1)
      self.assertEqual(jax.random.key_data(f1).aval.sharding.spec, P(None))
      return f1

    out = f(key)
    self.assertEqual(out.sharding, NamedSharding(mesh, P()))

  @parameterized.named_parameters(
      ("bits", partial(jax.random.bits, shape=(8, 12)), P('x', 'y')),
      ("uniform", partial(jax.random.uniform, shape=(8, 12)), P('x', 'y')),
      ("normal", partial(jax.random.normal, shape=(8, 12)), P('x', 'y')),
      ("randint", partial(jax.random.randint, shape=(8, 12), minval=0, maxval=10),
       P('x', 'y')),
      ("permutation_1d", partial(jax.random.permutation, x=8), P('x')),
      ("permutation_2d", partial(jax.random.permutation,
                                 x=np.arange(8 * 12).reshape(8, 12)),
       P('x', 'y')),
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_random_functions(self, fun, out_spec, mesh):
    @jax.jit
    def f(key):
      out = fun(key, out_sharding=out_spec)
      self.assertEqual(out.aval.sharding.spec, out_spec)
      return out

    key = jax.random.key(1)
    out = f(key)
    self.assertEqual(out.sharding, NamedSharding(mesh, out_spec))

    lowered_text = f.lower(key).as_text()
    if config.use_shardy_partitioner.value:
      self.assertIn('sdy.sharding_constraint', lowered_text)
      if out_spec == P('x', 'y'):
        self.assertIn('<@mesh, [{"x"}, {"y"}]>', lowered_text)
      else:
        assert out_spec == P('x')
        self.assertIn('<@mesh, [{"x"}]>', lowered_text)
    else:
      if out_spec == P('x', 'y'):
        self.assertIn('mhlo.sharding = "{devices=[2,2]<=[4]}"}', lowered_text)
      else:
        assert out_spec == P('x')
        self.assertIn(
            'mhlo.sharding = "{devices=[2,2]<=[4] last_tile_dim_replicate}"}',
            lowered_text)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_random_truncated_normal(self, mesh):
    @jax.jit
    def f(key, lower):
      out = jax.random.truncated_normal(key, lower, 2., shape=(8, 12),
                                        out_sharding=P('x', 'y'))
      self.assertEqual(out.aval.sharding.spec, P('x', 'y'))
      return out

    key = jax.random.key(1)
    out = f(key, -1.)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

    lowered_text = f.lower(key, -1.).as_text()
    if config.use_shardy_partitioner.value:
      self.assertIn('sdy.sharding_constraint', lowered_text)
      self.assertIn('<@mesh, [{"x"}, {"y"}]>', lowered_text)
    else:
      self.assertIn('mhlo.sharding = "{devices=[2,2]<=[4]}"}', lowered_text)

  def test_random_normal_wo_mesh_context_error(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    s = NamedSharding(mesh, P('x', 'y'))

    @jax.jit
    def f(key):
      out = jax.random.normal(key, shape=(8, 12), out_sharding=s)
      self.assertEqual(out.aval.sharding.spec, P('x', 'y'))
      self.assertEqual(out.aval.sharding.mesh, mesh.abstract_mesh)
      return out

    key = jax.random.key(1)
    with self.assertRaisesRegex(
        ValueError,
        'Length of device assignment.*is not equal to the size of the mesh'):
      f(key)

  def test_random_normal_wo_mesh_context(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'),
                           axis_types=(AxisType.Explicit,) * 2)
    s = NamedSharding(mesh, P('x', 'y'))

    @jax.jit
    def f(arr, key):
      out = jax.random.normal(key, shape=(8, 12), out_sharding=s)
      self.assertEqual(out.aval.sharding.spec, P('x', 'y'))
      return arr + out

    key = jax.random.key(1)
    out = f(jax.device_put(np.arange(8 * 12.).reshape(8, 12), s), key)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  def test_auto_axes_no_context_mesh(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'), axis_types=(AxisType.Explicit,) * 2)
    np_inp = np.arange(16.).reshape(8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @partial(auto_axes, axes='x',
             out_sharding=NamedSharding(mesh, P('x', 'y')))
    def h(y):
      self.assertEqual(y.aval.sharding.spec, P(None, 'y'))
      z = jnp.sin(y)
      self.assertEqual(z.aval.sharding.spec, P(None, 'y'))
      return z

    out = jax.jit(h)(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

    out = h(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  def test_scan_with_random_key_inside_jit(self):
    mesh = jtu.create_mesh((2,), ('x',))
    sharding = NamedSharding(mesh, P(None, 'x'))

    @jax.jit
    def scan(xs):
      def step(carry, x):
        next_carry = jax.vmap(jax.random.fold_in)(carry, x)
        next_carry = jnp.where(x % 2 == 0, carry, next_carry)
        return next_carry, None
      rng = jnp.broadcast_to(jax.random.key(0), xs.shape[1:])
      rng, _ = jax.lax.scan(step, rng, xs)
      return rng

    xs = jnp.arange(8).reshape(2, 4)
    scan(xs)

    xs = jax.device_put(xs, sharding)
    scan(xs)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_select_batch(self, mesh):
    y_sharding = NamedSharding(mesh, P('y', None))
    xy_sharding = NamedSharding(mesh, P('x', 'y', None))
    batch_a = jax.device_put(jnp.ones((4, 2, 3), dtype=jnp.float32), xy_sharding)
    batch_b = jax.device_put(jnp.ones((4, 2, 2), dtype=jnp.int32), xy_sharding)

    out_s = NamedSharding(mesh, P('x', 'y', None, None))

    def select(a, b):
      c = a.at[b].get(out_sharding=y_sharding)
      return c

    @jax.jit
    def vmap_select(batch_a, batch_b):
      out = jax.vmap(select)(batch_a, batch_b)
      self.assertEqual(out.aval.sharding.spec, out_s.spec)
      return out

    out = vmap_select(batch_a, batch_b)
    self.assertEqual(out.sharding, out_s)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_where_vmap(self, mesh):
    xy_sharding = NamedSharding(mesh, P('x', 'y', None))
    batch_a = jax.device_put(jnp.ones((4, 2, 3), dtype=jnp.float32), xy_sharding)
    batch_b = jax.device_put(jnp.ones((4, 2, 3), dtype=jnp.bool), xy_sharding)

    def where(a, b):
      out = jnp.where(b, a, 0)
      return out

    @jax.jit
    def vmap_where(batch_a, batch_b):
      out = jax.vmap(where)(batch_a, batch_b)
      self.assertEqual(out.aval.sharding.spec, xy_sharding.spec)
      return out

    out = vmap_where(batch_a, batch_b)
    self.assertEqual(out.sharding, xy_sharding)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_convert_element_type_vmap(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, P('x', 'y'))
    am = mesh.abstract_mesh

    @jax.jit
    @jax.vmap
    def f(x):
      y = lax_internal._convert_element_type(
          x, jnp.bfloat16, sharding=NamedSharding(am, P('y')))
      self.assertEqual(y.aval.sharding.spec, P('y'))
      return y

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_jnp_repeat(self, mesh):
    out = jnp.repeat(np.eye(3), np.array((2,2,2,)) - 1, axis=0)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, None)))

    a = jnp.eye(3)
    out = jnp.repeat(a, np.array((2,2,2,)) - 1, axis=0)
    self.assertEqual(out.sharding, a.sharding)

    a = jax.device_put(jnp.eye(4), P('x'))
    out = jnp.repeat(a, np.array((2,2,2,2)) - 1, axis=0, out_sharding=P('x'))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

    a = jax.device_put(jnp.eye(16).reshape(16, 16), P('x'))
    @jax.jit
    def f(x):
      return jnp.repeat(x, 3, axis=-1)
    f(a)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_scatter_gather(self, mesh):
    x = np.random.uniform(size=(mesh.size * 2, 3))
    i = np.random.randint(0, x.shape[1], len(x))
    j = np.random.randint(0, x.shape[1], len(x))
    x = jax.device_put(x, P("x"))
    i = jax.device_put(i, P("x"))
    j = jax.device_put(j, P("x"))

    @jax.jit
    def f1(x, i, j):
      x_a_j = x.at[:, j].get(out_sharding=jax.typeof(i).sharding)
      return x.at[:, i].set(x_a_j)
    f1(x,i,j)  # doesn't crash

    @jax.jit
    @jax.vmap
    def f2(x, i, j):
      x_j = x.at[j].get(out_sharding=jax.typeof(x).sharding)
      return x.at[i].set(x_j)
    f2(x,i,j)  # doesn't crash

  @jtu.with_explicit_mesh((4, 2), ('x', 'y'))
  def test_conv_general_dilated(self, mesh):
    arr = jax.device_put(np.zeros((16, 128, 8)), P('x', 'y'))

    @jax.jit
    def f(x):
      # Conv1D across sharded y-axis:
      out = jax.lax.conv_general_dilated(
          x, np.zeros((5, 8, 10)),
          window_strides=(1,), padding='SAME', feature_group_count=1,
          lhs_dilation=(1,), rhs_dilation=(1,),
          dimension_numbers=('NWC', 'WIO', 'NWC'))
      self.assertEqual(out.aval.sharding.spec, P('x', 'y', None))
      # Max pooling along sharded y-axis.
      out2 = jax.lax.reduce_window(
          out, -np.inf, jax.lax.max, (1,2,1), (1,2,1), 'SAME')
      self.assertEqual(out2.aval.sharding.spec, P('x', 'y', None))
      return out2

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', 'y', None)))
    self.check_wsc_in_lowered(f.lower(arr).as_text())

    jax.jit(jax.grad(lambda x: f(x).sum()))(arr)  # doesn't crash

    with self.assertRaises(core.ShardingTypeError):
      arr2 = jax.device_put(np.zeros((16, 128, 8)), P('x', None, 'y'))
      f(arr2)

  @parameterized.named_parameters(
      ('spec1', P('x', 'y', None)),
      ('spec2', P('x', None, 'y')),
      ('spec3', P(None, 'x', 'y')),
      ('spec4', P(('x', 'y'), None, None))
  )
  @jtu.with_explicit_mesh((4, 2), ('x', 'y'))
  def test_reduce_window(self, spec, mesh):
    arr = jax.device_put(np.zeros((16, 128, 8)), spec)

    @jax.jit
    def f(x):
      out = jax.lax.reduce_window(
          x, -np.inf, jax.lax.max, (1,2,1), (1,2,1), 'SAME')
      self.assertEqual(out.aval.sharding.spec, spec)
      return out

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, spec))
    self.check_wsc_in_lowered(f.lower(arr).as_text())

    jax.jit(jax.grad(lambda x: f(x).sum()))(arr)  # doesn't crash

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_jnp_dot(self, mesh):
    np_inp1 = np.arange(16).reshape(8, 2)
    np_inp2 = np.arange(16).reshape(2, 8)
    arr1 = jax.device_put(np_inp1, P('x', 'y'))
    arr2 = jax.device_put(np_inp2, P('x', 'y'))

    @jax.jit
    def f(x, y):
      out = jnp.dot(x, y, out_sharding=P('x'))
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    out = f(arr1, arr2)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))
    self.assertArraysEqual(out, np.dot(np_inp1, np_inp2))

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_jnp_ravel(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, P('x', 'y'))

    @jax.jit
    def f(x):
      out = jnp.ravel(x, out_sharding=P('x'))
      self.assertEqual(out.aval.sharding.spec, P('x'))
      return out

    out = f(arr)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))
    self.assertArraysEqual(out, np.ravel(np_inp))

  @jtu.with_explicit_mesh((4, 2), ('x', 'y'))
  def test_broadcast_forwarding(self, mesh):
    arr = jax.device_put(np.zeros(()), P())

    def f(x):
      out = jax.lax.full_like(x, 1.0)
      self.assertEqual(jax.typeof(out).sharding, jax.typeof(x).sharding)
      return out

    f(arr)  # doesn't crash
    jax.jit(f)(arr)  # doesn't crash

  @config.use_shardy_partitioner(True)
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_unreduced_basic(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    x = jax.device_put(np_inp, P('x', 'y'))
    y = jax.device_put(np_inp.T, P('y', None))
    a = jax.device_put(np_inp, P('x', 'y'))
    b = jax.device_put(np_inp.T, P('y', None))

    @jax.jit
    def f(x, y, a, b):
      m1 = jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', unreduced={'y'}))
      self.assertEqual(m1.aval.sharding.spec, P('x', None, unreduced={'y'}))

      m2 = jnp.einsum('xy,yz->xz', a, b, out_sharding=P('x', unreduced={'y'}))
      self.assertEqual(m2.aval.sharding.spec, P('x', None, unreduced={'y'}))

      s = m1 + m2  # unreduced
      self.assertEqual(s.aval.sharding.spec, P('x', None, unreduced={'y'}))

      out = reshard(s, P('x'))  # reduce
      self.assertEqual(out.aval.sharding.spec, P('x', None))
      return out

    traced = f.trace(x, y, a, b)
    lowered_text = traced.lower().as_text()
    self.assertIn('unreduced={"y"}', lowered_text)
    self.assertTrue(lowered_text.count('unreduced={"y"}') == 3)

  @jtu.with_explicit_mesh((2, 2, 1), ('x', 'y', 'z'))
  def test_dot_general_unreduced_error(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    # Case 1
    x = jax.device_put(np_inp, P('x', 'y'))
    y = jax.device_put(np_inp.T, P('y', None))

    @jax.jit
    def f(x, y):
      return jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', unreduced={'z'}))
    with self.assertRaisesRegex(
        core.ShardingTypeError,
        "unreduced axes should be equal to the contracting specs"):
      f.trace(x, y)

    # Case 2
    x = jax.device_put(np_inp, P('x', 'y'))
    y = jax.device_put(np_inp.T, P(None, None))
    @jax.jit
    def g(x, y):
      return jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', unreduced={'y'}))
    with self.assertRaisesRegex(
        core.ShardingTypeError,
        "lhs and rhs contracting dims should be sharded identically"):
      g.trace(x, y)

    # Case 3
    x = jax.device_put(np_inp, P('x', None))
    y = jax.device_put(np_inp.T, P(None, None))

    @jax.jit
    def h(x, y):
      return jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', unreduced={'y'}))
    with self.assertRaisesRegex(
        core.ShardingTypeError,
        "unreduced axes should be equal to the contracting specs"):
      h.trace(x, y)

  @jtu.with_explicit_mesh((2, 2, 1), ('x', 'y', 'z'))
  def test_add_unreduced_error(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    x = jax.device_put(np_inp, P('x', 'y'))
    y = jax.device_put(np_inp.T, P('y', None))
    a = jax.device_put(np_inp, P('x', 'z'))
    b = jax.device_put(np_inp.T, P('z', None))

    @jax.jit
    def f(x, y, a, b):
      m1 = jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', unreduced={'y'}))
      m2 = jnp.einsum('xy,yz->xz', a, b, out_sharding=P('x', unreduced={'z'}))
      return m1 + m2

    with self.assertRaisesRegex(
        core.ShardingTypeError,
        "lhs and rhs to `add` must be unreduced along the same mesh axes"):
      f.trace(x, y, a, b)

    @jax.jit
    def g(x, y):
      m1 = jnp.einsum('xy,yz->xz', x, y, out_sharding=P('x', unreduced={'y'}))
      m2 = jnp.einsum('xy,yz->xz', a, b, out_sharding=P('x'))
      return m1 + m2

    with self.assertRaisesRegex(
        core.ShardingTypeError, "lhs is unreduced while rhs is not"):
      g.trace(x, y)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_eval_shape(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, P('x', 'y'))

    @jax.jit
    def f(x):
      return x * 2

    out = jax.eval_shape(f, arr)
    self.assertIsInstance(out, jax.ShapeDtypeStruct)
    self.assertEqual(out.sharding,
                     NamedSharding(mesh.abstract_mesh, P('x', 'y')))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_he_normal(self, mesh):
    init = jax.nn.initializers.he_normal(in_axis=0, out_axis=1)
    key = jax.random.key(0)
    out = init(key, (8, 2), jnp.float32, out_sharding=P('x'))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_nn_uniform(self, mesh):
    init = jax.nn.initializers.uniform()
    key = jax.random.key(0)
    out = init(key, (8, 2), jnp.float32, out_sharding=P('x'))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_nn_constant(self, mesh):
    init = jax.nn.initializers.constant(-7)
    key = jax.random.key(0)
    out = init(key, (8, 2), jnp.float32, out_sharding=P('x'))
    self.assertArraysEqual(out, jnp.full((8, 2), -7, dtype=jnp.float32))
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x', None)))

  @config.numpy_rank_promotion('allow')
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_lax_map(self, mesh):
    def simple_func(w, x):
      return jnp.sum(w * x, axis=-1)

    w = jax.device_put(np.arange(4, dtype=np.float32), P('x'))
    x = jax.device_put(np.ones((4, 2, 4), dtype=np.float32),
                       P(None, 'y', None))

    jax.lax.map(lambda _x: simple_func(w, _x), x)  # doesn't crash

    jax.lax.map(lambda _x: simple_func(w, _x), x, batch_size=2)  # doesn't crash

  @config.numpy_rank_promotion('allow')
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_lax_map_remainder(self, mesh):
    def simple_func(w, x):
      return jnp.sum(w * x, axis=-1)

    w = jax.device_put(np.arange(4, dtype=np.float32), P())
    x = jax.device_put(np.ones((5, 2, 4), dtype=np.float32),
                       P(None, 'x', None))

    jax.lax.map(lambda _x: simple_func(w, _x), x, batch_size=2)  # doesn't crash

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_extended_dtypes(self, mesh):
    dtype = primal_tangent_dtype(jnp.dtype('int8'), jnp.dtype('bfloat16'))

    @jax.jit
    def f(x):
      x = jax.lax.convert_element_type(x, dtype)
      self.assertEqual(x.aval.sharding.spec, P('x'))
      x = jax.lax.convert_element_type(x, 'int8')
      self.assertEqual(x.aval.sharding.spec, P('x'))

    x = jax.device_put(jnp.arange(8, dtype='int8'), P('x',))
    f(x)  # doesn't crash


@jtu.pytest_mark_if_available('multiaccelerator')
class PJitErrorTest(jtu.JaxTestCase):

  @check_1d_2d_mesh(set_mesh=True)
  def testNonDivisibleArgs(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources, None)
    mesh_size = str(math.prod([dim[1] for dim in mesh]))
    error = re.compile(
        r"One of pjit arguments with pytree key path x.*" + spec_regex(spec) + r".*"
        r"implies that the global size of its dimension 0 should be "
        r"divisible by " + mesh_size + r", but it is equal to 3 "
        r"\(full shape: \(3, 2\)\)", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x, in_shardings=spec, out_shardings=None)(x)

  @check_1d_2d_mesh(set_mesh=True)
  def testNonDivisibleOuts(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources, None)
    mesh_size = str(math.prod([dim[1] for dim in mesh]))
    error = re.compile(
        r"One of pjit outputs with pytree key path result\['rrr'\].*" + spec_regex(spec) + r".*"
        r"implies that the global size of its dimension 0 should be "
        r"divisible by " + mesh_size + r", but it is equal to 3", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: {'rrr': x}, in_shardings=None,
           out_shardings=P(resources, None))(x)

  @check_1d_2d_mesh(set_mesh=False)
  @jtu.with_mesh([('z', 1)])
  def testUndefinedResourcesArgs(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(
        ValueError,
        r"Resource axis: x of.*" + spec_regex(spec) + r" is not found in mesh: \(.*\)."):
      pjit(lambda x: x, in_shardings=spec, out_shardings=None)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @jtu.with_mesh([('z', 1)])
  def testUndefinedResourcesOuts(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(
        ValueError,
        r"Resource axis: x of.*" + spec_regex(spec) + r" is not found in mesh: \(.*\)."):
      pjit(lambda x: x, in_shardings=None, out_shardings=spec)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @jtu.with_mesh([('z', 1)])
  def testUndefinedResourcesConstraint(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(
        ValueError,
        r"Resource axis: x of.*" + spec_regex(spec) + r" is not found in mesh: \(.*\)."):
      pjit(
          lambda x: with_sharding_constraint(x, spec),
          in_shardings=None,
          out_shardings=None,
      )(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowArgs(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = re.compile(
        r"One of pjit arguments.*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 1", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_shardings=spec, out_shardings=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowArgsAxisResourcesNone(self):
    x = jnp.arange(2)
    spec = P(None, None)
    error = re.compile(
        r"One of pjit arguments.*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 1", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_shardings=spec, out_shardings=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowOuts(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = re.compile(
        r"One of pjit outputs.*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 0", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_shardings=None, out_shardings=spec)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowConstraint(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = re.compile(
        r"One of with_sharding_constraint arguments" + r".*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 1", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(
          lambda x: with_sharding_constraint(x, spec), in_shardings=None,
          out_shardings=None,
      )(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRepeatedInResources(self):
    x = jnp.arange(2)
    for spec in [P('x', 'x'), P('x', ('y', 'x'))]:
      error = (r"A single in_shardings specification can map every mesh "
               r"axis to at most one positional dimension, but " +
               spec_regex(spec) + " has duplicate entries for `x`")
      with self.assertRaisesRegex(DuplicateSpecError, error):
        pjit(lambda x: x, in_shardings=spec, out_shardings=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRepeatedOutResources(self):
    x = jnp.arange(2)
    for spec in [P('x', 'x'), P('x', ('y', 'x'))]:
      error = (r"A single out_shardings specification can map every mesh "
               r"axis to at most one positional dimension, but " +
               spec_regex(spec) + " has duplicate entries for `x`")
      with self.assertRaisesRegex(DuplicateSpecError, error):
        pjit(lambda x: x, in_shardings=None, out_shardings=spec)(x)

  def testEmptyMesh(self):
    out = pjit(lambda x: x, in_shardings=None, out_shardings=None)(jnp.arange(4))
    self.assertEqual(out.sharding, SingleDeviceSharding(jax.devices()[0]))

  def test_pspec_to_wsc_without_mesh(self):
    error = (
        r'with_sharding_constraint requires a non-empty mesh if you are '
        r'passing `PartitionSpec`s or `None` to shardings.*')
    with self.assertRaisesRegex(RuntimeError, error):
      pjit(lambda x: with_sharding_constraint(x, P('x')))(jnp.arange(4))

  @jtu.with_mesh([('x', 2)])
  def testAxisResourcesMismatch(self):
    x = jnp.ones([])
    p = [None, None, None]

    pjit(lambda x: x, (p,), p)([x, x, x])  # OK

    error = re.escape(
        "pjit in_shardings specification must be a tree prefix of the "
        "positional arguments tuple passed to the `pjit`-decorated function. "
        "In particular, pjit in_shardings must either be a None, a "
        "PartitionSpec, or a tuple of length equal to the number of positional "
        "arguments. But pjit in_shardings is the wrong length: got a "
        "tuple or list of length 3 for an args tuple of length 2.")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x, y: x, p, p)(x, x)

    Foo = namedtuple('Foo', ['x'])
    error = "in_shardings is not a tuple.*might need to be wrapped"
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x, Foo(None), Foo(None))(Foo(x))

    pjit(lambda x: x, (Foo(None),), Foo(None))(Foo(x))  # OK w/ singleton tuple

    # TODO(apaszke,mattjj): Disable implicit list casts and enable this
    # error = ("it looks like pjit in_axis_resources might need to be wrapped in "
    #          "a singleton tuple.")
    # with self.assertRaisesRegex(ValueError, error):
    #   pjit(lambda x, y: x, p, p)([x, x, x])

    # TODO(apaszke): Disable implicit list casts and enable this
    # error = re.escape(
    # r"pjit in_axis_resources specification must be a tree prefix of the "
    # r"corresponding value, got specification (None, None, None) for value "
    # r"tree PyTreeDef(([*, *, *],)). Note that pjit in_axis_resources that "
    # r"are non-trivial pytrees should always be wrapped in a tuple representing "
    # r"the argument list. In particular, you're passing in a single argument "
    # r"which means that pjit in_axis_resources might need to be wrapped in a "
    # r"singleton tuple.")
    # with self.assertRaisesRegex(ValueError, error):
    # pjit(lambda x: x, p, p)([x, x, x])  # Error, but make sure we hint at singleton tuple

    error = re.escape(
        "pytree structure error: different lengths of list at "
        "key path\n"
        "    pjit out_shardings\n")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x, (p,), [p, None])([x, x, x])  # Error, we raise a generic tree mismatch message

  @jtu.with_mesh([('x', 2)])
  def testNestedDifferentResources(self):
    @partial(pjit, in_shardings=P('x'), out_shardings=None)
    def f(x):
      with jax.sharding.Mesh(np.array([jax.local_devices()[0]]), ('x')):
        @partial(pjit, in_shardings=P('x'), out_shardings=None)
        def h(x):
          return x
        return h(x)
    xshape = (2, 5, 6)
    x = jnp.arange(math.prod(xshape)).reshape(xshape)
    with self.assertRaisesRegex(
        ValueError, "Received incompatible devices for jitted computation.*"):
      f(x)

  @parameterized.named_parameters(
      ("committed", True),
      ("uncommitted", False),
  )
  def test_pjit_with_deleted_input_at_first_call(self, committed):
    shape = (8,)
    mesh = jtu.create_mesh((1,), ('x',))
    inp_data = np.arange(math.prod(shape)).reshape(shape)
    if committed:
      s = NamedSharding(mesh, P('x',))
      x = jax.device_put(inp_data, s)
    else:
      x = jax.device_put(inp_data)
    f = pjit(lambda x: x + 1)
    with self.assertRaisesRegex(RuntimeError, 'Array has been deleted.'):
      x.delete()
      _ = f(x)

  @parameterized.named_parameters(
      ("committed", True),
      ("uncommitted", False),
  )
  def test_pjit_with_deleted_input_at_subsequent_call(self, committed):
    shape = (8,)
    mesh = jtu.create_mesh((1,), ('x',))
    inp_data = np.arange(math.prod(shape)).reshape(shape)
    if committed:
      s = NamedSharding(mesh, P('x',))
      x = jax.device_put(inp_data, s)
    else:
      x = jax.device_put(inp_data)
    f = pjit(lambda x: x + 1)
    _ = f(x)
    with self.assertRaisesRegex((RuntimeError, ValueError),
                                '.*(Array|buffer|Buffer) has been deleted.*'):
      x.delete()
      _ = f(x)

  def test_aot_error_on_dced_avals_mismatch(self):
    x, y1, y2 = jnp.ones(4), jnp.ones(4), jnp.ones(1)

    @jax.jit
    def f(x, y):
      return x + 1 if y.shape[0] > 2 else x + 2

    f_out1 = f(x, y1)
    f(x, y2)

    g = f.lower(x, y1).compile()
    g_out1 = g(x, y1)
    self.assertArraysEqual(f_out1, g_out1)

    with self.assertRaisesRegex(
        TypeError,
        'Argument types differ from the types for which this computation was'
        ' compiled'):
      g(x, y2)

  def test_dce_no_array(self):
    mesh = jtu.create_mesh((2,), ('x',))
    arr = jax.device_put(np.arange(8.), NamedSharding(mesh, P('x')))

    @jax.jit
    def f(a, b, c):
      return a, c

    f(arr, 2., 3.)
    f(arr, 2., 3.)  # doesn't crash

  def test_named_sharding_of_none(self):
    mesh = jtu.create_mesh((2,), ('x',))
    with self.assertRaisesRegex(TypeError, 'Unexpected None'):
      jax.NamedSharding(mesh, None)


@jtu.pytest_mark_if_available('multiaccelerator')
class UtilTest(jtu.JaxTestCase):

  def testOpShardingRoundTrip(self):
    FakeDevice = namedtuple('FakeDevice', ['id'])
    mesh_named_shape = OrderedDict([('a', 2), ('b', 3), ('c', 4), ('d', 7), ('e', 4)])
    mesh_axes, mesh_shape = unzip2(mesh_named_shape.items())
    devices = [FakeDevice(i) for i in range(math.prod(mesh_shape))]
    mesh = pxla.Mesh(np.array(devices).reshape(*mesh_shape), tuple(mesh_axes))

    dims = 5
    aval = core.ShapedArray((len(devices),) * dims, jnp.float32)
    def roundtrip(spec):
      hlo_sharding = NamedSharding(mesh, spec)._to_xla_hlo_sharding(aval.ndim)
      recovered_spec = parse_flatten_op_sharding(hlo_sharding, mesh)[0]
      self.assertEqual(recovered_spec[:len(spec)], spec)
      self.assertEqual(recovered_spec[len(spec):], ((),) * (len(recovered_spec) - len(spec)))

    special_specs = [P()]
    for spec in special_specs:
      roundtrip(spec)

    rng = self.rng()
    for i in range(100):
      spec = [()] * dims
      for axis in rng.permutation(mesh_axes)[:rng.randint(low=1, high=len(mesh_axes) + 1)]:
        spec[rng.choice(dims)] += (axis,)
      while spec and spec[-1] == ():
        spec.pop()
      roundtrip(P(*spec))

  @parameterized.named_parameters(
      ("linear", {'x': 0, 'y': 1, 'z': 2}, P('x', 'y', 'z')),
      ("combine", {'x': 0, 'y': 0, 'z': 1}, P(('x', 'y'), 'z')),
      ("skip", {'x': 0, 'y': 0, 'z': 2}, P(('x', 'y'), None, 'z')),
      ("multi_skip", {'x': 0, 'y': 1, 'z': 3}, P('x', 'y', None, 'z')),
  )
  def test_array_mapping_to_axis_resources(self, inp, expected_out):
    self.assertEqual(
        sharding_impls.array_mapping_to_axis_resources(inp), expected_out
    )

  def test_op_sharding_equality_and_hash_equality(self):
    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [2, 2]
    op1.tile_assignment_devices = [0, 1, 2, 3]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.OTHER
    op2.tile_assignment_dimensions = [2, 2]
    op2.tile_assignment_devices = [0, 1, 2, 3]

    op3 = xc.OpSharding()
    op3.type = xc.OpSharding.Type.OTHER
    op3.tile_assignment_dimensions = [4, 2]
    op3.tile_assignment_devices = [0, 1, 2, 3, 4, 5, 6, 7]

    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op2))
    self.assertFalse(op_shardings.are_op_shardings_equal(op1, op3))
    self.assertFalse(op_shardings.are_op_shardings_equal(op2, op3))

    hs1 = xc.HloSharding.from_proto(op1)
    hs2 = xc.HloSharding.from_proto(op2)
    hs3 = xc.HloSharding.from_proto(op3)

    self.assertEqual(hs1, xc.HloSharding.iota_tile((2, 2)))
    self.assertEqual(hs2, xc.HloSharding.iota_tile((2, 2)))
    self.assertEqual(hs3, xc.HloSharding.iota_tile((4, 2)))
    self.assertEqual(hs1.num_devices(), 4)
    self.assertEqual(hs1.num_dimensions(), 2)
    self.assertEqual(hs1.tile_assignment_dimensions(), [2, 2])
    self.assertEqual(hs1.tile_assignment_devices(), [0, 1, 2, 3])
    self.assertTrue(hs1.is_tiled())
    self.assertFalse(hs1.replicate_on_last_tile_dim())
    self.assertEqual(hash(hs1), hash(hs2))
    self.assertNotEqual(hash(hs1), hash(hs3))
    self.assertNotEqual(hash(hs2), hash(hs3))

  def test_op_sharding_partial_sharding(self):
    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [4, 1]
    op1.tile_assignment_devices = [0, 2, 1, 3]
    op1.last_tile_dims = [xc.OpSharding.Type.REPLICATED]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.OTHER
    op2.tile_assignment_dimensions = [4, 1]
    op2.tile_assignment_devices = [0, 2, 1, 3]
    op2.last_tile_dims = [xc.OpSharding.Type.REPLICATED]

    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op2))

    hs1 = xc.HloSharding.from_proto(op1)
    hs2 = xc.HloSharding.from_proto(op2)
    self.assertEqual(
        hs1,
        xc.HloSharding.iota_tile(
            (4, 1),
            reshape_dims=(2, 2),
            transpose_perm=(1, 0),
            subgroup_types=[xc.OpSharding.Type.REPLICATED],
        ),
    )
    self.assertFalse(hs1.subgroup_types())
    self.assertTrue(hs1.is_tiled())
    self.assertEqual(
        hs2,
        xc.HloSharding.iota_tile(
            (4, 1),
            reshape_dims=(2, 2),
            transpose_perm=(1, 0),
            subgroup_types=[xc.OpSharding.Type.REPLICATED],
        ),
    )
    self.assertFalse(hs2.subgroup_types())
    self.assertTrue(hs2.is_tiled())
    self.assertEqual(hash(hs1), hash(hs2))

  def test_op_sharding_tuple_shardings(self):
    top1 = xc.OpSharding()
    top1.type = xc.OpSharding.Type.OTHER
    top1.tile_assignment_dimensions = [4, 1]
    top1.tile_assignment_devices = [0, 1, 2, 3]
    top1.replicate_on_last_tile_dim = True

    top2 = xc.OpSharding()
    top2.type = xc.OpSharding.Type.OTHER
    top2.tile_assignment_dimensions = [2, 2]
    top2.tile_assignment_devices = [0, 1, 2, 3]
    top2.replicate_on_last_tile_dim = True

    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.TUPLE
    op1.tuple_shardings = [top1, top2]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.TUPLE
    op2.tuple_shardings = [top2, top1]

    self.assertFalse(op_shardings.are_op_shardings_equal(op1, op2))

    hs1 = xc.HloSharding.from_proto(op1)
    hs2 = xc.HloSharding.from_proto(op2)
    self.assertNotEqual(hash(hs1), hash(hs2))

  def test_hlo_sharding_iota_tile_error(self):
    self.assertRaisesRegex(
        _jax.XlaRuntimeError,
        'INVALID_ARGUMENT: `dims` should not be empty.',
        lambda: xc.HloSharding.iota_tile(())
    )
    self.assertRaisesRegex(
        _jax.XlaRuntimeError,
        'INVALID_ARGUMENT: Cannot reshape from',
        lambda: xc.HloSharding.iota_tile(
            (2, 2),
            reshape_dims=(2, 4),
            transpose_perm=(1, 0),
        ),
    )
    self.assertRaisesRegex(
        _jax.XlaRuntimeError,
        'INVALID_ARGUMENT: `reshape_dims` and `transpose_perm` should have the'
        ' same size',
        lambda: xc.HloSharding.iota_tile(
            (2, 2),
            transpose_perm=(1, 0),
        ),
    )
    self.assertRaisesWithLiteralMatch(
        _jax.XlaRuntimeError,
        'INVALID_ARGUMENT: `subgroup_types`(3) should not have more dimensions '
        'than `dims`(2).',
        lambda: xc.HloSharding.iota_tile(
            (2, 2),
            subgroup_types=(
                xc.OpSharding.Type.REPLICATED,
                xc.OpSharding.Type.MANUAL,
                xc.OpSharding.Type.REPLICATED,
            ),
        ),
    )

  def test_device_indices_cache(self):
    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [1, 1, 2, 1]
    op1.tile_assignment_devices = [0, 1]
    op1.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.REPLICATED

    shape = (8, 4)
    devices = jax.devices()

    ops = GSPMDSharding(devices, op1)
    ops.devices_indices_map(shape)
    cache_info1 = common_devices_indices_map.cache_info()

    ops.devices_indices_map(shape)
    cache_info2 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)

    ops = GSPMDSharding(devices, op2)
    ops.devices_indices_map(shape)
    cache_info3 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info3.hits, cache_info2.hits + 1)

    ops.devices_indices_map(shape)
    cache_info4 = common_devices_indices_map.cache_info()
    self.assertEqual(cache_info4.hits, cache_info3.hits + 1)

  def test_op_sharding_semantically_replicated(self):
    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [1, 1, 2]
    op1.tile_assignment_devices = [0, 1]
    op1.last_tile_dims = [xc.OpSharding.Type.REPLICATED]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.REPLICATED

    op3 = xc.OpSharding()
    op3.type = xc.OpSharding.Type.OTHER
    op3.tile_assignment_dimensions = [1, 1, 1, 1]
    op3.tile_assignment_devices = [0]
    op3.last_tile_dims = [xc.OpSharding.Type.REPLICATED]

    op4 = xc.OpSharding()
    op4.type = xc.OpSharding.Type.OTHER
    op4.tile_assignment_dimensions = [1]
    op4.tile_assignment_devices = [0]

    self.assertTrue(op_shardings.is_op_sharding_replicated(op1))
    self.assertTrue(op_shardings.is_op_sharding_replicated(op2))
    self.assertTrue(op_shardings.is_op_sharding_replicated(op3))
    self.assertTrue(op_shardings.is_op_sharding_replicated(op4))
    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op2))
    self.assertTrue(op_shardings.are_op_shardings_equal(op2, op3))
    self.assertTrue(op_shardings.are_op_shardings_equal(op3, op4))

  def test_op_sharding_manual_replicated(self):
    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [1, 1, 2, 1]
    op1.tile_assignment_devices = [0, 1]
    op1.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.OTHER
    op2.tile_assignment_dimensions = [1, 1, 1, 2]
    op2.tile_assignment_devices = [0, 1]
    op2.last_tile_dims = [xc.OpSharding.Type.MANUAL, xc.OpSharding.Type.REPLICATED]

    op3 = xc.OpSharding()
    op3.type = xc.OpSharding.Type.REPLICATED

    self.assertTrue(op_shardings.is_op_sharding_replicated(op1))
    self.assertTrue(op_shardings.is_op_sharding_replicated(op2))
    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op2))
    self.assertTrue(op_shardings.are_op_shardings_equal(op1, op3))

    hs1 = xc.HloSharding.from_proto(op1)
    self.assertEqual(
        hs1,
        xc.HloSharding.iota_tile(
            (1, 1, 2, 1),
            subgroup_types=(
                xc.OpSharding.Type.REPLICATED,
                xc.OpSharding.Type.MANUAL,
            ),
        )
    )
    self.assertTrue(hs1.is_replicated())
    self.assertFalse(hs1.replicate_on_last_tile_dim())

    hs2 = xc.HloSharding.from_proto(op2)
    self.assertEqual(
        xc.HloSharding.from_proto(op2),
        xc.HloSharding.iota_tile(
            (1, 1, 1, 2),
            subgroup_types=(
                xc.OpSharding.Type.MANUAL,
                xc.OpSharding.Type.REPLICATED,
            ),
        )
    )
    self.assertTrue(hs2.is_replicated())
    self.assertFalse(hs2.replicate_on_last_tile_dim())
    self.assertEqual(
        xc.HloSharding.from_proto(op3), xc.HloSharding.replicate()
    )

  def test_hlo_sharding_manual_replicated(self):
    hs1 = xc.HloSharding.manual()
    self.assertTrue(hs1.is_manual())
    self.assertFalse(hs1.tile_assignment_devices())

    hs2 = xc.HloSharding.replicate()
    self.assertTrue(hs2.is_replicated())
    self.assertFalse(hs2.tile_assignment_devices())

    hs3 = xc.HloSharding.iota_tile(
        (3, 3),
        subgroup_types=(
            xc.OpSharding.Type.MANUAL,
            xc.OpSharding.Type.REPLICATED,
        ),
    )
    self.assertFalse(hs3.is_manual())
    self.assertFalse(hs3.is_replicated())
    self.assertEqual(hs3.num_dimensions(), 2)
    self.assertEqual(hs3.tile_assignment_dimensions(), [3, 3])
    self.assertEqual(hs3.num_devices(), 9)
    self.assertEqual(hs3.tile_assignment_devices(), list(range(0, 9)))
    self.assertEqual(
        hs3.subgroup_types(),
        [xc.OpSharding.Type.MANUAL, xc.OpSharding.Type.REPLICATED],
    )
    self.assertFalse(hs3.replicate_on_last_tile_dim())
    self.assertTrue(hs3.is_tiled())

    hs4 = xc.HloSharding.iota_tile(
        (3, 4), subgroup_types=[xc.OpSharding.Type.REPLICATED]
    )
    self.assertTrue(hs4.replicate_on_last_tile_dim())
    self.assertFalse(hs4.subgroup_types())
    self.assertTrue(hs4.is_tiled())

  def test_hlo_sharding_with_device_ordering(self):
    hs1 = xc.HloSharding.subgroup_with_device_ordering(
        np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int64),
        subgroup_types=[xc.OpSharding.Type.REPLICATED],
    )
    self.assertEqual(
        hs1,
        xc.HloSharding.iota_tile(
            (2, 2, 2), subgroup_types=[xc.OpSharding.Type.REPLICATED]
        ),
    )

  @jtu.thread_unsafe_test()
  def test_op_sharding_cache_on_mesh_pspec_sharding(self):
    ndim = 2
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    mps1 = NamedSharding(mesh, P('x', 'y'))
    sharding_impls.named_sharding_to_xla_hlo_sharding.cache_clear()
    op1 = mps1._to_xla_hlo_sharding(ndim)
    cache_info1 = sharding_impls.named_sharding_to_xla_hlo_sharding.cache_info()

    mps2 = NamedSharding(mesh, P('x', 'y'))
    op2 = mps2._to_xla_hlo_sharding(ndim)
    cache_info2 = sharding_impls.named_sharding_to_xla_hlo_sharding.cache_info()

    self.assertEqual(id(op1), id(op2))
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)
    self.assertEqual(cache_info2.currsize, cache_info1.currsize)

  def test_get_partition_spec(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y', None))

    spec = parse_flatten_op_sharding(s._to_xla_hlo_sharding(3), mesh)[0]
    self.assertEqual(spec, P('x', 'y'))

  def test_mesh_with_list_devices(self):
    mesh = jax.sharding.Mesh(jax.devices(), ('x',))
    self.assertIsInstance(mesh.devices, np.ndarray)
    self.assertEqual(mesh.size, jax.device_count())

  def test_mesh_with_string_axis_names(self):
    mesh = jax.sharding.Mesh(jax.devices(), 'dp')
    self.assertTupleEqual(mesh.axis_names, ('dp',))

  def test_sharded_in_place_assignment(self):
    mesh = jtu.create_mesh((8,), ('data',))

    idx = [0,  2,  5,  7,  8, 10, 13, 15]
    n = 16
    def _init():
      w = jnp.zeros((n, n))
      idx1 = jnp.array(idx)
      w = w.at[idx1, jnp.arange(n//2)].set(1)
      return w

    w = jax.jit(_init, out_shardings=NamedSharding(mesh, P(None, 'data')))()

    w_gt = np.zeros((n, n))
    for j, i in enumerate(idx):
      w_gt[i, j] = 1

    self.assertArraysEqual(w, w_gt)

  def test_get_intermediate_shardings(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x'))
    arr = jax.device_put(np.arange(8), s)

    @jax.jit
    def g(x):
      x = with_sharding_constraint(x, s)
      return with_sharding_constraint(x, s)

    @jax.jit
    def f(x, y):
      x, y = with_sharding_constraint((x, y), s)
      x, y = shard_map(lambda x, y: (x, y), mesh=mesh, in_specs=P('x'),
                       out_specs=P('x'))(x, y)
      x, y = jax.device_put((x, y), s)
      x, y = jax.jit(lambda x, y: (x, y), in_shardings=s, out_shardings=s)(x, y)
      return g(x), y

    jaxpr = f.trace(arr, arr).jaxpr
    out = dispatch.get_intermediate_shardings(jaxpr)
    self.assertLen(out, 16)


@jtu.with_config(jax_use_shardy_partitioner=True)
class ShardyTest(jtu.JaxTestCase):

  def test_lowering_input_output_sharding(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    @partial(jax.jit, out_shardings=s)
    def f(x):
      return x * 2

    self.assertIn('sdy.sharding = #sdy.sharding', f.lower(arr).as_text())

  def test_lowering_with_sharding_constraint(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    arr = np.arange(16).reshape(4, 2, 2)

    @jax.jit
    def f(x):
      return jax.lax.with_sharding_constraint(
          x, NamedSharding(mesh, P('x', None, 'y')))
    lowered_str = jax.jit(f).lower(arr).as_text()
    self.assertIn('sdy.sharding_constraint', lowered_str)
    self.assertIn('<@mesh, [{"x"}, {}, {"y"}]>', lowered_str)

  def test_lowering_with_sharding_constraint_unconstrained(self):
    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    arr = np.arange(16).reshape(4, 2, 2)

    @jax.jit
    def f(x):
      return jax.lax.with_sharding_constraint(
          x, NamedSharding(mesh, P('x', P.UNCONSTRAINED, 'y')))
    lowered_str = f.lower(arr).as_text()
    self.assertIn('sdy.sharding_constraint', lowered_str)
    self.assertIn('<@mesh, [{"x"}, {?}, {"y"}]>', lowered_str)

  # TODO(bartchr): run on CPU once Shardy is added to the XLA CPU pipeline.
  @jtu.skip_on_devices('cpu')
  def test_compile_with_inferred_out_sharding(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    x = jax.device_put(np.arange(8 * 4).reshape(8, 4),
                       NamedSharding(mesh, P('x', 'y')))
    y = jax.device_put(np.arange(4 * 16).reshape(4, 16),
                       NamedSharding(mesh, P('y')))

    @jax.jit
    def f(x, y):
      return x @ y

    out = f(x, y)
    self.assertArraysEqual(out, x @ y)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('x')))

  def test_fully_automatic_sharding(self):
    mesh = jtu.create_mesh((8,), ('x',))
    x = jax.ShapeDtypeStruct((128, 128), jnp.float32)

    @jax.jit
    def f(x, y):
      return x @ y

    lowered_str = jax.jit(f, in_shardings=[AUTO(mesh), AUTO(mesh)]).lower(x, x).as_text()
    self.assertIn('sdy.mesh @mesh = <["x"=8]>', lowered_str)

  def test_array_sharding_repr_with_priority(self):
    sharding = sharding_impls.SdyArray(
        mesh_shape=(('data', 4), ('model', 8), ('expert', 2)),
        dim_shardings=[
            sharding_impls.SdyDim(axes=['data', 'expert'], is_open=False),
            sharding_impls.SdyDim(axes=['model'], is_open=True, priority=2)])
    self.assertEqual(repr(sharding), "SdyArray([{'data', 'expert'}, {'model', ?}p2])")

  def test_array_sharding_repr_with_logical_ids(self):
    abstract_mesh = jax.sharding.AbstractMesh((4, 8, 2), ('x', 'y', 'z'))
    ns = NamedSharding(abstract_mesh, P(('x', 'y'), 'z', P.UNCONSTRAINED, None),
                       _logical_device_ids=[4, 5, 6, 7, 0, 1, 2, 3])
    self.assertEqual(repr(ns._to_sdy_sharding(4)),
                     "SdyArray([{'x', 'y'}, {'z'}, {?}, {}], "
                     "device_ids=[4, 5, 6, 7, 0, 1, 2, 3])")

  def test_dimension_sharding_repr(self):
    dim_sharding = sharding_impls.SdyDim(
        axes=['data', 'model'], is_open=True, priority=2)
    self.assertEqual(repr(dim_sharding),
                     "SdyDim({'data', 'model', ?}p2)")

  def test_tensor_dialect(self):
    # While this doesn't emit any `mlir::TensorDialect` ops, some pass in the
    # compiler pipeline is temporarily introducing it before then discarding it
    # again. Make sure this doesn't crash.
    mesh = jtu.create_mesh((2,), ('x'))
    in_sds = jax.ShapeDtypeStruct((4, 8), jnp.float32)

    @partial(jax.jit, out_shardings=NamedSharding(mesh, P('x')))
    def gen_dummy_inputs():
      return tuple(jax.random.normal(jax.random.key(42), shape=in_sds.shape
                   ).astype(in_sds.dtype))
    gen_dummy_inputs()  # doesn't crash

  @jtu.skip_on_devices('cpu')
  def test_custom_partition_with_sharding_rule_callback(self):
    if jtu.is_cloud_tpu():
      raise unittest.SkipTest("Custom partitioning is not supported on libtpu.")

    def partition(static_arg0, static_arg1, mesh, arg_shapes, result_shape):
      arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
      result_sharding = result_shape.sharding
      rank = len(arg_shapes[0].shape)

      self.assertEqual(static_arg0, 1)
      self.assertEqual(static_arg1, 2)
      def lower_fn(x, y):
        axis_name = arg_shardings[1].spec[rank-2][0]
        i = jax.lax.axis_index(axis_name)
        z = jax.lax.psum(jax.lax.dynamic_slice_in_dim(
            jax.lax.dynamic_slice_in_dim(x, i * 0, 8, axis=rank-2),
            i * 8, 8, axis=rank-1) @ y, (axis_name))
        return z

      return mesh, lower_fn, (result_sharding), arg_shardings

    def produce_sharding_rule(static_arg0, static_arg1, mesh, arg_shapes, result_shape):
      self.assertEqual(static_arg0, 1)
      self.assertEqual(static_arg1, 2)
      rank = len(arg_shapes[0].shape)
      leading_axes = ""
      for i in range(rank - 2):
        leading_axes += f" b{i}"
      return f"{leading_axes} i j, {leading_axes} j k -> {leading_axes} i k"

    @partial(custom_partitioning, static_argnums=(2,3))
    def f(x, y, static_arg0=1, static_arg1=2):
      return jnp.matmul(x, y)

    f.def_partition(
        infer_sharding_from_operands=None,
        partition=partition,
        sharding_rule=produce_sharding_rule)

    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    x = jax.device_put(np.arange(2 * 3 * 32 * 16).reshape(2, 3, 32, 16),
                       NamedSharding(mesh, P(None, None, 'x')))
    y = jax.device_put(np.arange(2 * 3 * 16 * 32).reshape(2, 3, 16, 32),
                       NamedSharding(mesh, P(None, None,'y')))
    result = jax.jit(f)(x, y)
    expected_result = f(x, y)
    self.assertArraysEqual(result, expected_result)
    self.assertEqual(result.sharding, NamedSharding(mesh, P(None, None, 'x')))

  def test_custom_partition_shardy_migration(self):
    if jtu.is_cloud_tpu():
      raise unittest.SkipTest("Custom partitioning is not supported on libtpu.")

    def partition(mesh, arg_shapes, result_shape):
      def lower_fn(x):
        return x

      return (
          mesh,
          lower_fn,
          arg_shapes[0].sharding,
          (arg_shapes[0].sharding,),
      )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      return arg_shapes[0].sharding

    def propagate_user_sharding(mesh, user_shape):
      return user_shape.sharding

    @custom_partitioning
    def f(x):
      return x

    f.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        propagate_user_sharding=propagate_user_sharding,
    )

    mesh = jtu.create_mesh((4, 2), ('x', 'y'))
    x = jax.device_put(np.arange(32 * 16).reshape(32, 16),
                       NamedSharding(mesh, P(None, 'x')))
    with self.assertRaisesRegex(
        NotImplementedError, 'provide sharding_rule to migrate to Shardy'):
      jax.jit(f)(x)

  def test_reshard_empty_mesh_error(self):
    arr = jax.device_put(np.arange(8), jax.devices()[0])
    with self.assertRaisesRegex(ValueError, "nonempty mesh"):
      reshard(arr, NamedSharding(mesh_lib.empty_abstract_mesh, P(None)))

  def test_reshard_none_sharding_error(self):
    arr = jax.device_put(np.arange(8), jax.devices()[0])
    with self.assertRaisesRegex(ValueError, "non-None"):
      reshard(arr, None)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
