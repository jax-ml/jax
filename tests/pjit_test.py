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

import os
import re
from functools import partial, lru_cache
import logging
import threading
import unittest
from collections import OrderedDict, namedtuple

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.config import parallel_functions_output_gda, jax_array
from jax import dtypes
from jax import stages
from jax.errors import JAXTypeError
from jax import lax
from jax import prng
# TODO(skye): do we still wanna call this PartitionSpec?
from jax.experimental import maps
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import xmap
from jax.experimental import global_device_array
from jax.experimental import array
from jax.experimental.sharding import MeshPspecSharding, Sharding, OpShardingSharding
import jax.experimental.pjit as pjit_lib
from jax.experimental.pjit import (pjit, pjit_p, with_sharding_constraint,
                                   FROM_GDA, AUTO)
from jax.interpreters import pxla
from jax.interpreters import mlir
from jax._src.lib import xla_client as xc, xla_bridge
from jax._src.lib import xla_extension_version
from jax._src.util import prod, curry, unzip2, safe_zip

from jax.config import config
config.parse_flags_with_absl()

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


def create_gda(global_shape, global_mesh, mesh_axes, global_data=None):
  if global_data is None:
    global_data = np.arange(
        prod(global_shape), dtype=np.float32).reshape(global_shape)

  if isinstance(mesh_axes, Sharding):
    mesh_axes = mesh_axes.spec

  return global_device_array.GlobalDeviceArray.from_callback(
      global_shape, global_mesh, mesh_axes, lambda idx: global_data[idx]), global_data


def create_array(global_shape, global_mesh, mesh_axes, global_data=None,
                 dtype=np.float32):
  if global_data is None:
    global_data = np.arange(
        prod(global_shape), dtype=dtype).reshape(global_shape)

  if isinstance(mesh_axes, Sharding):
    sharding = mesh_axes
  else:
    sharding = MeshPspecSharding(global_mesh, mesh_axes)

  return array.make_array_from_callback(
      global_shape, sharding, lambda idx: global_data[idx]), global_data


@lru_cache()
def simulated_cached_fun(s):
  return s


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
class PJitTest(jtu.BufferDonationTestCase):

  @jtu.with_mesh([('x', 1)])
  def testDeviceBufferAval(self):

    @partial(pjit, in_axis_resources=None, out_axis_resources=P('x'))
    def f(x):
      return x

    shape = (2, 2)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x)
    expected = x
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 1)
    self.assertAllClose(
        np.asarray(actual.device_buffers[0]), expected, check_dtypes=False)
    # Repro for a bug on device_buffer aval
    _ = repr(actual.device_buffers)

  @jtu.with_mesh([('x', 2)])
  def testBasic1D(self):
    @partial(pjit,
             in_axis_resources=(P('x'), P('x')),
             out_axis_resources=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), expected,
                        check_dtypes=False)

  @jtu.with_mesh([('x', 2)])
  def testJitOfPjitDisallowed(self):
    @partial(pjit,
             in_axis_resources=(P('x'), P('x')),
             out_axis_resources=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    with self.assertRaises(RuntimeError,
                           msg="Nesting pjit() inside jit() is not allowed."):
      jax.jit(f)(x, x + 1)

  @jtu.with_mesh([('x', 2)])
  def testUnevenShardingConstraint(self):
    @partial(pjit,
             in_axis_resources=(P('x'), P('x')),
             out_axis_resources=None)
    def f(x, y):
      x = x[:3]
      y = y[:3]
      x = with_sharding_constraint(x, P('x'))
      y = with_sharding_constraint(y, P('x'))
      out = x + y
      return jnp.pad(out, [[0, 1]])

    shape = (4,)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertAllClose(actual[:3], expected[:3], check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(np.asarray(actual.device_buffers[0])[:3], expected[:3],
                        check_dtypes=False)

  def testBasic1DWithMeshContextManager(self):
    @partial(pjit,
             in_axis_resources=(P('x'), P('x')),
             out_axis_resources=None)
    def f(x, y):
      return x + y

    shape = (8, 8)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    with jtu.create_global_mesh((2,), ('x')) as mesh:
      actual = f(x, x + 1)
    expected = x + (x + 1)
    self.assertEqual(mesh, jtu.create_global_mesh((2,), ('x')))
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), expected,
                        check_dtypes=False)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testBasic2D(self):
    @partial(pjit,
             in_axis_resources=(P(None, 'x', 'y'), P('y')),
             out_axis_resources=P('x'))
    def f(x, y):
      return x @ y

    x_shape = (8, 6, 4)
    y_shape = (4, 2)
    x = jnp.arange(np.prod(x_shape)).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape)).reshape(y_shape)
    actual = f(x, y)
    expected = x @ y
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 4)

    split0, split1 = np.split(expected, 2)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), split0,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[1]), split0,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[2]), split1,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[3]), split1,
                        check_dtypes=False)

  def testBasic2DWithMeshContextManager(self):
    @partial(pjit,
             in_axis_resources=(P(None, 'x', 'y'), P('y')),
             out_axis_resources=P('x'))
    def f(x, y):
      return x @ y

    x_shape = (8, 6, 4)
    y_shape = (4, 2)
    x = jnp.arange(np.prod(x_shape)).reshape(x_shape)
    y = jnp.arange(np.prod(y_shape)).reshape(y_shape)
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    with mesh:
      actual = f(x, y)
    expected = x @ y
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 4)

    split0, split1 = np.split(expected, 2)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), split0,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[1]), split0,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[2]), split1,
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[3]), split1,
                        check_dtypes=False)

  def testDifferentNestedMesh(self):
    with jtu.create_global_mesh((2, 1), ("x", "y")) as m1:
      with jtu.create_global_mesh((2, 2), ("a", "b")) as m2:
        self.assertEqual(pxla.thread_resources.env.physical_mesh, m2)
      self.assertEqual(pxla.thread_resources.env.physical_mesh, m1)
    self.assertEqual(pxla.thread_resources.env.physical_mesh,
                     pxla.EMPTY_ENV.physical_mesh)

  def testSameNestedMesh(self):
    mesh = jtu.create_global_mesh((2, 1), ("a", "b"))
    with mesh as m1:
      with mesh as m2:
        self.assertEqual(pxla.thread_resources.env.physical_mesh, m2)
      self.assertEqual(pxla.thread_resources.env.physical_mesh, m1)
    self.assertEqual(pxla.thread_resources.env.physical_mesh,
                     pxla.EMPTY_ENV.physical_mesh)

  def testMeshDecorator(self):
    x = jnp.arange(8)
    mesh_shape = (2, 2)
    size = prod(mesh_shape)
    if len(jax.devices()) < size:
      raise unittest.SkipTest(f"Test requires {size} global devices.")
    mesh_devices = np.array(jax.devices()[:size]).reshape(mesh_shape)

    @maps.Mesh(mesh_devices, ('x', 'y'))
    def dec():
      return pjit(lambda x: x, in_axis_resources=P('x'), out_axis_resources=None)(x)
    out = dec()
    self.assertArraysEqual(out, x)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testTwoMeshAxisSharding(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    actual = f(x, x + 1)
    expected = x @ (x + 1)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 4)

    splits = np.split(expected, 4)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), splits[0],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[1]), splits[1],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[2]), splits[2],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[3]), splits[3],
                        check_dtypes=False)

  @jtu.with_mesh([('x', 2)])
  def testBufferDonation(self):
    if jax.default_backend() not in {'gpu', 'tpu'}:
      raise unittest.SkipTest('Buffer donation only supported on GPU and TPU')

    @partial(pjit,
             in_axis_resources=P('x'),
             out_axis_resources=P('x'),
             donate_argnums=0)
    def f(x, y):
      return x + y

    shard = pjit(lambda x: x, in_axis_resources=P('x'),
                 out_axis_resources=P('x'))
    x = shard(jnp.ones((2, 5)) * 4)
    y = shard(jnp.ones((2, 5)) * 2)
    expected = x + y
    self.assertAllClose(f(x, y), expected)
    self.assertNotDeleted(y)
    self.assertDeleted(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testShardingConstraint(self):
    @partial(pjit, in_axis_resources=None, out_axis_resources=None)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, P('x', 'y'))
      return y * 2

    shape = (8, 8)
    x = np.arange(prod(shape)).reshape(shape)
    expected = (x + 1) * 2
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, pxla.ShardedDeviceArray)
    self.assertLen(actual.device_buffers, 2)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), expected,
                        check_dtypes=False)

    hlo = f.lower(np.ones(shape)).compiler_ir(dialect="hlo")
    # Annotation from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @jax_array(True)
  def testShardingConstraintWithArray(self):
    mesh = jtu.create_global_mesh((2, 1), ('x', 'y'))
    s = MeshPspecSharding(mesh, P(None))

    @partial(pjit, in_axis_resources=s, out_axis_resources=s)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, MeshPspecSharding(mesh, P('x', 'y')))
      return y * 2

    shape = (8, 8)
    x = np.arange(prod(shape)).reshape(shape)
    expected = (x + 1) * 2
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, array.Array)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(np.asarray(actual._arrays[0]), expected,
                        check_dtypes=False)

    hlo = f.lower(np.ones(shape)).compiler_ir(dialect="hlo")
    # Annotation from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @jax_array(True)
  def testShardingConstraintWithArrayOpSharding(self):
    shape = (8, 8)
    mesh = jtu.create_global_mesh((2, 1), ('x', 'y'))
    s = MeshPspecSharding(mesh, P(None))
    ops = pjit_lib.to_op_sharding_sharding(
        MeshPspecSharding(mesh, P('x', 'y')), len(shape))

    @partial(pjit, in_axis_resources=s, out_axis_resources=s)
    def f(x):
      y = x + 1
      y = with_sharding_constraint(y, ops)
      return y * 2

    x = np.arange(prod(shape)).reshape(shape)
    expected = (x + 1) * 2
    actual = f(x)
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertIsInstance(actual, array.Array)
    self.assertLen(actual.addressable_shards, 2)
    self.assertAllClose(np.asarray(actual._arrays[0]), expected,
                        check_dtypes=False)

    hlo = f.lower(np.ones(shape)).compiler_ir(dialect="hlo")
    # Annotation from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testShardingConstraintPyTree(self):
    @partial(pjit, in_axis_resources=None, out_axis_resources=None)
    def f(x):
      x = with_sharding_constraint(x, [P('x', 'y'), P('y', 'x')])
      x = x.copy()
      x[0]["a"] *= 2
      return x

    shape = (8, 8)
    v = np.arange(prod(shape)).reshape(shape)
    x = [{"a": v, "b": v * 2}, v * 3]
    actual = f(x)

    expected = x.copy()
    expected[0]["a"] *= 2
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual[0]["a"].device_buffers, 2)

    hlo = f.lower(x).compiler_ir(dialect="hlo")
    # Annotations from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    self.assertIn("sharding={devices=[1,2]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @jax_array(True)
  def testShardingConstraintPyTreeWithArray(self):
    mesh = jtu.create_global_mesh((2, 1), ('x', 'y'))
    s = MeshPspecSharding(mesh, P(None))

    @partial(pjit, in_axis_resources=s, out_axis_resources=s)
    def f(x):
      x = with_sharding_constraint(x, [
          MeshPspecSharding(mesh, P('x', 'y')),
          MeshPspecSharding(mesh, P('y', 'x'))
      ])
      x = x.copy()
      x[0]["a"] *= 2
      return x

    shape = (8, 8)
    v = np.arange(prod(shape)).reshape(shape)
    x = [{"a": v, "b": v * 2}, v * 3]
    actual = f(x)

    expected = x.copy()
    expected[0]["a"] *= 2
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual[0]["a"].addressable_shards, 2)

    hlo = f.lower(x).compiler_ir(dialect="hlo")
    # Annotations from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    self.assertIn("sharding={devices=[1,2]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testShardingConstraintPyTreeWithUnconstrainedDims(self):

    @partial(pjit, in_axis_resources=None, out_axis_resources=None)
    def f(x):
      x = with_sharding_constraint(
          x, [P(P.UNCONSTRAINED, 'y', None),
              P('x', P.UNCONSTRAINED, None)])
      x = x.copy()
      x[0]['a'] *= 2
      return x

    shape = (2, 8, 8)
    v = np.arange(prod(shape)).reshape(shape)
    x = [{'a': v, 'b': v * 2}, v * 3]
    actual = f(x)

    expected = x.copy()
    expected[0]['a'] *= 2
    self.assertAllClose(actual, expected, check_dtypes=False)
    self.assertLen(actual[0]['a'].device_buffers, 4)

    mhlo_str = str(f.lower(x).compiler_ir(dialect="mhlo"))
    self.assertIn("unspecified_dims=[0]", mhlo_str)
    self.assertIn("unspecified_dims=[1]", mhlo_str)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testShardingConstraintPyTreeVmapWithUnconstrainedDims(self):

    @partial(pjit, in_axis_resources=None, out_axis_resources=None)
    def f(x):
      x = jax.vmap(lambda x: with_sharding_constraint(
          x, [P(P.UNCONSTRAINED, 'y'),
              P('x', P.UNCONSTRAINED)]))(x)
      x = x.copy()
      x[0]['a'] *= 2
      return x

    shape = (2, 8, 8)
    v = np.arange(prod(shape)).reshape(shape)
    x = [{'a': v, 'b': v * 2}, v * 3]

    mhlo_str = str(f.lower(x).compiler_ir(dialect="mhlo"))
    self.assertIn("unspecified_dims=[0,1]", mhlo_str)
    self.assertIn("unspecified_dims=[0,2]", mhlo_str)

  def testCaching(self):
    def f(x):
      assert should_be_tracing
      return jnp.sin(x) * 2

    x = np.arange(16).reshape(4, 4)
    devices = np.array(list(jax.local_devices())[:4])
    if devices.size < 4:
      raise unittest.SkipTest("Test requires 4 devices")
    devices = devices.reshape((2, 2))
    with maps.Mesh(devices, ('x', 'y')):
      should_be_tracing = True
      pjit(f, in_axis_resources=P(('x', 'y')), out_axis_resources=None)(x)
      should_be_tracing = False
      pjit(f, in_axis_resources=P(('x', 'y')), out_axis_resources=None)(x)
    # Re-create the mesh to make sure that has no influence on caching
    with maps.Mesh(devices, ('x', 'y')):
      should_be_tracing = False
      pjit(f, in_axis_resources=P(('x', 'y')), out_axis_resources=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testNested(self):
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    f = pjit(lambda x: x.sum() + h.sum(), in_axis_resources=P('x', 'y'), out_axis_resources=None)
    g = pjit(lambda x: f(jnp.sin(x)), in_axis_resources=P('x', None), out_axis_resources=None)
    x = jnp.arange(16.).reshape((4, 4))
    y = g(x)
    self.assertAllClose(y, jnp.sin(x).sum() + h.sum())
    self.assertTrue(hasattr(y, "sharding_spec"))

  @check_1d_2d_mesh(set_mesh=True)
  def testAutodiff(self, mesh, resources):
    if len(mesh) != 2: return
    assert resources == ('x', 'y')
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    f = pjit(lambda x: x.sum(1) * h.sum(),
             in_axis_resources=P('x', 'y'), out_axis_resources=P(('x', 'y')))
    g = pjit(lambda x: f(jnp.sin(x * 4 + 2)),
             in_axis_resources=P('x', None), out_axis_resources=P(('x', 'y')))
    jtu.check_grads(g, (jnp.arange(16.).reshape((4, 4)) / 100,), order=2)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testEvalJaxpr(self):
    x, y = jnp.arange(4.), jnp.arange(5.)
    f = pjit(lambda x, y: x.sum() + jnp.sin(y),
             in_axis_resources=(P('x'), P('y')),
             out_axis_resources=P('y'))
    f_jaxpr = jax.make_jaxpr(f)(x, y)
    f_eval = jax.core.jaxpr_as_fun(f_jaxpr)
    r, = f_eval(x, y)
    self.assertAllClose(r, x.sum() + jnp.sin(y))

  @jtu.with_mesh([('x', 2)])
  def testNonArrayArg(self):
    self.assertEqual(pjit(lambda x: x + 2,
                          in_axis_resources=None,
                          out_axis_resources=None)(1), 3)

  @jtu.with_mesh([('x', 2)])
  def testNonHashableAxisResources(self):
    x = jnp.arange(4)
    y = pjit(lambda x: {'b': x['a'] + 2},
             in_axis_resources=({'a': P('x')},),
             out_axis_resources={'b': P('x')})({'a': x})
    self.assertAllClose(y, {'b': x + 2})

  @jtu.with_mesh([('x', 2)])
  def testGradOfConstraint(self):
    # Make sure that we can compute grads through sharding constraints
    h = lambda x: jnp.sin(with_sharding_constraint(x, P('x'))).sum()
    f = pjit(lambda x: jax.grad(h)(x),
             in_axis_resources=None, out_axis_resources=None)
    x = jnp.arange(8, dtype=jnp.float32)
    self.assertAllClose(f(x), jnp.cos(x))

  @jtu.with_mesh([('x', 2)])
  def testNoopPartitionSpecs(self):
    noops = [P(), P(None), P(()), P((), None), P(None, None, ())]
    x = jnp.arange(8).reshape((2, 2, 2))
    for spec in noops:
      y = pjit(lambda x: x * 2, in_axis_resources=spec, out_axis_resources=spec)(x)
      self.assertAllClose(y, x * 2)

  @jtu.with_mesh([('x', 2)])
  def testVMap(self):
    f = pjit(lambda x, y: (x + y, x), in_axis_resources=P('x'), out_axis_resources=P('x'))
    x = jnp.arange(4)
    y = jnp.arange(5*4).reshape((5, 4))
    z, w = jax.vmap(f, in_axes=(None, 0), out_axes=(0, None))(x, y)
    self.assertAllClose(z, x[jnp.newaxis] + y)
    self.assertAllClose(w, x)
    self.assertEqual(z.sharding_spec.sharding, (pxla.NoSharding(), pxla.Chunked([2])))
    self.assertEqual(w.sharding_spec.sharding, (pxla.Chunked([2]),))

  @jtu.with_mesh([('x', 2)])
  def testVMapShardingConstraint(self):
    f = pjit(lambda x: with_sharding_constraint(x, P('x')),
             in_axis_resources=P(), out_axis_resources=P('x'))
    x = jnp.arange(5*4).reshape((5, 4))
    jaxpr = jax.make_jaxpr(jax.vmap(f))(x)
    pjit_eqn, = jaxpr.eqns
    constraint_eqn, = pjit_eqn.params['jaxpr'].eqns
    op = constraint_eqn.params['sharding']._op_sharding
    self.assertEqual(op.type, xc.OpSharding.Type.OTHER)
    self.assertListEqual(op.tile_assignment_dimensions, [1, 2])
    self.assertListEqual(op.tile_assignment_devices, [0, 1])
    self.assertFalse(pxla.is_op_sharding_replicated(op))

  @jtu.with_mesh([('x', 2)])
  def testVMapShardingConstraintWithSpmdAxis(self):
    f = pjit(
        jax.vmap(
            lambda x: with_sharding_constraint(x, P(None)),
            spmd_axis_name='x',
        ),
        in_axis_resources=P('x'),
        out_axis_resources=P('x'))
    x = jnp.arange(16 * 4).reshape((16, 4))
    jaxpr = jax.make_jaxpr(f)(x)
    pjit_eqn, = jaxpr.eqns
    constraint_eqn, = pjit_eqn.params['jaxpr'].eqns
    op = constraint_eqn.params['sharding']._op_sharding
    self.assertEqual(op.type, xc.OpSharding.Type.OTHER)
    self.assertListEqual(op.tile_assignment_dimensions, [2, 1])
    self.assertListEqual(op.tile_assignment_devices, [0, 1])
    self.assertFalse(pxla.is_op_sharding_replicated(op))

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testShardingInXMap(self):
    h = pjit(lambda x: x, in_axis_resources=P('x'), out_axis_resources=None)
    f = xmap(lambda x: h(x * 2), in_axes=['i', ...], out_axes=['i', ...],
             axis_resources={'i': 'y'})
    x = jnp.arange(16).reshape((4, 4))
    rule = mlir._lowerings[pjit_p]
    test_rule_called = False
    def _test_rule(*args, **kwargs):
      nonlocal test_rule_called
      test_rule_called = True
      in_shardings = kwargs['in_shardings']
      self.assertLen(in_shardings, 1)
      self.assertListEqual(in_shardings[0]._op_sharding.tile_assignment_dimensions,
                           [1, 1, 2])
      self.assertFalse(pxla.is_op_sharding_replicated(in_shardings[0]._op_sharding))

      return rule(*args, **kwargs)
    try:
      mlir._lowerings[pjit_p] = _test_rule
      f(x)
      self.assertTrue(test_rule_called)
    finally:
      mlir._lowerings[pjit_p] = rule

  @jtu.with_mesh([('x', 2)])
  def testLowerWithDuckTyping(self):
    x = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    # Make sure this doesn't crash
    pjit(lambda x: x + 4,
         in_axis_resources=P('x'), out_axis_resources=P('x')).lower(x)

  @jtu.with_mesh([('x', 2)])
  def testLowerDonateArgnumsAvailable(self):
    x = jax.ShapeDtypeStruct((2, 2), jnp.float32)
    def f(*args):
      x, *_ = args
      return x
    f_low = pjit(f, donate_argnums=(0,),
                 in_axis_resources=P('x'), out_axis_resources=P('x')).lower(x)
    f_com = f_low.compile()
    f_low.donate_argnums == f_com.donate_argnums == (0,)

  def testInfeed(self):
    devices = np.array(jax.local_devices())
    nr_devices = len(devices)
    shape = (nr_devices * 3, nr_devices * 5)

    def f_for_jit(x):
      token = lax.create_token(x)
      (y,), token = lax.infeed(
          token, shape=(jax.ShapedArray(x.shape, np.float32),))
      (z,), token = lax.infeed(
          token, shape=(jax.ShapedArray(x.shape, np.float32),))
      (w,), token = lax.infeed(
          token, shape=(jax.ShapedArray(x.shape, np.float32),))

      return x + y + z + w

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = x * 2.
    z = x * 3.
    w = x * 4.

    # Transfer data to infeed before executing the function. For GPUs, the
    # execution of the compiled function is blocking, so transferring data
    # to infeed before executing ensures that the execution does not deadlock
    # waiting for the infeed data.
    logging.info('Transfering to infeed for the jit call')
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
          shape=(jax.ShapedArray(x.shape, np.float32),),
          partitions=(None,))
      # An infeed sharded on first axis
      (z,), token = lax.infeed(
          token,
          shape=(jax.ShapedArray(x.shape, np.float32),),
          partitions=(P(nr_devices, 1),))
      # An infeed sharded on second axis
      (w,), token = lax.infeed(
          token,
          shape=(jax.ShapedArray(x.shape, np.float32),),
          partitions=(P(1, nr_devices),))
      return x + y + z + w

    logging.info('Transfering to infeed for the pjit call')
    for didx, d in enumerate(devices):
      # Transfer the whole array to all devices for replicated.
      d.transfer_to_infeed((y,))
      # For sharded infeed, transfer only the needed slices to each device.
      d.transfer_to_infeed(z[3 * didx:3 * didx + 3, :])
      d.transfer_to_infeed((w[:, 5 * didx:5 * didx + 5],))

    with maps.Mesh(devices, ['d']):
      logging.info('Making pjit call')
      res = pjit(
          f_for_pjit, in_axis_resources=(P('d'),), out_axis_resources=P('d'))(
              x)

    self.assertAllClose(res0, res, check_dtypes=True)

  def testOutfeed(self):
    devices = np.array(jax.local_devices())
    nr_devices = len(devices)
    shape = (nr_devices * 3, nr_devices * 5)

    def f(x):
      token = lax.create_token(x)
      token = lax.outfeed(token, x, partitions=(None,))
      token = lax.outfeed(token, x, partitions=(P(nr_devices, 1),))
      token = lax.outfeed(token, x, partitions=(P(1, nr_devices),))
      return x

    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    def dispatch():
      with maps.Mesh(devices, ['d']):
        logging.info('Making pjit call')
        pjit(f, in_axis_resources=(P('d'),), out_axis_resources=P('d'))(x)
    execution = threading.Thread(target=dispatch)
    execution.start()

    def check_outfeed(d, x):
      y, = d.transfer_from_outfeed(
          xc.shape_from_pyval((x,)).with_major_to_minor_layout_if_absent())
      self.assertAllClose(x, y, check_dtypes=True)

    logging.info('Transfering from outfeed for the pjit call')
    for didx, d in enumerate(devices):
      # Transfer the whole array from all devices for replicated.
      check_outfeed(d, x)
      # For sharded outfeed, the results are sliced.
      check_outfeed(d, x[3 * didx:3 * didx + 3, :])
      check_outfeed(d, x[:, 5 * didx:5 * didx + 5])

    execution.join()

  @jtu.with_mesh([('x', 2)])
  def testWithCustomPRNGKey(self):
    if not config.jax_enable_custom_prng:
      raise unittest.SkipTest("test requires jax_enable_custom_prng")
    key = jax.prng.seed_with_impl(jax.prng.rbg_prng_impl, 87)
    # Make sure this doesn't crash
    pjit(lambda x: x, in_axis_resources=(None), out_axis_resources=(None))(key)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompile(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    expected = x @ (x + 1)

    lowered = f.lower(x, x + 1)
    compiled = lowered.compile()
    actual = compiled(x, x + 1)

    self.assertEqual(lowered.in_avals, compiled.in_avals)
    self.assertEqual(
        lowered.in_avals,
        ((jax.ShapedArray(x.shape, x.dtype, weak_type=False),) * 2, {}))

    splits = np.split(expected, 4)
    self.assertAllClose(np.asarray(actual.device_buffers[0]), splits[0],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[1]), splits[1],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[2]), splits[2],
                        check_dtypes=False)
    self.assertAllClose(np.asarray(actual.device_buffers[3]), splits[3],
                        check_dtypes=False)

    for obj in [lowered, compiled]:
      self.assertTrue(obj._no_kwargs, True)
      self.assertEqual(obj.in_tree, jax.tree_util.tree_flatten(((0, 0), {}))[1])

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileWithKwargs(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y, **kwargs):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    exe = f.lower(x, x + 1).compile()

    self.assertRaisesRegex(
        NotImplementedError,
        "function was compiled by a transformation that does not support "
        "keyword arguments, but called with keyword arguments: a, b",
        lambda: exe(x, x + 1, a=1, b=2))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileInTreeMismatch(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    exe = f.lower(x, x + 1).compile()

    self.assertRaisesRegex(
        TypeError, "function compiled for .*, called with .*",
        lambda: exe([x], [x + 1]))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileArgTypeMismatch(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    exe = f.lower(x_f32, x_f32).compile()
    self.assertRaisesRegex(
        TypeError,
        "Computation compiled for input types:\n.*float32.*\n"
        "called with:\n.*int32.*",
        lambda: exe(x_i32, x_i32))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerAsText(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1)
    self.assertIsInstance(f.as_text(), str)
    self.assertIsInstance(f.as_text(dialect='hlo'), str)
    self.assertIsInstance(f.as_text(dialect='mhlo'), str)

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompilerIR(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1)
    self.assertIsNotNone(f.compiler_ir())
    self.assertIsNotNone(f.compiler_ir(dialect='hlo'))
    self.assertIsNotNone(f.compiler_ir(dialect='mhlo'))

  @jtu.ignore_warning(category=DeprecationWarning)
  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileCompilerIR(self):
    # TODO(frostig): remove (deprecated)
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    self.assertIsNotNone(f.compiler_ir())

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileAsText(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    self.assertIsInstance(f.as_text(), (str, type(None)))

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileCostAnalysis(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    f.cost_analysis()  # doesn't raise

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileMemoryAnalysis(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)
    f = f.lower(x, x + 1).compile()
    f.memory_analysis()  # doesn't raise

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def testLowerCompileExecutable(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    x = jnp.arange(np.prod(shape)).reshape(shape)

    f = f.lower(x, x + 1).compile()
    self.assertIsNotNone(f.runtime_executable())

  @jtu.with_mesh([('x', 2)])
  def test_static_argnums(self):
    @partial(pjit, in_axis_resources=None, out_axis_resources=None,
             static_argnums=(1,))
    def f(x, y):
      return x + (3 if y == 'hi' else 4)

    self.assertEqual(f(1, 'hi' ), 4)
    self.assertEqual(f(1, 'bye'), 5)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def testLowerCompileWithAvals(self):
    @partial(pjit,
             in_axis_resources=P(('x', 'y'),),
             out_axis_resources=P(('x', 'y'),))
    def f(x, y):
      return x @ y

    shape = (8, 8)
    aval = jax.ShapedArray(shape, dtypes.canonicalize_dtype(jnp.int64))
    x = jnp.arange(np.prod(shape)).reshape(shape)
    exe = f.lower(aval, x, _global_avals=True).compile()
    self.assertIsInstance(exe, stages.Compiled)
    self.assertArraysEqual(exe(x, x), x @ x)

  def test_local_sharded_key_array_sda(self):
    input_shape = (8, 4)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    seeds = jnp.arange(
        prod(input_shape), dtype=np.uint32).reshape(input_shape)

    with mesh:
      def make_keys(seeds):
        make_key = partial(prng.seed_with_impl, prng.threefry_prng_impl)
        return make_key(seeds)

      f = pjit(make_keys, in_axis_resources=P(None), out_axis_resources=P(None))

      out = f(seeds)
      self.assertIsInstance(out, jax.random.KeyArray)
      self.assertEqual(out.shape, input_shape)
      out.unsafe_raw_array()  # doesn't crash


class GDAPjitTest(jtu.JaxTestCase):

  def test_pjit_gda_single_output(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return input_data[index]

    gda_obj = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)

    with parallel_functions_output_gda(True):
      with global_mesh:
        @partial(pjit, in_axis_resources=FROM_GDA, out_axis_resources=P('x', 'y'))
        def f(x):
          return x @ x.T
        expected_matrix_mul = input_data @ input_data.T

        out = f(gda_obj)
        self.assertIsInstance(out, global_device_array.GlobalDeviceArray)
        self.assertEqual(out.shape, (8, 8))
        self.assertEqual(out.local_shards[0].data.shape, (2, 4))
        self.assertDictEqual(out.mesh.shape, {'x': 4, 'y': 2})
        for s in out.local_shards:
          self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

        out2 = f(out)
        self.assertIsInstance(out2, global_device_array.GlobalDeviceArray)

        with self.assertRaisesRegex(
            ValueError, ('For a non-GDA input, the corresponding resource in '
                         'in_axis_resources cannot be `pjit.FROM_GDA`.')):
          f(input_data)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_pjit_gda_multi_input_multi_output(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return input_data[index]

    mesh_axes1 = P('x', 'y')
    gda1 = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes1, cb)
    mesh_axes2 = P('x')
    gda2 = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes2, cb)
    mesh_axes3 = P(('x', 'y'))
    gda3 = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes3, cb)
    mesh_axes4 = P(None)
    gda4 = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes4, cb)

    with parallel_functions_output_gda(True):
      @partial(
          pjit,
          # `FROM_GDA` will be replicated for all the inputs.
          in_axis_resources=FROM_GDA,
          out_axis_resources=(mesh_axes1, mesh_axes4, mesh_axes2, mesh_axes3))
      def f(x, y, z, a):
        return x @ x.T, y, z, a
      out1, out2, out3, out4 = f(gda1, gda2, gda3, gda4)

      self.assertIsInstance(out1, global_device_array.GlobalDeviceArray)
      self.assertEqual(out1.shape, (8, 8))
      self.assertEqual(out1.local_shards[0].data.shape, (2, 4))
      self.assertEqual(out1.local_shards[0].index, (slice(0, 2), slice(0, 4)))
      self.assertEqual(out1.local_shards[1].index, (slice(0, 2), slice(4, 8)))
      self.assertListEqual([s.replica_id for s in out1.local_shards],
                           [0, 0, 0, 0, 0, 0, 0, 0])
      expected_matrix_mul = input_data @ input_data.T
      for s in out1.local_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

      self.assertIsInstance(out2, global_device_array.GlobalDeviceArray)
      self.assertEqual(out2.shape, (8, 2))
      self.assertEqual(out2.local_shards[0].data.shape, (8, 2))
      self.assertEqual(out2.local_shards[0].index, (slice(None), slice(None)))
      self.assertEqual(out2.local_shards[1].index, (slice(None), slice(None)))
      self.assertListEqual([s.replica_id for s in out2.local_shards],
                           [0, 1, 2, 3, 4, 5, 6, 7])
      for s in out2.local_shards:
        self.assertArraysEqual(s.data, input_data)

      self.assertIsInstance(out3, global_device_array.GlobalDeviceArray)
      self.assertEqual(out3.shape, (8, 2))
      self.assertEqual(out3.local_shards[0].data.shape, (2, 2))
      self.assertEqual(out3.local_shards[0].index, (slice(0, 2), slice(None)))
      self.assertEqual(out3.local_shards[1].index, (slice(0, 2), slice(None)))
      self.assertListEqual([s.replica_id for s in out3.local_shards],
                           [0, 1, 0, 1, 0, 1, 0, 1])
      for s in out3.local_shards:
        self.assertArraysEqual(s.data, input_data[s.index])

      self.assertIsInstance(out4, global_device_array.GlobalDeviceArray)
      self.assertEqual(out4.shape, (8, 2))
      self.assertEqual(out4.local_shards[0].data.shape, (1, 2))
      self.assertEqual(out4.local_shards[0].index, (slice(0, 1), slice(None)))
      self.assertEqual(out4.local_shards[1].index, (slice(1, 2), slice(None)))
      self.assertListEqual([s.replica_id for s in out4.local_shards],
                           [0, 0, 0, 0, 0, 0, 0, 0])
      for s in out4.local_shards:
        self.assertArraysEqual(s.data, input_data[s.index])

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_pjit_gda_mixed_inputs(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    input_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)
    def cb(index):
      return input_data[index]

    gda_obj = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)

    with parallel_functions_output_gda(True):
      @partial(pjit,
               in_axis_resources=(FROM_GDA, P('x', 'y')),
               out_axis_resources=(P('x', 'y'), P(('x', 'y'))))
      def f(x, y):
        return x @ x.T, y @ y.T
      expected_matrix_mul = input_data @ input_data.T

      out1, out2 = f(gda_obj, input_data)
      self.assertIsInstance(out1, global_device_array.GlobalDeviceArray)
      self.assertEqual(out1.shape, (8, 8))
      self.assertEqual(out1.local_shards[0].data.shape, (2, 4))
      self.assertDictEqual(out1.mesh.shape, {'x': 4, 'y': 2})
      for s in out1.local_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

      self.assertIsInstance(out2, global_device_array.GlobalDeviceArray)
      self.assertEqual(out2.shape, (8, 8))
      self.assertEqual(out2.local_shards[0].data.shape, (1, 8))
      self.assertDictEqual(out2.mesh.shape, {'x': 4, 'y': 2})
      for s in out2.local_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_pjit_gda_non_gda_inputs(self):
    input_shape = (8, 2)
    input_data = np.arange(prod(input_shape)).reshape(input_shape)

    with parallel_functions_output_gda(True):
      @partial(pjit,
               in_axis_resources=(None, P('x', 'y')),
               out_axis_resources=(P('x', 'y'), P(('x', 'y'))))
      def f(x, y):
        return x @ x.T, y @ y.T

      expected_matrix_mul = input_data @ input_data.T
      out1, out2 = f(input_data, input_data)

      self.assertIsInstance(out1, global_device_array.GlobalDeviceArray)
      self.assertEqual(out1.shape, (8, 8))
      self.assertEqual(out1.local_shards[0].data.shape, (2, 4))
      self.assertDictEqual(out1.mesh.shape, {'x': 4, 'y': 2})
      for s in out1.local_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

      self.assertIsInstance(out2, global_device_array.GlobalDeviceArray)
      self.assertEqual(out2.shape, (8, 8))
      self.assertEqual(out2.local_shards[0].data.shape, (1, 8))
      self.assertDictEqual(out2.mesh.shape, {'x': 4, 'y': 2})
      for s in out2.local_shards:
        self.assertArraysEqual(s.data, expected_matrix_mul[s.index])

  @jtu.with_mesh([('x', 2), ('y', 2)])
  def test_pjit_gda_mesh_mismatch(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    global_input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]

    gda_obj = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)

    with self.assertRaisesRegex(ValueError,
                                "Pjit's mesh and GDA's mesh should be equal."):
      @partial(pjit, in_axis_resources=FROM_GDA, out_axis_resources=P('x', 'y'))
      def f(x):
        return x

      f(gda_obj)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_pjit_gda_wrong_resource_for_gda_input(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x')
    global_input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)
    def cb(index):
      return global_input_data[index]

    gda_obj = global_device_array.GlobalDeviceArray.from_callback(
        global_input_shape, global_mesh, mesh_axes, cb)

    with self.assertRaisesRegex(
        ValueError,
        r"Got an input GDA to pjit with different partitioning than specified "
        r'in the in_axis_resources argument to pjit. The partitioning must match, or '
        r'use `jax.experimental.pjit.FROM_GDA` in `in_axis_resources` for GDA. '
        r"Got GDA sharding.*PartitionSpec\('x',\).*and "
        r"pjit sharding.*PartitionSpec\('x', 'y'\).*"):
      @partial(pjit, in_axis_resources=P('x', 'y'), out_axis_resources=P('x', 'y'))
      def f(x):
        return x

      f(gda_obj)

  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_pjit_gda_caching(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    input_data = np.arange(
        prod(input_shape), dtype=np.float32).reshape(input_shape)
    def cb(index):
      return input_data[index]

    gda_obj = global_device_array.GlobalDeviceArray.from_callback(
        input_shape, global_mesh, mesh_axes, cb)

    @partial(pjit, in_axis_resources=mesh_axes, out_axis_resources=P('x', 'y'))
    def f(x, y):
      return x @ y.T

    before_lower_cache = pjit_lib._pjit_lower_cached.cache_info()

    f(gda_obj, gda_obj)
    after_lower_cache1 = pjit_lib._pjit_lower_cached.cache_info()
    self.assertEqual(before_lower_cache.hits, after_lower_cache1.hits)
    self.assertEqual(before_lower_cache.misses + 1, after_lower_cache1.misses)

    f(gda_obj, gda_obj)
    after_lower_cache2 = pjit_lib._pjit_lower_cached.cache_info()
    self.assertEqual(after_lower_cache1.hits + 1, after_lower_cache2.hits)
    self.assertEqual(after_lower_cache1.misses, after_lower_cache2.misses)

    f(input_data, input_data)
    after_lower_cache3 = pjit_lib._pjit_lower_cached.cache_info()
    self.assertEqual(after_lower_cache2.hits, after_lower_cache3.hits)
    self.assertEqual(after_lower_cache2.misses + 1, after_lower_cache3.misses)

    f(gda_obj, input_data)
    after_lower_cache4 = pjit_lib._pjit_lower_cached.cache_info()
    self.assertEqual(after_lower_cache3.hits, after_lower_cache4.hits)
    self.assertEqual(after_lower_cache3.misses + 1, after_lower_cache4.misses)


  @jtu.with_mesh([('x', 4), ('y', 2)])
  def test_partition_spec_mismatch_semantically_equivalent(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P(None)
    global_input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    def cb(index):
      return global_input_data[index]

    with parallel_functions_output_gda(True):
      gda_obj = global_device_array.GlobalDeviceArray.from_callback(
          global_input_shape, global_mesh, mesh_axes, cb)

      @partial(pjit, in_axis_resources=P(None), out_axis_resources=P(None))
      def f(x):
        return x

      output_gda = f(gda_obj)
      # Ensure output_gda.mesh_axes = P() is matched with P(None).
      self.assertEqual(output_gda.mesh_axes, ())
      # P(None) is in_axis_resources.
      f(output_gda)

  def test_from_gda_duplicates(self):
    global_mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P('x', 'y')
    input_gda, _ = create_gda(global_input_shape, global_mesh, mesh_axes)

    # It's occasionally possible to end up with two FROM_GDA singletons (e.g. if
    # pickling in_axis_resources and sending to other processes). Make sure this
    # this doesn't cause an error to avoid user confusion.
    from_gda_dup = pjit_lib._FromGdaSingleton()
    with maps.Mesh(global_mesh.devices, global_mesh.axis_names):
      pjit(lambda x: x, in_axis_resources=from_gda_dup, out_axis_resources=None)(
          input_gda)

  def test_no_recompilation_due_to_in_axis_resources(self):
    global_mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P(None,)
    input_gda, _ = create_gda(global_input_shape, global_mesh, mesh_axes)

    with parallel_functions_output_gda(True):
      @partial(pjit, in_axis_resources=mesh_axes, out_axis_resources=mesh_axes)
      def f(x):
        return x

      with global_mesh:
        out_gda = f(input_gda)
        self.assertEqual(out_gda.mesh_axes, ())

        before_cache = pjit_lib._pjit_lower_cached.cache_info()
        f(out_gda)
        after_cache = pjit_lib._pjit_lower_cached.cache_info()

        self.assertEqual(before_cache.hits + 1, after_cache.hits)
        self.assertEqual(before_cache.misses, after_cache.misses)

  def test_no_recompilation_due_to_fully_replicated_and_gda_inputs(self):
    global_mesh = jtu.create_global_mesh((1, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    mesh_axes = P(None)
    global_data = np.arange(
        prod(global_input_shape)).reshape(global_input_shape)

    with parallel_functions_output_gda(True):
      f = pjit(lambda x: x, in_axis_resources=mesh_axes,
               out_axis_resources=mesh_axes)

      with global_mesh:
        out_gda = f(global_data)
        self.assertEqual(out_gda.mesh_axes, ())

        before_cache = pjit_lib._pjit_lower_cached.cache_info()
        f(out_gda)
        after_cache = pjit_lib._pjit_lower_cached.cache_info()

        self.assertEqual(before_cache.hits + 1, after_cache.hits)
        self.assertEqual(before_cache.misses, after_cache.misses)

  def test_pjit_gda_aot_sharding_mismatch(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    input_gda, _ = create_gda(global_input_shape, global_mesh, P('x', 'y'))

    with global_mesh:
      f = pjit(lambda x: x, in_axis_resources=P('x'), out_axis_resources=P('x'))
      compiled = f.lower(jax.ShapedArray(global_input_shape, jnp.float32)).compile()
      with self.assertRaisesRegex(
          ValueError, "GDA sharding does not match the input sharding."):
        compiled(input_gda)

  def test_pjit_gda_same_sharding_aot(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)

    g1, _ = create_gda(global_input_shape, global_mesh, P(None,))
    with global_mesh:
      f = pjit(lambda x: x, in_axis_resources=P(None), out_axis_resources=P('x'))
      compiled = f.lower(jax.ShapedArray(global_input_shape, jnp.float32)).compile()
      compiled(g1)  # no error

class AutoShardingPjitTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    ('2d_gda', (4, 2), (4, 2), ('x', 'y'),
     parallel_functions_output_gda, create_gda, global_device_array.GlobalDeviceArray),
    # TODO(b/226977360): Support 3D mesh shape for example (2, 2, 2).
    ('3d_gda', (1, 4, 2), (2, 4, 8, 4), ('x', 'y', 'z'),
     parallel_functions_output_gda, create_gda, global_device_array.GlobalDeviceArray),
    ('1d_gda', (8,), (8, 2), ('x'),
     parallel_functions_output_gda, create_gda, global_device_array.GlobalDeviceArray),
    ('2d_array', (4, 2), (4, 2), ('x', 'y'),
     jax_array, create_array, array.Array),
    # TODO(b/226977360): Support 3D mesh shape for example (2, 2, 2).
    ('3d_array', (1, 4, 2), (2, 4, 8, 4), ('x', 'y', 'z'),
     jax_array, create_array, array.Array),
    ('1d_array', (8,), (8, 2), ('x'), jax_array, create_array, array.Array),
  )
  def test_pjit_arr_auto_sharding(self, mesh_shape, global_input_shape,
                                  mesh_axis_names, ctx, create_fun, arr_type):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('AutoSharding is not supported on stream_executor yet.')
    global_mesh = jtu.create_global_mesh(mesh_shape, mesh_axis_names)
    input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with ctx(True):
      with global_mesh:
        f = pjit(lambda x: x, in_axis_resources=AUTO,
                 out_axis_resources=AUTO)

        inp = jax.ShapedArray(input_data.shape, input_data.dtype)
        compiled = f.lower(inp, _global_avals=True).compile()
        inputs = [create_fun(global_input_shape, global_mesh, ip, input_data)[0]
                  for ip in compiled.input_shardings]
        out = compiled(*inputs)
        self.assertIsInstance(out, arr_type)
        self.assertArraysEqual(out._value, input_data)

  @parameterized.named_parameters(
    ('gda', parallel_functions_output_gda, create_gda, 'GDA'),
    ('array', jax_array, create_array, 'Array'),
  )
  def test_xla_arr_sharding_mismatch(self, ctx, create_fun, arr_type):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('AutoSharding is not supported on stream_executor yet.')
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (4, 2)
    input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with ctx(True):
      with global_mesh:
        f = pjit(lambda x: x, in_axis_resources=AUTO, out_axis_resources=AUTO)
        inp = jax.ShapedArray(input_data.shape, input_data.dtype)
        compiled = f.lower(inp, _global_avals=True).compile()

        different_pspec = (P('y', 'x')
                           if compiled.input_shardings[0].spec == P(('x',), ('y',))
                           else P('x', 'y'))
        arr, _ = create_fun(global_input_shape, global_mesh, different_pspec,
                            input_data)
        with self.assertRaisesRegex(
            ValueError,
            f"{arr_type} sharding does not match the input sharding."):
          compiled(arr)

  def test_gda_auto_shardings_len(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('AutoSharding is not supported on stream_executor yet.')
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    global_input_shape = (4, 2)
    input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with global_mesh:
      f = pjit(lambda x, y, z: (x, y, z), in_axis_resources=AUTO,
               out_axis_resources=AUTO)
      inp = jax.ShapedArray(input_data.shape, input_data.dtype)
      compiled = f.lower(inp, inp, inp, _global_avals=True).compile()
      self.assertLen(compiled.output_shardings, 3)
      self.assertLen(compiled.input_shardings, 3)

  @parameterized.named_parameters(
    ('3d_gda', (1, 1, 2), ('x', 'y', 'z'), P(('x', 'y', 'z')),
     parallel_functions_output_gda, create_gda, global_device_array.GlobalDeviceArray),
    ('2d_gda', (4, 2), ('x', 'y'), P('y', 'x'),
     parallel_functions_output_gda, create_gda, global_device_array.GlobalDeviceArray),
    ('1d_gda', (8,), ('x'), P('x'),
     parallel_functions_output_gda, create_gda, global_device_array.GlobalDeviceArray),
    ('3d_array', (1, 1, 2), ('x', 'y', 'z'), P(('x', 'y', 'z')),
     jax_array, create_array, array.Array),
    ('2d_array', (4, 2), ('x', 'y'), P('y', 'x'),
     jax_array, create_array, array.Array),
    ('1d_array', (8,), ('x'), P('x'),
     jax_array, create_array, array.Array),
  )
  def test_pjit_arr_partial_auto_sharding(self, mesh_shape, mesh_axis_names,
                                          pspec, ctx, create_fun, arr_type):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('AutoSharding is not supported on stream_executor yet.')
    global_mesh = jtu.create_global_mesh(mesh_shape, mesh_axis_names)
    global_input_shape = (8, 4)
    input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    if arr_type is array.Array:
      in_resource = MeshPspecSharding(global_mesh, pspec)
    else:
      in_resource = pspec

    with ctx(True):
      with global_mesh:
        f = pjit(lambda x, y: (x, y), in_axis_resources=(in_resource, AUTO),
                 out_axis_resources=AUTO)

        inp = jax.ShapedArray(input_data.shape, input_data.dtype)
        compiled = f.lower(inp, inp, _global_avals=True).compile()
        inputs = [create_fun(global_input_shape, global_mesh, ip, input_data)[0]
                  for ip in compiled.input_shardings]
        out1, out2 = compiled(*inputs)
        for o in [out1, out2]:
          self.assertIsInstance(o, arr_type)
          self.assertArraysEqual(o._value, input_data)

  @unittest.skip('The error is not raised yet. Enable this back once we raise '
                 'the error in pjit again.')
  def test_pjit_array_error(self):
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('AutoSharding is not supported on stream_executor yet.')

    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    global_input_shape = (8, 2)
    input_data = np.arange(
        prod(global_input_shape), dtype=np.float32).reshape(global_input_shape)

    with jax_array(True):
      with global_mesh:
        f = pjit(lambda x: x, in_axis_resources=AUTO,
                 out_axis_resources=AUTO)

        inp = jax.ShapedArray(input_data.shape, input_data.dtype)
        compiled = f.lower(inp, _global_avals=True).compile()
        inputs = [create_array(global_input_shape, global_mesh, ip, input_data)[0]
                  for ip in compiled.input_shardings]
        with self.assertRaisesRegex(
            ValueError,
            ('Passing sharding on pjit and on args while using the '
             'auto spmd partitioner is not allowed. Please call the '
             'compiled object on the inputs.')):
          f(*inputs)


class ArrayPjitTest(jtu.JaxTestCase):

  @parameterized.named_parameters(
    ('fully_sharded_output', P('x', 'y'), (2, 4)),
    ('fully_replicated_output', P(None), (8, 8)),
  )
  @jax_array(True)
  def test_pjit_array_single_output(self, out_axis_resources, shard_shape):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, input_data = create_array(global_input_shape, global_mesh, mesh_axes)

    f = pjit(lambda x: x @ x.T, out_axis_resources=MeshPspecSharding(
        global_mesh, out_axis_resources))
    expected_matrix_mul = input_data @ input_data.T

    out = f(input_array)
    self.assertIsInstance(out, array.Array)
    self.assertEqual(out.shape, (8, 8))
    self.assertEqual(out.addressable_shards[0].data.shape, shard_shape)
    for s in out.addressable_shards:
      self.assertLen(s.data._arrays, 1)
      self.assertArraysEqual(s.data._arrays[0], expected_matrix_mul[s.index])
    self.assertArraysEqual(out._value, expected_matrix_mul)

  @parameterized.named_parameters(
    ('fully_sharded_output', P('x', 'y'), (2, 4)),
    ('fully_replicated_output', P(None), (8, 8)),
  )
  @jax_array(True)
  def test_pjit_array_single_output_with_mesh_context_manager(
      self, out_axis_resources, shard_shape):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, input_data = create_array(global_input_shape, global_mesh, mesh_axes)

    with global_mesh:
      f = pjit(lambda x: x @ x.T, out_axis_resources=MeshPspecSharding(
          global_mesh, out_axis_resources))
      expected_matrix_mul = input_data @ input_data.T

      out = f(input_array)
      self.assertIsInstance(out, array.Array)
      self.assertEqual(out.shape, (8, 8))
      self.assertEqual(out.addressable_shards[0].data.shape, shard_shape)
      for s in out.addressable_shards:
        self.assertLen(s.data._arrays, 1)
        self.assertArraysEqual(s.data._arrays[0], expected_matrix_mul[s.index])
      self.assertArraysEqual(out._value, expected_matrix_mul)

  def test_numpy_array_input_assume_fully_replicated(self):
    input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_data = np.arange(
        prod(input_shape), dtype=np.float32).reshape(input_shape)
    with jax_array(True):
      with global_mesh:
        f = pjit(lambda x: x,
                 out_axis_resources=MeshPspecSharding(
                     global_mesh, P('x', 'y')))
        # Since no in_axis_resources is provided, pjit will assume that
        # the numpy input is fully replicated over the mesh.
        out = f(input_data)
        self.assertIsInstance(out, array.Array)
        for s in out.addressable_shards:
          self.assertEqual(s.data.shape, (2, 1))
          self.assertArraysEqual(s.data._arrays[0], input_data[s.index])
        self.assertArraysEqual(out._value, input_data)

  def test_numpy_array_input(self):
    input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_data = np.arange(
        prod(input_shape), dtype=np.float32).reshape(input_shape)
    with jax_array(True):
      with global_mesh:
        f = pjit(lambda x: x,
                 in_axis_resources=MeshPspecSharding(
                     global_mesh, P(None)),
                 out_axis_resources=MeshPspecSharding(
                     global_mesh, P('x', 'y')))
        out = f(input_data)
        self.assertIsInstance(out, array.Array)
        for s in out.addressable_shards:
          self.assertEqual(s.data.shape, (2, 1))
          self.assertArraysEqual(s.data._arrays[0], input_data[s.index])
        self.assertArraysEqual(out._value, input_data)

  @jax_array(True)
  def test_unspecified_out_axis_resources(self):

    def _checks(out, input_data):
      self.assertIsInstance(out, array.Array)
      self.assertIsInstance(out.sharding, OpShardingSharding)
      self.assertEqual(out.shape, (8, 2))
      self.assertEqual(out.addressable_shards[0].data.shape, (2, 1))
      for s in out.addressable_shards:
        self.assertLen(s.data._arrays, 1)
        self.assertArraysEqual(s.data._arrays[0], input_data[s.index])
      self.assertArraysEqual(out._value, input_data)

    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, input_data = create_array(global_input_shape, global_mesh, mesh_axes)

    f = pjit(lambda x: x)

    out = f(input_array)
    _checks(out, input_data)

    out2 = f(out)
    _checks(out2, input_data)

  @parameterized.named_parameters(
    ('mesh1', (4, 2), (2, 1), (2, 2), (1, 2), (8, 2)),
    ('mesh2', (2, 2), (4, 1), (4, 2), (2, 2), (8, 2)),
    ('mesh3', (2, 1), (4, 2), (4, 2), (4, 2), (8, 2)),
  )
  @jax_array(True)
  def test_pjit_array_multi_input_multi_output(self, mesh_shape, s1_shape,
                                               s2_shape, s3_shape, s4_shape):
    # Disable on SE runtime type because XLA sharding propagation is not
    # supported.
    if xla_bridge.get_backend().runtime_type == 'se':
      raise unittest.SkipTest('Needs TFRT runtime.')
    global_mesh = jtu.create_global_mesh(mesh_shape, ('x', 'y'))
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
    out_tree = f((a1, (a2, (a3, a4))))
    (out1, out2, out3, out4), _ = jax.tree_util.tree_flatten(out_tree)

    self.assertIsInstance(out1, array.Array)
    self.assertEqual(out1.shape, (8, 2))
    self.assertEqual(out1.addressable_shards[0].data.shape, s1_shape)
    for s in out1.addressable_shards:
      self.assertArraysEqual(s.data._arrays[0], input_data[s.index])

    self.assertIsInstance(out2, array.Array)
    self.assertEqual(out2.shape, (8, 2))
    self.assertEqual(out2.addressable_shards[0].data.shape, s2_shape)
    for s in out2.addressable_shards:
      self.assertArraysEqual(s.data._arrays[0], input_data[s.index])

    self.assertIsInstance(out3, array.Array)
    self.assertEqual(out3.shape, (8, 2))
    self.assertEqual(out3.addressable_shards[0].data.shape, s3_shape)
    for s in out3.addressable_shards:
      self.assertArraysEqual(s.data._arrays[0], input_data[s.index])

    self.assertIsInstance(out4, array.Array)
    self.assertEqual(out4.shape, (8, 2))
    self.assertEqual(out4.addressable_shards[0].data.shape, s4_shape)
    for s in out4.addressable_shards:
      self.assertArraysEqual(s.data._arrays[0], input_data)

  def test_in_axis_resources_mismatch_error(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, _ = create_array(global_input_shape, global_mesh, mesh_axes)

    with jax_array(True):
      with global_mesh:
        f = pjit(lambda x: x,
                 in_axis_resources=MeshPspecSharding(global_mesh, P('x')))
        with self.assertRaisesRegex(
            ValueError,
            ('Sharding passed to pjit does not match the sharding on the '
             'respective arg')):
          f(input_array)

  def test_in_axis_resources_same_as_array_sharding(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, _ = create_array(global_input_shape, global_mesh, mesh_axes)

    with jax_array(True):
      with global_mesh:
        out = pjit(
            lambda x: x,
            in_axis_resources=MeshPspecSharding(global_mesh, P('x' ,'y')))(input_array)
        self.assertIsInstance(out, array.Array)

  def test_in_axis_resources_error(self):
    mesh = jtu.create_global_mesh((2,), ('x'))
    with jax_array(True):
      with self.assertRaisesRegex(
            ValueError,
            ('When `config.jax_array` flag is enabled, '
             'in_axis_resources should contain instances of `Sharding` '
             'or `pjit.AUTO`.')):
        pjit(lambda x: x,
             in_axis_resources=(MeshPspecSharding(mesh, P('x')),
                                pjit_lib._UNSPECIFIED))

  def test_out_axis_resources_error(self):
    with jax_array(True):
      with self.assertRaisesRegex(
            ValueError,
            ('When `config.jax_array` flag is enabled, '
             'out_axis_resources should contain instances of `Sharding` '
             'or `pjit.AUTO`.')):
        pjit(lambda x: x, out_axis_resources=P('x'))

  def test_no_input_output(self):
    with jax_array(True):
      def f():
        pass
      pjit(f)

  def test_array_device_assignment_mismatch_with_mesh(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mesh_axes = P('x', 'y')

    input_array, _ = create_array(
        global_input_shape, jtu.create_global_mesh((2, 2), ('x', 'y')),
        mesh_axes)

    with jax_array(True):
      with global_mesh:
        with self.assertRaisesRegex(
            ValueError, "Pjit's devices and Array's devices should be equal"):
          pjit(lambda x: x)(input_array)

  def test_array_lower_compile(self):
    global_input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))

    a1, input_data = create_array(global_input_shape, global_mesh, P('x', 'y'))
    a2, _ = create_array(global_input_shape, global_mesh, P('x'))

    aval = jax.ShapedArray(global_input_shape, np.float32)

    with jax_array(True):
      with global_mesh:
        f = pjit(
            lambda x, y: x @ y.T,
            in_axis_resources=MeshPspecSharding(global_mesh, P('x' ,'y')))
        compiled = f.lower(aval, aval).compile()
        out = compiled(a1, a1)
        self.assertIsInstance(out, array.Array)
        self.assertArraysEqual(out._value, input_data @ input_data.T)

        with self.assertRaisesRegex(
            ValueError, 'Array sharding does not match the input sharding'):
          compiled(a2, a2)

  @jax_array(True)
  def test_globally_sharded_key_array_result_8x4_single_device(self):
    input_shape = (8, 4)
    seeds = jnp.arange(
        prod(input_shape), dtype=np.uint32).reshape(input_shape)

    @pjit
    def make_keys(seeds):
      make_key = partial(prng.seed_with_impl, prng.threefry_prng_impl)
      return make_key(seeds)

    out = make_keys(seeds)
    self.assertIsInstance(out, jax.random.KeyArray)
    self.assertEqual(out.shape, input_shape)
    out.unsafe_raw_array()  # doesn't crash

  # TODO(yashkatariya,frostig): re-enable together with implementation in the
  # global result handler in `jax._src.prng.KeyTy`
  @unittest.skip('XLA output sharding support not yet implemented')
  @jax_array(True)
  def test_globally_sharded_key_array_result_8x4_multi_device(self):
    input_shape = (8, 4)
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    seeds, _ = create_array(input_shape, mesh, spec, dtype=np.uint32)

    # TODO(yashkatariya,frostig): also test pjit with axis resource annotations
    @pjit
    def make_keys(seeds):
      make_key = partial(prng.seed_with_impl, prng.threefry_prng_impl)
      return make_key(seeds)

    # TODO(yashkatariya,frostig): also test shape, dype, maybe a reference
    # value (from execution without pjit)
    out = make_keys(seeds)
    out.unsafe_raw_array()  # doesn't crash

  def test_array_device_assignment_mismatch_out_shardings(self):
    input_shape = (8, 2)
    m1 = jtu.create_global_mesh((4, 2), ('x', 'y'))
    m2 = jtu.create_global_mesh((2, 2), ('x', 'y'))
    spec = P('x', 'y')

    a1, _ = create_array(input_shape, m1, spec)

    with jax_array(True):
      with m1:
        with self.assertRaisesRegex(
            ValueError, "Pjit's devices and Array's devices should be equal"):
          pjit(lambda x, y: (x, y),
               out_axis_resources=(MeshPspecSharding(m1, spec),
                                   MeshPspecSharding(m2, spec)))(a1, a1)

  def test_array_device_assignment_mismatch_in_and_out_shardings(self):
    input_shape = (8, 2)
    m1 = jtu.create_global_mesh((4, 2), ('x', 'y'))
    m2 = jtu.create_global_mesh((2, 2), ('x', 'y'))
    spec = P('x', 'y')

    a1, _ = create_array(input_shape, m2, spec)

    with jax_array(True):
      with m1:
        with self.assertRaisesRegex(
            ValueError, "Pjit's devices and Array's devices should be equal"):
          pjit(lambda x, y: (x, y),
               in_axis_resources=MeshPspecSharding(m2, spec),
               out_axis_resources=MeshPspecSharding(m1, spec))(a1, a1)

  def test_mixed_inputs(self):
    input_shape = (8, 2)
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    spec = P('x', 'y')

    a1, input_data = create_array(input_shape, global_mesh, spec)

    with jax_array(True):
      with global_mesh:
        f = pjit(lambda x, y: (x, y),
                 in_axis_resources=MeshPspecSharding(global_mesh, P(None)))
        with self.assertRaisesRegex(
            ValueError,
            ('Sharding passed to pjit does not match the sharding on the '
             'respective arg')):
          f(input_data, a1)

  def test_pjit_array_same_sharding_aot(self):
    global_mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    input_shape = (8, 2)
    a1, _ = create_array(input_shape, global_mesh, P(None,))
    with jax_array(True):
      with global_mesh:
        f = pjit(lambda x: x, in_axis_resources=MeshPspecSharding(global_mesh, P(None,)))
        compiled = f.lower(jax.ShapedArray(input_shape, jnp.float32)).compile()
        compiled(a1)  # no error

  @jax_array(True)
  def test_pjit_single_device_sharding_add(self):
    a = jnp.array([1, 2, 3], dtype=jnp.float32)
    b = jnp.array([4, 5, 6], dtype=jnp.float32)

    @pjit
    def add(x, y):
      return x + y
    out = add(a, b)
    self.assertIsInstance(out, array.Array)
    self.assertArraysEqual(out, a + b)

    out2 = add(out, out)
    self.assertIsInstance(out2, array.Array)
    self.assertArraysEqual(out2, 2 * (a + b))

  @jax_array(True)
  def test_pjit_single_device_sharding_mul(self):
    a = jnp.arange(16).reshape((8, 2))

    @pjit
    def mul(x):
      return x @ x.T

    out = mul(a)
    self.assertIsInstance(out, array.Array)
    self.assertArraysEqual(out, a @ a.T)

  @jax_array(True)
  def test_pjit_single_device_sharding_cache(self):
    a = jnp.arange(16).reshape((8, 2))
    f = pjit(lambda x: x)

    out = f(a)
    cache_info1 = pjit_lib._pjit_lower_cached.cache_info()

    _ = f(out)
    cache_info2 = pjit_lib._pjit_lower_cached.cache_info()

    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)

  @jax_array(True)
  def test_pjit_different_device_recompilation(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest('Requires 2 or more devices.')

    val1 = jnp.array([1, 2, 3], dtype=jnp.float32)
    a = jax.device_put(val1, jax.devices()[0])

    val2 = jnp.array([4, 5, 6], dtype=jnp.float32)
    b = jax.device_put(val2, jax.devices()[1])

    f = pjit(lambda x: x)

    out1 = f(a)
    cache_info1 = pjit_lib._pjit_lower_cached.cache_info()

    out2 = f(b)
    cache_info2 = pjit_lib._pjit_lower_cached.cache_info()

    self.assertEqual(cache_info2.hits, cache_info1.hits)
    self.assertEqual(cache_info2.misses, cache_info1.misses + 1)
    self.assertArraysEqual(out1, val1)
    self.assertArraysEqual(out2, val2)

  @jax_array(True)
  def test_grad_of_pjit_single_device_sharding(self):
    a = jnp.array(16, dtype=jnp.float32)
    f = lambda x: x
    out = jax.grad(pjit(f))(a)
    self.assertIsInstance(out, array.Array)
    self.assertArraysEqual(out, jax.grad(f)(a))

  @jax_array(True)
  def test_autodiff_with_single_device_sharding(self):
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4.)
    f = pjit(lambda x: x.sum(1) * h.sum())
    g = pjit(lambda x: f(jnp.sin(x * 4 + 2)))
    jtu.check_grads(g, (jnp.arange(16.).reshape((4, 4)) / 100,), order=2)

  @jax_array(True)
  def test_fast_path_array(self):
    devices = jax.devices()
    if len(devices) < 8:
      raise unittest.SkipTest("Test requires 8 global devices.")
    mesh_devices = np.array([[devices[0], devices[2]],
                             [devices[3], devices[1]],
                             [devices[4], devices[6]],
                             [devices[7], devices[5]]])
    shape = (8, 2)
    mesh = maps.Mesh(mesh_devices, ('x', 'y'))
    s = MeshPspecSharding(mesh, P('x', 'y'))
    inp_data = np.arange(prod(shape), dtype=np.float32).reshape(shape)

    # Explicitly put on the ordering of devices which does not match the mesh
    # ordering to make sure we reorder them in the constructor and the output
    # is correct.
    bufs = [jax.device_put(inp_data[s.device_indices(d, shape)], d)
            for d in jax.local_devices()]
    arr = array.Array(jax.ShapedArray(shape, np.float32), s, bufs, committed=True)

    f = pjit(lambda x: x, out_axis_resources=s)
    out = f(arr)
    self.assertArraysEqual([o.device() for o in out._arrays], list(mesh.devices.flat))
    self.assertArraysEqual(out, inp_data)
    out2 = f(out)
    self.assertArraysEqual([o.device() for o in out2._arrays], list(mesh.devices.flat))
    self.assertArraysEqual(out2, inp_data)


def spec_regex(s):
  return str(s).replace(r"(", r"\(").replace(r")", r"\)")


class PJitErrorTest(jtu.JaxTestCase):

  @check_1d_2d_mesh(set_mesh=True)
  def testNonDivisibleArgs(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources, None)
    mesh_size = str(np.prod([dim[1] for dim in mesh], dtype=np.int64))
    error = re.compile(
        r"One of pjit arguments.*" + spec_regex(spec) + r".*"
        r"implies that the size of its dimension 0 should be "
        r"divisible by " + mesh_size + r", but it is equal to 3", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x, in_axis_resources=spec, out_axis_resources=None)(x)

  @check_1d_2d_mesh(set_mesh=True)
  def testNonDivisibleOuts(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources, None)
    mesh_size = str(np.prod([dim[1] for dim in mesh], dtype=np.int64))
    error = re.compile(
        r"One of pjit outputs.*" + spec_regex(spec) + r".*"
        r"implies that the size of its dimension 0 should be "
        r"divisible by " + mesh_size + r", but it is equal to 3", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x, in_axis_resources=None, out_axis_resources=P(resources, None))(x)

  @check_1d_2d_mesh(set_mesh=False)
  @jtu.with_mesh([('z', 1)])
  def testUndefinedResourcesArgs(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(
        ValueError,
        r"Resource axis: x of.*" + spec_regex(spec) + " is undefined"):
      pjit(lambda x: x, in_axis_resources=spec, out_axis_resources=None)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @jtu.with_mesh([('z', 1)])
  def testUndefinedResourcesOuts(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(
        ValueError,
        r"Resource axis: x of.*" + spec_regex(spec) + " is undefined"):
      pjit(lambda x: x, in_axis_resources=None, out_axis_resources=spec)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @jtu.with_mesh([('z', 1)])
  def testUndefinedResourcesConstraint(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(
        ValueError,
        r"Resource axis: x of.*" + spec_regex(spec) + " is undefined"):
      pjit(lambda x: with_sharding_constraint(x, spec),
           in_axis_resources=None, out_axis_resources=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowArgs(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = re.compile(
        r"One of pjit arguments.*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 1", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_axis_resources=spec, out_axis_resources=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowArgsAxisResourcesNone(self):
    x = jnp.arange(2)
    spec = P(None, None)
    error = re.compile(
        r"One of pjit arguments.*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 1", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_axis_resources=spec, out_axis_resources=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowOuts(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = re.compile(
        r"One of pjit outputs.*" + spec_regex(spec) +
        r".*rank at least 2, but was applied to a value of rank 0", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_axis_resources=None, out_axis_resources=spec)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowConstraint(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = re.compile(
        r"One of with_sharding_constraint arguments" + r".*" + spec_regex(
            pxla.array_mapping_to_axis_resources(pxla._get_array_mapping(spec))) +
        r".*rank at least 2, but was applied to a value of rank 1", re.M | re.S)
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: with_sharding_constraint(x, spec),
           in_axis_resources=None, out_axis_resources=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRepeatedInResources(self):
    x = jnp.arange(2)
    for spec in [P('x', 'x'), P('x', ('y', 'x'))]:
      error = (r"A single in_axis_resources specification can map every mesh "
               r"axis to at most one positional dimension, but " +
               spec_regex(spec) + " has duplicate entries for `x`")
      with self.assertRaisesRegex(ValueError, error):
        pjit(lambda x: x, in_axis_resources=spec, out_axis_resources=None)(x)

  @jtu.with_mesh([('x', 2), ('y', 1)])
  def testRepeatedOutResources(self):
    x = jnp.arange(2)
    for spec in [P('x', 'x'), P('x', ('y', 'x'))]:
      error = (r"A single out_axis_resources specification can map every mesh "
               r"axis to at most one positional dimension, but " +
               spec_regex(spec) + " has duplicate entries for `x`")
      with self.assertRaisesRegex(ValueError, error):
        pjit(lambda x: x, in_axis_resources=None, out_axis_resources=spec)(x)

  @jtu.with_mesh([('x', 2)])
  def testInputShardsXMapAxis(self):
    spec = P('x')
    f = xmap(pjit(lambda x: x + 2, in_axis_resources=spec, out_axis_resources=None),
             in_axes=['i', ...], out_axes=['i', ...], axis_resources={'i': 'x'})
    x = jnp.arange(4).reshape((2, 2))
    error = (r"pjit input has an axis resources specification of " +
             spec_regex(spec) + r" that uses one or more "
             "mesh axes already used by "
             r"xmap to partition a named axis appearing in its named_shape \(both "
             r"use mesh axes `x`\)")
    with self.assertRaisesRegex(JAXTypeError, error):
      f(x)

  @jtu.with_mesh([('x', 2)])
  def testOutputShardsXMapAxis(self):
    spec = P('x')
    f = xmap(pjit(lambda x: x + 2, in_axis_resources=None, out_axis_resources=spec),
             in_axes=['i', ...], out_axes=['i', ...], axis_resources={'i': 'x'})
    x = jnp.arange(4).reshape((2, 2))
    error = (r"pjit output has an axis resources specification of " +
             spec_regex(spec) + r" that uses one or more "
             "mesh axes already used by "
             r"xmap to partition a named axis appearing in its named_shape \(both "
             r"use mesh axes `x`\)")
    with self.assertRaisesRegex(JAXTypeError, error):
      f(x)

  @jtu.with_mesh([('x', 2)])
  def testConstraintShardsXMapAxis(self):
    spec = P('x')
    f = xmap(lambda x: with_sharding_constraint(x, axis_resources=spec),
             in_axes=['i', ...], out_axes=['i', ...], axis_resources={'i': 'x'})
    x = jnp.arange(4).reshape((2, 2))
    error = (r"with_sharding_constraint input has an axis resources specification of " +
             spec_regex(spec) + r" that uses one or more "
             "mesh axes already used by "
             r"xmap to partition a named axis appearing in its named_shape \(both "
             r"use mesh axes `x`\)")
    with self.assertRaisesRegex(JAXTypeError, error):
      f(x)

  @jtu.with_mesh([('x', 2)])
  def testCatchesInnerXMapErrors(self):
    f = pjit(xmap(lambda x, y: x, in_axes=(['i'], ['j']), out_axes=['i', 'j'],
                  axis_resources={'i': 'x', 'j': 'x'}),
             in_axis_resources=None, out_axis_resources=None)
    x = jnp.arange(4)
    with self.assertRaises(JAXTypeError):
      f(x, x)

  def testEmptyMesh(self):
    error = (r"pjit requires a non-empty mesh! Are you sure that it's defined "
             r"at the call site?")
    with self.assertRaisesRegex(RuntimeError, error):
      pjit(lambda x: x, in_axis_resources=None, out_axis_resources=None)(jnp.arange(4))

  @jtu.with_mesh([('x', 2)])
  def testAxisResourcesMismatch(self):
    x = jnp.ones([])
    p = [None, None, None]

    pjit(lambda x: x, (p,), p)([x, x, x])  # OK

    error = re.escape(
        "pjit in_axis_resources specification must be a tree prefix of the "
        "positional arguments tuple passed to the `pjit`-decorated function. "
        "In particular, pjit in_axis_resources must either be a None, a "
        "PartitionSpec, or a tuple of length equal to the number of positional "
        "arguments. But pjit in_axis_resources is the wrong length: got a "
        "tuple or list of length 3 for an args tuple of length 2.")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x, y: x, p, p)(x, x)

    Foo = namedtuple('Foo', ['x'])
    error = "in_axis_resources is not a tuple.*might need to be wrapped"
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
        "pytree structure error: different numbers of pytree children at "
        "key path\n"
        "    pjit out_axis_resources tree root\n"
        "At that key path, the prefix pytree pjit out_axis_resources has a "
        "subtree of type\n"
        "    <class 'list'>\n"
        "with 2 children, but at the same key path the full pytree has a "
        "subtree of the same type but with 3 children.")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x, (p,), [p, None])([x, x, x])  # Error, we raise a generic tree mismatch message

  @jtu.with_mesh([('x', 2)])
  def testNestedDifferentResources(self):
    @partial(pjit, in_axis_resources=P('x'), out_axis_resources=None)
    def f(x):
      with maps.Mesh(np.array([jax.local_devices()[0]]), ('x')):
        @partial(pjit, in_axis_resources=P('x'), out_axis_resources=None)
        def h(x):
          return x
        return h(x)
    xshape = (2, 5, 6)
    x = jnp.arange(np.prod(xshape)).reshape(xshape)
    with self.assertRaisesRegex(RuntimeError,
                                "Changing the physical mesh is not allowed.*"):
      f(x)


class UtilTest(jtu.JaxTestCase):

  def testOpShardingRoundTrip(self):
    FakeDevice = namedtuple('FakeDevice', ['id'])
    mesh_named_shape = OrderedDict([('a', 2), ('b', 3), ('c', 4), ('d', 7), ('e', 4)])
    mesh_axes, mesh_shape = unzip2(mesh_named_shape.items())
    devices = [FakeDevice(i) for i in range(np.prod(list(mesh_shape)))]
    mesh = pxla.Mesh(np.array(devices).reshape(*mesh_shape), tuple(mesh_axes))

    dims = 5
    aval = jax.core.ShapedArray((len(devices),) * dims, jnp.float32)
    def roundtrip(spec):
      op_sharding = MeshPspecSharding(mesh, spec)._to_xla_op_sharding(aval.ndim)
      parsed_spec = pjit_lib.parse_flatten_op_sharding(op_sharding, mesh)[0].partitions
      self.assertEqual(parsed_spec[:len(spec)], spec)
      self.assertEqual(parsed_spec[len(spec):], ((),) * (len(parsed_spec) - len(spec)))

    special_specs = [P()]
    for spec in special_specs:
      roundtrip(spec)

    rng = self.rng()
    for i in range(100):
      spec = [()] * dims
      for axis in rng.permutation(mesh_axes)[:rng.randint(low=1, high=len(mesh_axes) + 1)]:
        spec[rng.choice(dims)] += (axis,)
      roundtrip(P(*spec))

  @parameterized.named_parameters(
      ("linear", {'x': 0, 'y': 1, 'z': 2}, P(('x',), ('y',), ('z',))),
      ("combine", {'x': 0, 'y': 0, 'z': 1}, P(('x', 'y'), ('z',))),
      ("skip", {'x': 0, 'y': 0, 'z': 2}, P(('x', 'y'), None, ('z',))),
      ("multi_skip", {'x': 0, 'y': 1, 'z': 3}, P(('x',), ('y',), None, ('z',))),
  )
  def test_array_mapping_to_axis_resources(self, inp, expected_out):
    self.assertEqual(pxla.array_mapping_to_axis_resources(inp), expected_out)

  def test_get_input_metadata_fully_replicated(self):
    global_mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    global_in_aval1 = jax.core.ShapedArray((4, 4), jnp.int32)
    global_in_aval2 = jax.core.ShapedArray((4, 4, 4), jnp.int32)
    global_in_aval3 = jax.core.ShapedArray((), jnp.int32)
    in_avals = [global_in_aval1, global_in_aval2, global_in_aval3]

    mp = MeshPspecSharding(global_mesh, P(None))

    _, out_indices, _ = pxla._get_input_metadata(
        in_avals, [mp, mp, mp], [False, False, False])

    self.assertLen(out_indices, len(in_avals))
    self.assertTrue(all(len(out) == len(global_mesh.local_devices)
                    for out in out_indices))
    self.assertTrue(all(len(i) == aval.ndim
                    for out, aval in safe_zip(out_indices, in_avals) for i in out))
    self.assertTrue(all(i == (slice(None),) * aval.ndim
                    for out, aval in safe_zip(out_indices, in_avals) for i in out))

  def test_mesh_sharding_spec(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    array_mapping = pxla._get_array_mapping(P('x', 'y'))
    aval = jax.ShapedArray((1, 1), jnp.int32)
    with self.assertRaisesRegex(
        ValueError,
        'The aval shape on dimension 0 is 1 and the size of axis x is 4. The '
        'aval shape % axis size should be zero but got 1'
    ):
      pxla.mesh_sharding_specs(mesh.shape, mesh.axis_names)(aval, array_mapping)

  @parameterized.named_parameters(
      ("all_unspecified", (pjit_lib._UNSPECIFIED, pjit_lib._UNSPECIFIED), AssertionError),
      ("only_unspecified", pjit_lib._UNSPECIFIED),
      ("all_specified", (P('x'), P('y'))),
      ("only_specified", P('x')),
      ("mix_1", (P('x'), pjit_lib._UNSPECIFIED), ValueError),
      ("mix_2", (P('x'), pjit_lib._UNSPECIFIED, P('y')), ValueError),
      ("mix_3", (pjit_lib._UNSPECIFIED, P('x'), P('y')), ValueError),
      ("mix_4", (pjit_lib._UNSPECIFIED, P('x'), pjit_lib._UNSPECIFIED), ValueError),
  )
  def test_all_or_non_unspecified(self, axis_resources, error=None):
    entries, _ = jax.tree_util.tree_flatten(axis_resources, is_leaf=lambda x: x is None)
    if error is not None:
      with self.assertRaises(error):
        pjit_lib._check_all_or_none_unspecified(entries, 'test axis resources')
    else:
      pjit_lib._check_all_or_none_unspecified(entries, 'test axis resources')

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

    self.assertTrue(pxla.are_op_shardings_equal(op1, op2))
    self.assertFalse(pxla.are_op_shardings_equal(op1, op3))
    self.assertFalse(pxla.are_op_shardings_equal(op2, op3))

    if xla_extension_version >= 81:
      hs1 = xc.HloSharding.from_proto(op1)
      hs2 = xc.HloSharding.from_proto(op2)
      hs3 = xc.HloSharding.from_proto(op3)

      self.assertEqual(hash(hs1), hash(hs2))
      self.assertNotEqual(hash(hs1), hash(hs3))
      self.assertNotEqual(hash(hs2), hash(hs3))

  def test_op_sharding_partial_sharding(self):
    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [4, 1]
    op1.tile_assignment_devices = [0, 1, 2, 3]
    op1.last_tile_dims = [xc.OpSharding.Type.REPLICATED]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.OTHER
    op2.tile_assignment_dimensions = [4, 1]
    op2.tile_assignment_devices = [0, 1, 2, 3]
    op2.last_tile_dims = [xc.OpSharding.Type.REPLICATED]

    self.assertTrue(pxla.are_op_shardings_equal(op1, op2))

    if xla_extension_version >= 81:
      hs1 = xc.HloSharding.from_proto(op1)
      hs2 = xc.HloSharding.from_proto(op2)
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

    self.assertFalse(pxla.are_op_shardings_equal(op1, op2))

    if xla_extension_version >= 81:
      hs1 = xc.HloSharding.from_proto(op1)
      hs2 = xc.HloSharding.from_proto(op2)
      self.assertNotEqual(hash(hs1), hash(hs2))

  def test_device_indices_cache(self):
    if xla_extension_version < 81:
      raise unittest.SkipTest('HloSharding is available after '
                              'xla_extension_version >= 81')

    op1 = xc.OpSharding()
    op1.type = xc.OpSharding.Type.OTHER
    op1.tile_assignment_dimensions = [1, 1, 2, 1]
    op1.tile_assignment_devices = [0, 1]
    op1.last_tile_dims = [xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL]

    op2 = xc.OpSharding()
    op2.type = xc.OpSharding.Type.REPLICATED

    shape = (8, 4)
    devices = jax.devices()

    ops = OpShardingSharding(devices, op1)
    ops.devices_indices_map(shape)
    cache_info1 = OpShardingSharding.devices_indices_map.cache_info()

    ops.devices_indices_map(shape)
    cache_info2 = OpShardingSharding.devices_indices_map.cache_info()
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)

    ops = OpShardingSharding(devices, op2)
    ops.devices_indices_map(shape)
    cache_info3 = OpShardingSharding.devices_indices_map.cache_info()
    self.assertEqual(cache_info3.hits, cache_info2.hits + 1)

    ops.devices_indices_map(shape)
    cache_info4 = OpShardingSharding.devices_indices_map.cache_info()
    self.assertEqual(cache_info4.hits, cache_info3.hits + 1)


  def test_op_sharding_semantically_replicated(self):
    if xla_extension_version < 81:
      raise unittest.SkipTest(
          'HloSharding is not available for this test so it cannot be tested.')

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

    self.assertTrue(pxla.is_op_sharding_replicated(op1))
    self.assertTrue(pxla.is_op_sharding_replicated(op2))
    self.assertTrue(pxla.is_op_sharding_replicated(op3))
    self.assertTrue(pxla.is_op_sharding_replicated(op4))
    self.assertTrue(pxla.are_op_shardings_equal(op1, op2))
    self.assertTrue(pxla.are_op_shardings_equal(op2, op3))
    self.assertTrue(pxla.are_op_shardings_equal(op3, op4))

  def test_op_sharding_manual_replicated(self):
    if xla_extension_version < 81:
      raise unittest.SkipTest(
          'HloSharding is not available for this test so it cannot be tested.')

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

    self.assertTrue(pxla.is_op_sharding_replicated(op1))
    self.assertTrue(pxla.is_op_sharding_replicated(op2))
    self.assertTrue(pxla.are_op_shardings_equal(op1, op2))
    self.assertTrue(pxla.are_op_shardings_equal(op1, op3))

  def test_op_sharding_cache_on_mesh_pspec_sharding(self):
    ndim = 2
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    mps1 = MeshPspecSharding(mesh, P('x', 'y'))
    op1 = mps1._to_xla_op_sharding(ndim)
    cache_info1 = MeshPspecSharding._to_xla_op_sharding.cache_info()

    mps2 = MeshPspecSharding(mesh, P('x', 'y'))
    op2 = mps2._to_xla_op_sharding(ndim)
    cache_info2 = MeshPspecSharding._to_xla_op_sharding.cache_info()

    self.assertEqual(id(op1), id(op2))
    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)
    self.assertEqual(cache_info2.currsize, cache_info1.currsize)

  def test_simulated_training_cache_in_pjit(self):
    ndim = 2
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))

    mps1 = MeshPspecSharding(mesh, P('x', 'y'))
    op_sharding_sharding = pjit_lib.to_op_sharding_sharding(mps1, ndim)
    next_loop_sharding = simulated_cached_fun(op_sharding_sharding)
    cache_info1 = simulated_cached_fun.cache_info()

    next_op_sharding_sharding = pjit_lib.to_op_sharding_sharding(
        next_loop_sharding, ndim)
    simulated_cached_fun(next_op_sharding_sharding)
    cache_info2 = simulated_cached_fun.cache_info()

    self.assertEqual(cache_info2.hits, cache_info1.hits + 1)
    self.assertEqual(cache_info2.misses, cache_info1.misses)
    self.assertEqual(id(next_op_sharding_sharding._op_sharding),
                     id(op_sharding_sharding._op_sharding))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
