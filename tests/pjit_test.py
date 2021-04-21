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

from contextlib import contextmanager
from functools import partial
from typing import Generator, List, Tuple
from unittest import SkipTest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax import test_util as jtu
# TODO(skye): do we still wanna call this PartitionSpec?
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax.interpreters import pxla
from jax._src.util import unzip2, prod, curry

from jax.config import config
config.parse_flags_with_absl()

ignore_pjit_warning = partial(
    jtu.ignore_warning, message=".*is an experimental.*")

# TODO(skye): move into test_util and dedup with xmap_test.py
MeshSpec = List[Tuple[str, int]]

@contextmanager
def with_mesh(named_shape: MeshSpec) -> Generator[None, None, None]:
  """Test utility for setting up meshes given mesh data from `schedules`."""
  # This is similar to the `with_mesh` function above, but isn't a decorator.
  axis_names, shape = unzip2(named_shape)
  size = prod(shape)
  local_devices = list(jax.local_devices())
  if len(local_devices) < size:
    raise SkipTest(f"Test requires {size} local devices")
  mesh_devices = np.array(local_devices[:size]).reshape(shape)
  with mesh(mesh_devices, axis_names):
    yield

def with_mesh_from_kwargs(f):
  return lambda *args, **kwargs: with_mesh(kwargs['mesh'])(f)(*args, **kwargs)


# TODO(skye): make the buffer donation utils part of JaxTestCase
class PJitTest(jtu.BufferDonationTestCase):

  @ignore_pjit_warning()
  @with_mesh([('x', 2)])
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
    self.assertAllClose(actual.device_buffers[0].to_py(), expected,
                        check_dtypes=False)

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 2)])
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
    self.assertAllClose(actual.device_buffers[0].to_py(), split0,
                        check_dtypes=False)
    self.assertAllClose(actual.device_buffers[1].to_py(), split0,
                        check_dtypes=False)
    self.assertAllClose(actual.device_buffers[2].to_py(), split1,
                        check_dtypes=False)
    self.assertAllClose(actual.device_buffers[3].to_py(), split1,
                        check_dtypes=False)

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 2)])
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
    self.assertAllClose(actual.device_buffers[0].to_py(), splits[0],
                        check_dtypes=False)
    self.assertAllClose(actual.device_buffers[1].to_py(), splits[1],
                        check_dtypes=False)
    self.assertAllClose(actual.device_buffers[2].to_py(), splits[2],
                        check_dtypes=False)
    self.assertAllClose(actual.device_buffers[3].to_py(), splits[3],
                        check_dtypes=False)

  @ignore_pjit_warning()
  @with_mesh([('x', 2)])
  def testBufferDonation(self):
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

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 1)])
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
    self.assertAllClose(actual.device_buffers[0].to_py(), expected,
                        check_dtypes=False)

    hlo = jax.xla_computation(f)(np.ones(shape))
    # Annotation from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 1)])
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

    hlo = jax.xla_computation(f)(x)
    # Annotations from with_sharding_constraint
    self.assertIn("sharding={devices=[2,1]0,1}", hlo.as_hlo_text())
    self.assertIn("sharding={devices=[1,2]0,1}", hlo.as_hlo_text())
    # Annotation from pjit
    self.assertIn("sharding={replicated}", hlo.as_hlo_text())

  @ignore_pjit_warning()
  def testCaching(self):
    def f(x):
      assert should_be_tracing
      return jnp.sin(x) * 2

    x = np.arange(16).reshape(4, 4)
    devices = np.array(list(jax.local_devices())[:4])
    if devices.size < 4:
      raise SkipTest("Test requires 4 devices")
    devices = devices.reshape((2, 2))
    with mesh(devices, ('x', 'y')):
      should_be_tracing = True
      pjit(f, in_axis_resources=P(('x', 'y')), out_axis_resources=None)(x)
      should_be_tracing = False
      pjit(f, in_axis_resources=P(('x', 'y')), out_axis_resources=None)(x)
    # Re-create the mesh to make sure that has no influence on caching
    with mesh(devices, ('x', 'y')):
      should_be_tracing = False
      pjit(f, in_axis_resources=P(('x', 'y')), out_axis_resources=None)(x)

  @with_mesh([('x', 2), ('y', 1)])
  def testNested(self):
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4)
    f = pjit(lambda x: x.sum() + h.sum(), in_axis_resources=P('x', 'y'), out_axis_resources=None)
    g = pjit(lambda x: f(jnp.sin(x)), in_axis_resources=P('x', None), out_axis_resources=None)
    x = jnp.arange(16).reshape((4, 4))
    y = g(x)
    self.assertAllClose(y, jnp.sin(x).sum() + h.sum())
    self.assertTrue(hasattr(y, "sharding_spec"))

  @with_mesh([('x', 2), ('y', 1)])
  def testJVP(self):
    # Add a constant captured by the nested pjit to make things more complicated
    h = jnp.arange(4)
    f = pjit(lambda x: x.sum() + h.sum(), in_axis_resources=P('x', 'y'), out_axis_resources=None)
    g = pjit(lambda x: f(x + 2), in_axis_resources=P('x', None), out_axis_resources=None)
    jtu.check_grads(g, (jnp.arange(16, dtype=jnp.float32).reshape((4, 4)),),
                    order=2, modes=["fwd"], eps=1)

  # TODO(skye): add more unit tests once API is more finalized

@curry
def check_1d_2d_mesh(f, set_mesh):
  return parameterized.named_parameters(
    {"testcase_name": "_" + name, "mesh": mesh, "resources": resources}
    for name, mesh, resources in (
      ("2", (("x", 2),), "x"),
      ("2x1", (("x", 2), ("y", 1)), ("x", "y")),
      ("2x2", (("x", 2), ("y", 2)), ("x", "y")),
    ))(with_mesh_from_kwargs(f) if set_mesh else f)

def spec_regex(s):
  return str(s).replace(r"(", r"\(").replace(r")", r"\)")

class PJitErrorTest(jtu.JaxTestCase):
  @check_1d_2d_mesh(set_mesh=True)
  @ignore_pjit_warning()
  def testNonDivisibleArgs(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources, None)
    mesh_size = str(np.prod([dim[1] for dim in mesh], dtype=np.int64))
    with self.assertRaisesRegex(ValueError,
                                r"One of pjit arguments.*" + spec_regex(spec) + r".*"
                                r"implies that the size of its dimension 0 should be "
                                r"divisible by " + mesh_size + r", but it is equal to 3"):
      pjit(lambda x: x, in_axis_resources=spec, out_axis_resources=None)(x)

  @check_1d_2d_mesh(set_mesh=True)
  @ignore_pjit_warning()
  def testNonDivisibleOuts(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources, None)
    mesh_size = str(np.prod([dim[1] for dim in mesh], dtype=np.int64))
    with self.assertRaisesRegex(ValueError,
                                r"One of pjit outputs.*" + spec_regex(spec) + r".*"
                                r"implies that the size of its dimension 0 should be "
                                r"divisible by " + mesh_size + r", but it is equal to 3"):
      pjit(lambda x: x, in_axis_resources=None, out_axis_resources=P(resources, None))(x)

  @check_1d_2d_mesh(set_mesh=True)
  @ignore_pjit_warning()
  def testNonDivisibleConstraint(self, mesh, resources):
    x = jnp.ones((3, 2))
    spec = P(resources,)
    mesh_size = str(np.prod([dim[1] for dim in mesh], dtype=np.int64))
    with self.assertRaisesRegex(ValueError,
                                r"One of with_sharding_constraint arguments"
                                r".*" + spec_regex(spec) + r".*implies that the size of "
                                r"its dimension 0 should be divisible by " + mesh_size +
                                r", but it is equal to 3"):
      pjit(lambda x: with_sharding_constraint(x, spec),
           in_axis_resources=None, out_axis_resources=None)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @ignore_pjit_warning()
  def testUndefinedResourcesArgs(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(ValueError,
                                r"One of pjit arguments.*" + spec_regex(spec) + r", "
                                r"but resource axis x is undefined."):
      pjit(lambda x: x, in_axis_resources=spec, out_axis_resources=None)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @ignore_pjit_warning()
  def testUndefinedResourcesOuts(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(ValueError,
                                r"One of pjit outputs.*" + spec_regex(spec) + r", "
                                r"but resource axis x is undefined."):
      pjit(lambda x: x, in_axis_resources=None, out_axis_resources=spec)(x)

  @check_1d_2d_mesh(set_mesh=False)
  @ignore_pjit_warning()
  def testUndefinedResourcesConstraint(self, mesh, resources):
    x = jnp.ones((2, 2))
    spec = P(resources,)
    with self.assertRaisesRegex(ValueError,
                                r"One of with_sharding_constraint arguments"
                                r".*" + spec_regex(spec) + r", but resource axis "
                                r"x is undefined."):
      pjit(lambda x: with_sharding_constraint(x, spec),
           in_axis_resources=None, out_axis_resources=None)(x)

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowArgs(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = (r"One of pjit arguments.*" + spec_regex(spec) + r", which implies "
             r"that it has a rank of at least 2, but it is 1")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_axis_resources=spec, out_axis_resources=None)(x)

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowOuts(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = (r"One of pjit outputs.*" + spec_regex(spec) + r", which implies "
             r"that it has a rank of at least 2, but it is 0")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: x.sum(), in_axis_resources=None, out_axis_resources=spec)(x)

  @ignore_pjit_warning()
  @with_mesh([('x', 2), ('y', 1)])
  def testRankTooLowConstraint(self):
    x = jnp.arange(2)
    spec = P('x', 'y')
    error = (r"One of with_sharding_constraint arguments " +
             r"was given.*" + spec_regex(spec) + r", which implies "
             r"that it has a rank of at least 2, but it is 1")
    with self.assertRaisesRegex(ValueError, error):
      pjit(lambda x: with_sharding_constraint(x, spec),
           in_axis_resources=None, out_axis_resources=None)(x)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
