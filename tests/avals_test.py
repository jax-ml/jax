# Copyright 2026 The JAX Authors.
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
import numpy as np

import jax
from jax import avals
from jax.avals import AvalDType, MeshAxis, bf16, bool_, c64, c128, f16, f32, f64, float0, i8, i16, i32, i64, u8, u16, u32, u64
from jax.export import symbolic_shape
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P, use_abstract_mesh
from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import test_util as jtu

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class AvalsTest(jtu.JaxTestCase):

  def test_basic_shapes(self):
    self.assertEqual(f32[()], core.ShapedArray((), np.float32))
    self.assertEqual(f32[3], core.ShapedArray((3,), np.float32))
    self.assertEqual(f32[3, 4], core.ShapedArray((3, 4), np.float32))
    self.assertEqual(f32[np.int32(3), np.int64(4)],
                     core.ShapedArray((3, 4), np.float32))

  def test_dtypes_and_no_x64_truncation(self):
    cases = [
        (bool_, np.bool_),
        (avals.bool, np.bool_),
        (avals.i2, dtypes.int2),
        (avals.i4, dtypes.int4),
        (i8, np.int8),
        (i16, np.int16),
        (i32, np.int32),
        (i64, np.int64),
        (avals.u2, dtypes.uint2),
        (avals.u4, dtypes.uint4),
        (u8, np.uint8),
        (u16, np.uint16),
        (u32, np.uint32),
        (u64, np.uint64),
        (AvalDType(dtypes.float8_e4m3fn), dtypes.float8_e4m3fn),
        (bf16, dtypes.bfloat16),
        (f16, np.float16),
        (f32, np.float32),
        (f64, np.float64),
        (c64, np.complex64),
        (c128, np.complex128),
        (float0, dtypes.float0),
    ]
    with jax.enable_x64(False):
      for builder, dt in cases:
        aval = builder[2, 3]
        self.assertEqual(aval, core.ShapedArray((2, 3), np.dtype(dt)))
        self.assertEqual(aval.dtype, np.dtype(dt))

  def test_extended_dtype(self):
    key_dtype = jax.typeof(jax.random.key(0)).dtype
    key_builder = AvalDType(key_dtype)
    self.assertEqual(key_builder[4], core.ShapedArray((4,), key_dtype))
    self.assertEqual(key_builder[()], core.ShapedArray((), key_dtype))

  def test_update_chained(self):
    aval = f32[3].update(weak_type=True)
    self.assertEqual(aval, core.ShapedArray((3,), np.float32, weak_type=True))

  def test_sharded_with_abstract_mesh_context(self):
    mesh = AbstractMesh((3, 2), ('X', 'Y'))
    X = MeshAxis('X')
    Y = MeshAxis('Y')
    with use_abstract_mesh(mesh):
      self.assertEqual(
          f32[3@X],
          core.ShapedArray((3,), np.float32, sharding=NamedSharding(mesh, P('X'))))
      self.assertEqual(
          f32[6@X, 4, 2@Y],
          core.ShapedArray((6, 4, 2), np.float32,
                           sharding=NamedSharding(mesh, P('X', None, 'Y'))))
      self.assertEqual(
          f32[4],
          core.ShapedArray((4,), np.float32, sharding=NamedSharding(mesh, P(None))))
      self.assertEqual(
          f32[()],
          core.ShapedArray((), np.float32, sharding=NamedSharding(mesh, P())))

  def test_multi_axis_accumulation(self):
    mesh = AbstractMesh((3, 2), ('X', 'Y'))
    X, Y = mesh.axes
    with use_abstract_mesh(mesh):
      aval = f32[6 @ X @ Y, 4]
    expected = core.ShapedArray(
        (6, 4), np.float32, sharding=NamedSharding(mesh, P(('X', 'Y'), None)))
    self.assertEqual(aval, expected)

  def test_mesh_axes_with_set_mesh(self):
    if jax.device_count() < 6:
      self.skipTest('Requires at least 6 devices')
    mesh = jax.make_mesh((3, 2), ('X', 'Y'))
    X, Y = mesh.axes
    with jax.set_mesh(mesh):
      arr = jnp.zeros((6, 4), dtype=np.float32, out_sharding=P(('X', 'Y'), None))
      self.assertEqual(f32[6 @ X @ Y, 4], jax.typeof(arr))

  def test_inside_shard_map(self):
    if jax.device_count() < 6:
      self.skipTest('Requires at least 6 devices')
    mesh = jax.make_mesh((3, 2), ('X', 'Y'))
    _, Y = mesh.axes

    with jax.set_mesh(mesh):
      @jax.shard_map(in_specs=P('X', None), out_specs=P('X', None),
                     axis_names={'X'})
      def f(x):
        z = jnp.zeros((1, 2), dtype=np.float32, out_sharding=P(None, 'Y'))
        self.assertEqual(f32[1, 2@Y], jax.typeof(z))
        self.assertEqual(
            f32[1, 2@Y].update(
                manual_axis_type=jax.sharding.ManualAxisType(varying={'X'})),
            jax.typeof(x))
        return x

      f(jnp.zeros((3, 2), dtype=np.float32, out_sharding=P('X', 'Y')))

  def test_symbolic_shapes(self):
    mesh = AbstractMesh((4,), ('X',))
    X, = mesh.axes
    b, n = symbolic_shape('4*b, n')
    self.assertEqual(f32[b, n], core.ShapedArray((b, n), np.float32))
    with use_abstract_mesh(mesh):
      self.assertEqual(
          f32[b@X, n],
          core.ShapedArray((b, n), np.float32,
                           sharding=NamedSharding(mesh, P('X', None))))

  def test_errors(self):
    # Sharded axis without any active context mesh:
    m1 = AbstractMesh((2,), ('X',))
    X, = m1.axes
    with self.assertRaisesRegex(ValueError, "is not an Explicit axis in the active context mesh"):
      _ = f32[2@X]

    # Axis not in active context mesh:
    m2 = AbstractMesh((2,), ('Y',))
    with use_abstract_mesh(m2):
      with self.assertRaisesRegex(ValueError, "is not an Explicit axis in the active context mesh"):
        _ = f32[2@X]

    # Non-Explicit (Auto or Manual) mesh axis:
    auto_mesh = AbstractMesh((2,), ('X',), axis_types=(AxisType.Auto,))
    with use_abstract_mesh(auto_mesh):
      with self.assertRaisesRegex(ValueError, "is not an Explicit axis in the active context mesh"):
        _ = f32[2@X]

    if jax.device_count() >= 2:
      raw_mesh = Mesh(np.array(jax.devices()[:2]), ('X',))  # defaults to Auto
      with jax.set_mesh(raw_mesh):
        with self.assertRaisesRegex(ValueError, "is not an Explicit axis in the active context mesh"):
          _ = f32[2@X]

    manual_mesh = AbstractMesh((2,), ('X',), axis_types=(AxisType.Manual,))
    with use_abstract_mesh(manual_mesh):
      with self.assertRaisesRegex(ValueError, "is not an Explicit axis in the active context mesh"):
        _ = f32[2@X]

    # Indivisible shape:
    with use_abstract_mesh(m1):
      with self.assertRaisesRegex(ValueError, "does not evenly divide"):
        _ = f32[5@X]


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
