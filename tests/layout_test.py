# Copyright 2023 The JAX Authors.
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

import math
import os
import re
from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
from jax._src import config
from jax._src.layout import Layout, DeviceLocalLayout as DLL
from jax._src import test_util as jtu
from jax._src.util import safe_zip
from jax._src import xla_bridge
from jax._src.lib import xla_extension_version

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


def tearDownModule():
  if prev_xla_flags is None:
    del os.environ["XLA_FLAGS"]
  else:
    os.environ["XLA_FLAGS"] = prev_xla_flags
  xla_bridge.get_backend.cache_clear()


pattern = re.compile(r"\{(.*?):")

# Extract minor_to_major from str(layout) because layout doesn't have a
# minor_to_major property yet.
def extract_minor_to_major(l):
  match = re.search(pattern, str(l))
  return tuple(int(i) for i in match.groups()[0].split(','))


class LayoutTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest("Layouts do not work on CPU and GPU backends yet.")
    if xla_extension_version < 215:
      self.skipTest('All tests require xla_extension_version >= 215')
    super().setUp()

  def test_auto_layout(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape1 = (128, 128)
    shape2 = (128, 128)
    s1 = NamedSharding(mesh, P('x', 'y'))
    s2 = NamedSharding(mesh, P('x'))

    def apply(x, y):
      return x.T, y.T

    def init(x, y):
      return x * 2, y * 2

    np_inp1 = np.arange(math.prod(shape1)).reshape(shape1)
    np_inp2 = np.arange(math.prod(shape2)).reshape(shape2)
    sds1 = jax.ShapeDtypeStruct(np_inp1.shape, np_inp1.dtype, sharding=s1)
    sds2 = jax.ShapeDtypeStruct(np_inp2.shape, np_inp2.dtype, sharding=s2)

    lowered_apply = jax.jit(apply).lower(
        sds1, sds2, _in_layouts=Layout(DLL.AUTO), _out_layouts=Layout(DLL.AUTO))
    compiled_apply = lowered_apply.compile()

    arg_layouts, kw_layouts = compiled_apply._input_layouts()
    self.assertEmpty(kw_layouts)

    for i, o in zip(arg_layouts, compiled_apply._output_layouts()):
      self.assertEqual(extract_minor_to_major(i),
                       extract_minor_to_major(o)[::-1])

    init_compiled = jax.jit(init).lower(
        sds1, sds2, _out_layouts=arg_layouts).compile()

    for i, o in zip(init_compiled._input_layouts()[0],
                    init_compiled._output_layouts()):
      self.assertEqual(i, o)

    arr1 = jax.device_put(np_inp1, s1)
    arr2 = jax.device_put(np_inp2, s2)

    with jtu.count_aot_jit_cpp_cache_miss() as init_count:
      init_out = init_compiled(arr1, arr2)
      init_compiled(arr1, arr2)
    self.assertEqual(init_count[0], 1)

    self.assertEqual(init_out[0].layout, init_compiled._output_layouts()[0])
    self.assertEqual(init_out[1].layout, init_compiled._output_layouts()[1])

    with jtu.count_aot_jit_cpp_cache_miss() as apply_count:
      apply_out = compiled_apply(*init_out)
      compiled_apply(*init_out)
    self.assertEqual(apply_count[0], 1)

    self.assertEqual(apply_out[0].layout, compiled_apply._output_layouts()[0])
    self.assertEqual(apply_out[1].layout, compiled_apply._output_layouts()[1])

    self.assertTupleEqual(extract_minor_to_major(apply_out[0].layout),
                          extract_minor_to_major(init_out[0].layout)[::-1])
    self.assertTupleEqual(extract_minor_to_major(apply_out[1].layout),
                          extract_minor_to_major(init_out[1].layout)[::-1])

    self.assertArraysEqual(init_out[0], np_inp1 * 2)
    self.assertArraysEqual(init_out[1], np_inp2 * 2)
    self.assertArraysEqual(apply_out[0], (np_inp1 * 2).T)
    self.assertArraysEqual(apply_out[1], (np_inp2 * 2).T)

  def test_default_layout(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (4, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    lowered = jax.jit(f).lower(sds, _in_layouts=None, _out_layouts=None)
    self.assertIn("default", lowered.as_text())
    compiled = lowered.compile()
    out = compiled(arr)

    self.assertTupleEqual(
        extract_minor_to_major(compiled._input_layouts()[0][0]), (2, 1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled._output_layouts()), (2, 1, 0))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    compiled_auto = jax.jit(f).lower(sds, _in_layouts=Layout(DLL.AUTO),
                                     _out_layouts=Layout(DLL.AUTO)).compile()
    self.assertTupleEqual(
        extract_minor_to_major(compiled_auto._input_layouts()[0][0]), (2, 1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled_auto._output_layouts()), (0, 1, 2))

  def test_in_layouts_out_layouts(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (8, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    compiled = jax.jit(f).lower(
        arr, _in_layouts=Layout(), _out_layouts=Layout(DLL.AUTO)).compile()
    self.assertTupleEqual(
        extract_minor_to_major(compiled._input_layouts()[0][0]), (1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled._output_layouts()), (0, 1))

    out = compiled(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.layout, compiled._output_layouts())
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))

  def test_sharding_and_layouts(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (4, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))

    compiled = jax.jit(lambda x: x.T, in_shardings=s, out_shardings=s).lower(
        np_inp, _in_layouts=Layout(DLL.AUTO),
        _out_layouts=Layout(DLL.AUTO)).compile()
    out = compiled(np_inp)
    self.assertTupleEqual(
        extract_minor_to_major(compiled._input_layouts()[0][0]), (1, 0))
    self.assertTupleEqual(
        extract_minor_to_major(compiled._output_layouts()), (0, 1))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, s)

  def test_dce_in_layouts(self):
    def f(x, y, z, a, b, c):
      return z * 2, b.T

    shape = (8, 2)
    inps = [np.arange(math.prod(shape)).reshape(shape)] * 6
    compiled = jax.jit(f).lower(*inps, _in_layouts=Layout(DLL.AUTO),
                                _out_layouts=Layout(DLL.AUTO)).compile()
    arg_layouts, _ = compiled._input_layouts()
    out1, out2 = compiled(*inps)

    compiled2 = jax.jit(f).lower(*inps, _in_layouts=arg_layouts).compile()
    out3, out4 = compiled2(*inps)

    for l1, l2 in safe_zip(arg_layouts, compiled2._input_layouts()[0]):
      self.assertEqual(l1, l2)

    self.assertArraysEqual(out1, out3)
    self.assertArraysEqual(out2, out4)

    arrs = [jax.device_put(i, l) for i, l in zip(inps, arg_layouts)]
    out5, out6 = jax.jit(f)(*arrs)
    self.assertArraysEqual(out1, out5)
    self.assertArraysEqual(out2, out6)

  def test_aot_layout_mismatch(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (256, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x'))

    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return (x * 2).T

    with self.assertRaisesRegex(
        ValueError,
        'Layout passed to jit does not match the layout on the respective arg'):
      jax.jit(f).lower(arr, _in_layouts=Layout(DLL.AUTO))

    compiled = jax.jit(f).lower(
        sds, _in_layouts=Layout(DLL.AUTO),
        _out_layouts=Layout(DLL.AUTO)).compile()

    with self.assertRaisesRegex(
        ValueError,
        r'Compiled object called with input layout\(s\) does'
        r' not match the layout\(s\) the computation was'
        ' compiled with'):
      compiled(arr)

  def test_cpu_default_backend_layout(self):
    out_cpu = jax.jit(jnp.dot, backend='cpu')(np.ones((8, 8)), np.ones((8, 8)))

    jax.jit(jnp.dot, backend=jax.default_backend()).lower(
        out_cpu, out_cpu).compile()  # doesn't crash

  def test_device_put_concrete_layout(self):
    mesh = jtu.create_global_mesh((2, 2), ('x', 'y'))
    shape = (8, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    compiled = jax.jit(
        lambda x: x * 2).lower(arr, _out_layouts=Layout(DLL.AUTO)).compile()
    col = compiled._output_layouts()

    out = jax.device_put(np_inp, col)
    self.assertEqual(out.layout, col)
    self.assertArraysEqual(out, np_inp)
    for s in out.addressable_shards:
      self.assertEqual(out.layout.device_local_layout,
                       s.data.layout.device_local_layout)

  def test_device_put_non_concrete_layout_error(self):
    np_inp = np.arange(16).reshape(8, 2)

    l1 = Layout(DLL.AUTO, SingleDeviceSharding(jax.devices()[0]))
    with self.assertRaisesRegex(
        ValueError, 'sharding and device_local_layout.*should be concrete'):
      jax.device_put(np_inp, l1)

    l2 = Layout(DLL.AUTO)
    with self.assertRaisesRegex(
        ValueError, 'sharding and device_local_layout.*should be concrete'):
      jax.device_put(np_inp, l2)

    l3 = Layout(None, SingleDeviceSharding(jax.devices()[0]))
    out = jax.device_put(np_inp, l3)
    self.assertArraysEqual(out, np_inp)
    self.assertTrue(out._committed)

  def invalid_layout_spec(self):
    x = np.arange(8)
    compiled = jax.jit(lambda x: x).lower(x).compile()
    with self.assertRaisesRegex(
        ValueError, 'Sharding has to be concrete when layout.*'):
      Layout(compiled._output_layouts()[0], None)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
