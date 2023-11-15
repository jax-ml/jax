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
from absl.testing import absltest
import numpy as np

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src import config
from jax._src import layout
from jax._src import test_util as jtu
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


class LayoutTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest("Layouts do not work on CPU and GPU backends yet.")
    if xla_extension_version < 215:
      self.skipTest('All tests require xla_extension_version >= 215')
    super().setUp()

  def test_auto_layout(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape1 = (128, 128)
    shape2 = (128, 128)

    def apply(x, y):
      return x.T, y.T

    def init(x, y):
      return x * 2, y * 2

    np_inp1 = np.arange(math.prod(shape1)).reshape(shape1)
    arr1 = jax.device_put(np_inp1, NamedSharding(mesh, P('x', 'y')))
    np_inp2 = np.arange(math.prod(shape2)).reshape(shape2)
    arr2 = jax.device_put(np_inp2, NamedSharding(mesh, P('x')))

    lowered_apply = jax.jit(apply).lower(arr1, arr2, _in_layouts=layout.AUTO,
                                         _out_layouts=layout.AUTO)
    compiled_apply = lowered_apply.compile()

    arg_layouts, kw_layouts = compiled_apply._input_layouts()
    self.assertEmpty(kw_layouts)
    for i, o in zip(arg_layouts, compiled_apply._output_layouts()):
      self.assertEqual(i.minor_to_major, o.minor_to_major[::-1])

    init_compiled = jax.jit(init).lower(
        arr1, arr2, _out_layouts=arg_layouts).compile()

    for i, o in zip(init_compiled._input_layouts()[0],
                    init_compiled._output_layouts()):
      self.assertEqual(i.minor_to_major, o.minor_to_major)

    with jtu.count_aot_jit_cpp_cache_miss() as init_count:
      init_out = init_compiled(arr1, arr2)
      init_compiled(arr1, arr2)
    self.assertEqual(init_count[0], 1)

    with jtu.count_aot_jit_cpp_cache_miss() as apply_count:
      apply_out = compiled_apply(*init_out)
      compiled_apply(*init_out)
    self.assertEqual(apply_count[0], 1)

    self.assertArraysEqual(init_out[0], np_inp1 * 2)
    self.assertArraysEqual(init_out[1], np_inp2 * 2)
    self.assertArraysEqual(apply_out[0], (np_inp1 * 2).T)
    self.assertArraysEqual(apply_out[1], (np_inp2 * 2).T)

  def test_specified_layouts_on_jit(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape = (8, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    sl = layout.SpecifiedLayout((0, 2, 1))

    out1 = jax.jit(lambda x: x).lower(arr, _out_layouts=sl).compile()(arr)

    compiled = jax.jit(f).lower(out1, _in_layouts=sl, _out_layouts=sl).compile()
    out2 = compiled(out1)

    self.assertEqual(compiled._input_layouts()[0][0], sl)
    self.assertEqual(compiled._output_layouts(), sl)
    self.assertArraysEqual(out2, np_inp.T)
    self.assertEqual(out2.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

  def test_default_layout(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape = (8, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    compiled = jax.jit(f).lower(
        arr,
        _in_layouts=layout.DefaultLayout(),
        _out_layouts=layout.DefaultLayout()).compile()
    out = compiled(arr)

    self.assertTupleEqual(compiled._input_layouts()[0][0].minor_to_major, (2, 1, 0))
    self.assertTupleEqual(compiled._output_layouts().minor_to_major, (2, 1, 0))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    compiled_auto = jax.jit(f).lower(arr, _in_layouts=layout.AUTO,
                                     _out_layouts=layout.AUTO).compile()
    self.assertTupleEqual(compiled_auto._input_layouts()[0][0].minor_to_major,
                          (2, 1, 0))
    self.assertTupleEqual(compiled_auto._output_layouts().minor_to_major,
                          (0, 1, 2))

  # TODO(yashkatariya): Enable after mixture of auto, default and specified
  # layouts work.
  # def test_auto_specified_default_layout_with_sharding(self):
  #   mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
  #   shape = (8, 4, 2)
  #   np_inp = np.arange(math.prod(shape)).reshape(shape)
  #   s = NamedSharding(mesh, P('x', 'y'))
  #   arr = jax.device_put(np_inp, s)

  #   def f(x, y, z):
  #     return x.T, y.T, z * 2

  #   lowered = jax.jit(f).lower(
  #       arr, arr, arr,
  #       _in_layouts=(
  #           layout.SpecifiedLayout((0, 2, 1)),
  #           layout.AUTO,
  #           layout.DefaultLayout()),
  #       _out_layouts=layout.AUTO)
  #   compiled = lowered.compile()

  #   il1, il2, il3 = compiled._input_layouts()[0]
  #   ol1, ol2, ol3 = compiled._output_layouts()

  #   self.assertTupleEqual(il1.minor_to_major, ol1.minor_to_major[::-1])
  #   self.assertTupleEqual(il2.minor_to_major, ol2.minor_to_major[::-1])
  #   self.assertEqual(il3, ol3)

  #   out1, out2, out3 = compiled(arr, arr, arr)
  #   np_inp_t = np_inp.T
  #   self.assertArraysEqual(out1, np_inp_t)
  #   self.assertArraysEqual(out2, np_inp_t)
  #   self.assertArraysEqual(out3, np_inp * 2)

  def test_in_layouts_out_layouts(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape = (8, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T
    compiled = jax.jit(f).lower(arr, _in_layouts=layout.DefaultLayout(),
                                _out_layouts=layout.AUTO).compile()
    self.assertTupleEqual(compiled._input_layouts()[0][0].minor_to_major, (1, 0))
    self.assertTupleEqual(compiled._output_layouts().minor_to_major, (0, 1))

    out = compiled(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))

  def test_sharding_and_layouts(self):
    mesh = jtu.create_global_mesh((4, 2), ('x', 'y'))
    shape = (4, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))

    compiled = jax.jit(lambda x: x.T, in_shardings=s, out_shardings=s).lower(
        np_inp, _in_layouts=layout.AUTO, _out_layouts=layout.AUTO).compile()
    out = compiled(np_inp)
    self.assertTupleEqual(compiled._input_layouts()[0][0].minor_to_major, (1, 0))
    self.assertTupleEqual(compiled._output_layouts().minor_to_major, (0, 1))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, s)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
