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
from functools import partial
from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding
from jax._src import config
from jax._src.layout import Layout, DeviceLocalLayout as DLL
from jax._src import test_util as jtu
from jax._src.lib import xla_extension_version
from jax._src.util import safe_zip
from jax.experimental.compute_on import compute_on

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class LayoutTest(jtu.JaxTestCase):

  # Remove this setUp once the released xla_extension_version is >= 308.
  def setUp(self):
    if xla_extension_version < 308 and not jtu.test_device_matches(['tpu', 'gpu']):
      self.skipTest("Layouts do not work on CPU backend yet.")
    super().setUp()

  def test_auto_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
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

    lowered_apply = jax.jit(apply, in_shardings=Layout(DLL.AUTO),
                            out_shardings=Layout(DLL.AUTO)).lower(sds1, sds2)
    compiled_apply = lowered_apply.compile()

    arg_layouts, kw_layouts = compiled_apply.input_layouts
    self.assertEmpty(kw_layouts)

    for i, o in zip(arg_layouts, compiled_apply.output_layouts):
      self.assertEqual(i.device_local_layout.major_to_minor,
                       o.device_local_layout.major_to_minor[::-1])

    init_compiled = jax.jit(
        init, out_shardings=arg_layouts).lower(sds1, sds2).compile()

    for i, o in zip(init_compiled.input_layouts[0],
                    init_compiled.output_layouts):
      self.assertEqual(i, o)

    arr1 = jax.device_put(np_inp1, s1)
    arr2 = jax.device_put(np_inp2, s2)

    with jtu.count_aot_jit_cpp_cache_miss() as init_count:
      init_out = init_compiled(arr1, arr2)
      init_compiled(arr1, arr2)
    self.assertEqual(init_count(), 1)

    self.assertEqual(init_out[0].layout, init_compiled.output_layouts[0])
    self.assertEqual(init_out[1].layout, init_compiled.output_layouts[1])

    with jtu.count_aot_jit_cpp_cache_miss() as apply_count:
      apply_out = compiled_apply(*init_out)
      compiled_apply(*init_out)
    self.assertEqual(apply_count(), 1)

    self.assertEqual(apply_out[0].layout, compiled_apply.output_layouts[0])
    self.assertEqual(apply_out[1].layout, compiled_apply.output_layouts[1])

    self.assertTupleEqual(apply_out[0].layout.device_local_layout.major_to_minor,
                          init_out[0].layout.device_local_layout.major_to_minor[::-1])
    self.assertTupleEqual(apply_out[1].layout.device_local_layout.major_to_minor,
                          init_out[1].layout.device_local_layout.major_to_minor[::-1])

    self.assertArraysEqual(init_out[0], np_inp1 * 2)
    self.assertArraysEqual(init_out[1], np_inp2 * 2)
    self.assertArraysEqual(apply_out[0], (np_inp1 * 2).T)
    self.assertArraysEqual(apply_out[1], (np_inp2 * 2).T)

  def test_default_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (4, 4, 2)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    lowered = jax.jit(f, in_shardings=None, out_shardings=None).lower(sds)
    compiled = lowered.compile()
    out = compiled(arr)

    self.assertTupleEqual(
        compiled.input_layouts[0][0].device_local_layout.major_to_minor[::-1],
        (2, 1, 0))
    self.assertTupleEqual(
        compiled.output_layouts.device_local_layout.major_to_minor[::-1],
        (2, 1, 0))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    compiled_auto = jax.jit(f, in_shardings=Layout(DLL.AUTO),
                            out_shardings=Layout(DLL.AUTO)).lower(sds).compile()
    self.assertTupleEqual(
        compiled_auto.input_layouts[0][0].device_local_layout.major_to_minor[::-1],
        (2, 1, 0))
    self.assertTupleEqual(
        compiled_auto.output_layouts.device_local_layout.major_to_minor[::-1],
        (0, 1, 2))

    with self.assertRaisesRegex(
        ValueError, "jax.jit` does not accept device-local layouts directly"):
      jax.jit(f, in_shardings=DLL.AUTO,
              out_shardings=DLL.AUTO).lower(sds).compile()

  def test_in_layouts_out_layouts(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (8, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    compiled = jax.jit(f, in_shardings=Layout(),
                       out_shardings=Layout(DLL.AUTO)).lower(arr).compile()
    self.assertTupleEqual(
        compiled.input_layouts[0][0].device_local_layout.major_to_minor[::-1],
        (1, 0))
    self.assertTupleEqual(
        compiled.output_layouts.device_local_layout.major_to_minor[::-1],
        (0, 1))

    out = compiled(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.layout, compiled.output_layouts)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))

  def test_sharding_and_layouts(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (4, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))

    compiled = jax.jit(lambda x: x.T, in_shardings=Layout(DLL.AUTO, s),
                       out_shardings=Layout(DLL.AUTO, s)).lower(np_inp).compile()
    out = compiled(np_inp)
    self.assertTupleEqual(
        compiled.input_layouts[0][0].device_local_layout.major_to_minor[::-1],
        (1, 0))
    if not jtu.test_device_matches(['cpu']):
      self.assertTupleEqual(
          compiled.output_layouts.device_local_layout.major_to_minor[::-1],
          (0, 1))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, s)

  def test_dce_in_layouts(self):
    def f(x, y, z, a, b, c):
      return z * 2, b.T

    shape = (8, 2)
    inps = [np.arange(math.prod(shape)).reshape(shape)] * 6
    compiled = jax.jit(f, in_shardings=Layout(DLL.AUTO),
                       out_shardings=Layout(DLL.AUTO)).lower(*inps).compile()
    arg_layouts, _ = compiled.input_layouts
    out1, out2 = compiled(*inps)

    compiled2 = jax.jit(f, in_shardings=arg_layouts).lower(*inps).compile()
    out3, out4 = compiled2(*inps)

    for l1, l2 in safe_zip(arg_layouts, compiled2.input_layouts[0]):
      self.assertEqual(l1, l2)

    self.assertArraysEqual(out1, out3)
    self.assertArraysEqual(out2, out4)

    arrs = [jax.device_put(i, l) for i, l in zip(inps, arg_layouts)]
    out5, out6 = jax.jit(f)(*arrs)
    self.assertArraysEqual(out1, out5)
    self.assertArraysEqual(out2, out6)

  def test_no_error_dced_args(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    shape = (8, 2)
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr1 = jax.device_put(np_inp, s)
    arr2 = jax.device_put(np_inp, s)
    arrs = [arr1, arr2]

    def f(x, y):
      return x * 2

    jf = jax.jit(f, in_shardings=Layout(DLL.AUTO, s),
                 out_shardings=Layout(DLL.AUTO, s))
    compiled = jf.lower(np_inp, np_inp).compile()
    arg_layouts, _ = compiled.input_layouts
    arrs = [jax.device_put(i, l) for i, l in zip(arrs, arg_layouts)]
    compiled(*arrs)

  def test_aot_layout_mismatch(self):
    if jtu.test_device_matches(['cpu', 'gpu']):
      # The test fails on GPU because the compilation with both input and
      # output set to auto layout is underspecified. The GPU compiler chooses
      # the default layout as the input layout and that choice does not
      # raise an exception.
      self.skipTest('This test does not work on CPU or GPU backends.')
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
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
      jax.jit(f, in_shardings=Layout(DLL.AUTO)).lower(arr)

    compiled = jax.jit(f, in_shardings=Layout(DLL.AUTO),
                       out_shardings=Layout(DLL.AUTO)).lower(sds).compile()

    with self.assertRaisesRegex(
        ValueError,
        r'Compiled object called with input layout\(s\) does'
        r' not match the layout\(s\) the computation was'
        ' compiled with'):
      compiled(arr)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message="backend and device argument")
  def test_cpu_default_backend_layout(self):
    inp = jax.device_put(np.ones((8, 8)), device=jax.devices('cpu')[0])
    out_cpu = jax.jit(jnp.dot)(inp, inp)

    jax.jit(jnp.dot, backend=jax.default_backend()).lower(
        out_cpu, out_cpu).compile()  # doesn't crash

  def test_device_put_concrete_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (8, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    compiled = jax.jit(
        lambda x: x * 2, out_shardings=Layout(DLL.AUTO)).lower(arr).compile()
    col = compiled.output_layouts

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
      Layout(compiled.output_layouts[0], None)

  def test_layout_on_sds(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    out_layout = jax.jit(jnp.sin, out_shardings=Layout(DLL.AUTO)).lower(
        arr).compile().output_layouts

    sds = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=out_layout)
    arg_layout, _ = jax.jit(lambda x: x * 2).lower(sds).compile().input_layouts
    self.assertEqual(arg_layout[0], out_layout)

    with self.assertRaisesRegex(
        TypeError,
        'DeviceLocalLayout.AUTO` cannot be used in place of a device-local'
        ' layout in a `ShapeDtypeStruct`'):
      jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=Layout(DLL.AUTO))

  def test_make_array_from_callback(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)

    layout = jax.jit(lambda x: x * 2).lower(sds).compile().output_layouts

    out = jax.make_array_from_callback(np_inp.shape, layout,
                                       lambda idx: np_inp[idx])
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.layout, layout)

    with self.assertRaisesRegex(
        TypeError,
        '`DeviceLocalLayout.AUTO` cannot be used in place of a device-local'
        ' layout'):
      jax.make_array_from_callback(np_inp.shape, Layout(DLL.AUTO, s),
                                   lambda idx: np_inp[idx])

    with self.assertRaisesRegex(
        TypeError, 'sharding should be an instance of `jax.sharding`'):
      jax.make_array_from_callback(
          np_inp.shape, Layout(None, None), lambda idx: np_inp[idx])

  def test_wsc_concrete_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    # Create a custom layout instead of using `arr.layout` to test the API.
    custom_dll = DLL(major_to_minor=(0, 1))

    @jax.jit
    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      return jax.lax.with_sharding_constraint(y, Layout(custom_dll, s))

    out = f(arr)
    self.assertEqual(out.layout.device_local_layout.major_to_minor,
                     custom_dll.major_to_minor)
    self.assertEqual(out.layout, arr.layout)
    self.assertArraysEqual(out, np_inp.T)

  def test_wsc_bfloat16_concrete_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    inp = jnp.arange(math.prod(shape), dtype=jnp.bfloat16).reshape(shape)
    arr = jax.device_put(inp, s)

    # Create a custom layout instead of using `arr.layout` to test the API.
    custom_dll = DLL(major_to_minor=(0, 1))

    @jax.jit
    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      return jax.lax.with_sharding_constraint(y, Layout(custom_dll, s))

    out = f(arr)
    self.assertEqual(out.layout.device_local_layout.major_to_minor,
                     custom_dll.major_to_minor)
    self.assertEqual(out.layout, arr.layout)
    self.assertArraysEqual(out, inp.T)

  def test_device_put_user_concrete_layout(self):

    shape = (8, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    dll = DLL(major_to_minor=(1, 0))
    s = SingleDeviceSharding(jax.devices()[0])

    out = jax.device_put(np_inp, Layout(dll, s))
    self.assertEqual(out.layout.device_local_layout.major_to_minor,
                     dll.major_to_minor)
    self.assertArraysEqual(out, np_inp)

  def test_concrete_layout_jit(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    custom_dll = DLL(major_to_minor=(0, 1))
    f = jax.jit(f, out_shardings=Layout(custom_dll, s))

    out = f(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.layout.device_local_layout.major_to_minor,
                     custom_dll.major_to_minor)

  def test_compatible_aval_error(self):
    custom_dll = DLL(major_to_minor=(0, 1, 2))
    l = Layout(custom_dll, SingleDeviceSharding(jax.devices()[0]))
    inp = np.arange(8)

    @partial(jax.jit, in_shardings=l)
    def f(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError,
        '.*Length of major_to_minor and the rank of the value should match.*'):
      f(inp)

  def test_incompatible_aval_error_device_put(self):
    custom_dll = DLL(major_to_minor=(0, 1, 2))
    l = Layout(custom_dll, SingleDeviceSharding(jax.devices()[0]))
    inp = np.arange(8)

    with self.assertRaisesRegex(
        ValueError,
        '.*Length of major_to_minor and the rank of the value should match.*'):
      jax.device_put(inp, l)

  def test_concrete_layout_in_shardings(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (16, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    custom_dll = DLL(major_to_minor=(0, 1))

    @partial(jax.jit,
             in_shardings=Layout(custom_dll, s),
             out_shardings=Layout(DLL.AUTO))
    def f(x):
      return x.T

    out = f(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.layout.device_local_layout.major_to_minor,
                     custom_dll.major_to_minor[::-1])

    custom_dll2 = DLL(major_to_minor=(1, 0))

    @partial(jax.jit, in_shardings=Layout(custom_dll2, s))
    def g(x):
      return x.T

    with self.assertRaisesRegex(
        ValueError,
        'Layout passed to jit does not match the layout on the respective arg'):
      g(arr)

  def test_in_layouts_jit_jnp_input(self):
    major_last_layout = DLL(major_to_minor=(1, 0))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    f = jax.jit(lambda x: x + 1,
                in_shardings=Layout(major_last_layout, sharding))

    arr = jnp.arange(8 * 128).reshape(8, 128)
    out = f(arr)
    self.assertArraysEqual(out, arr + 1)

    # cpp dispatch should call into shard_args from cpp.
    out2 = f(arr)
    self.assertArraysEqual(out2, arr + 1)

    np_inp = np.arange(8 * 128).reshape(8, 128)
    out3 = f(np_inp)
    self.assertArraysEqual(out3, np_inp + 1)

    # cpp dispatch should call into shard_args from cpp.
    out4 = f(np_inp)
    self.assertArraysEqual(out4, np_inp + 1)

  def test_layout_donation(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (16, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)

    custom_dll = DLL(major_to_minor=(0, 1))
    arr = jax.device_put(np_inp, Layout(custom_dll, s))

    @partial(jax.jit, in_shardings=Layout(custom_dll, s), donate_argnums=0)
    def f(x):
      return x

    f(arr)
    self.assertTrue(arr.is_deleted())

  def test_layout_donation_auto(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (128, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)

    arr = jax.device_put(np_inp, s)

    @partial(jax.jit, out_shardings=Layout(DLL.AUTO), donate_argnums=0)
    def f(x):
      return x * x

    f(arr)
    self.assertTrue(arr.is_deleted())

  def test_layout_donation_matching_in_and_out(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (128, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)

    custom_dll = DLL(major_to_minor=(0, 1))
    l = Layout(custom_dll, s)
    arr = jax.device_put(np_inp, l)

    @partial(jax.jit, in_shardings=l, out_shardings=l, donate_argnums=0)
    def f(x):
      return x * x

    f(arr)
    self.assertTrue(arr.is_deleted())

  @jtu.skip_on_devices('cpu', 'gpu')
  def test_layout_donation_mismatching_in_and_out_fails(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (16*2, 32016*2)
    np_inp = np.arange(math.prod(shape), dtype=jnp.bfloat16).reshape(shape)

    custom_dll1 = DLL(major_to_minor=(1, 0), _tiling=((8,128), (2,1)))
    l1 = Layout(custom_dll1, s)
    arr = jax.device_put(np_inp, s)

    @partial(jax.jit, out_shardings=l1, donate_argnums=0)
    def f(x):
      return x * x

    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    f.lower(sds).compile()(arr)
    self.assertFalse(arr.is_deleted())

  def test_donation_error_on_auto(self):
    @partial(jax.jit, donate_argnums=0, in_shardings=Layout(DLL.AUTO))
    def f(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError, ".*Did you mean to set the.*output layout.*AUTO.*"):
      f(jnp.arange(8))

    @partial(jax.jit, donate_argnums=0, out_shardings=Layout(DLL.AUTO))
    def g(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError, ".*Did you mean to set the.*input layout.*AUTO.*"):
      g(jnp.arange(8))

  def test_sparsecore_compute(self):
    if not (jax.devices()[0].device_kind == 'TPU v5' or
            jtu.is_device_tpu_at_least(6)):
      self.skipTest('Does not have a sparsecore present')
    shape = (128, 128)
    inp = jnp.arange(math.prod(shape)).reshape(shape)

    dll = DLL(major_to_minor=(0, 1), _tiling=((8,),))
    s = SingleDeviceSharding(jax.devices()[0])
    sparse_layout = Layout(dll, s)
    sparecore_arr = jax.device_put(inp, sparse_layout)
    dense_layout = Layout(DLL(major_to_minor=(0, 1)), s)

    @compute_on('tpu_sparsecore')
    @jax.jit
    def sparsecore_compute(x):
      return x * x

    @partial(jax.jit, out_shardings=(dense_layout, sparse_layout))
    def f(x, y):
      return x * 2, sparsecore_compute(y)

    f(inp, sparecore_arr)

  def test_sparsecore_compute_twice(self):
    if not (
        jax.devices()[0].device_kind == 'TPU v5'
        or jtu.is_device_tpu_at_least(6)
    ):
      self.skipTest('Does not have a sparsecore present')
    shape = (4096, 8)
    inp = jnp.arange(math.prod(shape)).reshape(shape)

    dll = DLL(major_to_minor=(0, 1), _tiling=((8,),))
    s = SingleDeviceSharding(jax.devices()[0])
    sparse_layout = Layout(dll, s)
    sparecore_arr = jax.device_put(inp, sparse_layout)

    @compute_on('tpu_sparsecore')
    @jax.jit
    def sparsecore_multiply(x, y):
      return x * y

    @compute_on('tpu_sparsecore')
    @jax.jit
    def sparsecore_add(x, y):
      return x + y

    @partial(jax.jit, donate_argnums=0, out_shardings=sparse_layout)
    def f(x):
      return sparsecore_multiply(sparsecore_add(x, x) + 1, x)

    f(sparecore_arr)

  def test_sparsecore_and_host_compute(self):
    if not (
        jax.devices()[0].device_kind == 'TPU v5'
        or jtu.is_device_tpu_at_least(6)
    ):
      self.skipTest('Does not have a sparsecore present')
    shape = (128, 128)
    inp = jnp.arange(math.prod(shape)).reshape(shape)
    s = SingleDeviceSharding(jax.devices()[0])

    sparse_dll = DLL(major_to_minor=(0, 1), _tiling=((8,),))
    sparse_layout = Layout(sparse_dll, s)
    sparecore_arr = jax.device_put(inp, sparse_layout)

    host_dll = DLL(major_to_minor=(0, 1), _tiling=((1,),))
    host_layout = Layout(host_dll, s)
    host_arr = jax.device_put(inp, host_layout)

    @compute_on('tpu_sparsecore')
    @jax.jit
    def sparsecore_compute(x):
      return x * x

    @compute_on('device_host')
    @jax.jit
    def host_compute(x):
      return x + x

    @partial(
        jax.jit,
        in_shardings=(sparse_layout, host_layout),
        out_shardings=(sparse_layout, host_layout),
    )
    def f(x, y):
      return sparsecore_compute(x), host_compute(y)

    f(sparecore_arr, host_arr)

  def test_cpp_layout_cache_miss(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (16, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    arr_m2m = arr.layout.device_local_layout.major_to_minor
    custom_layout = Layout(DLL(major_to_minor=arr_m2m[::-1]), s)
    arr2 = jax.device_put(np_inp, custom_layout)

    @jax.jit
    def f(x):
      return x @ x.T

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = f(arr)
      out2 = f(arr2)
    self.assertEqual(count(), 2)

    self.assertArraysEqual(out, np_inp @ np_inp.T)
    self.assertArraysEqual(out2, np_inp @ np_inp.T)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
