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

import functools
import math
import re

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.layout import LayoutMode, use_layout_mode
from jax._src.sharding_impls import make_single_device_sharding
from jax._src.util import safe_zip
from jax.experimental.layout import (Format, Layout, with_layout_constraint,
                                     explicit_layout)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class LayoutTest(jtu.JaxTestCase):

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

    np_inp1 = np.arange(math.prod(shape1), dtype=np.int32).reshape(shape1)
    np_inp2 = np.arange(math.prod(shape2), dtype=np.int32).reshape(shape2)
    sds1 = jax.ShapeDtypeStruct(np_inp1.shape, np_inp1.dtype, sharding=s1)
    sds2 = jax.ShapeDtypeStruct(np_inp2.shape, np_inp2.dtype, sharding=s2)

    lowered_apply = jax.jit(apply, in_shardings=Format(Layout.AUTO),
                            out_shardings=Format(Layout.AUTO)).lower(sds1, sds2)
    compiled_apply = lowered_apply.compile()

    arg_formats, kw_layouts = compiled_apply.input_formats
    self.assertEmpty(kw_layouts)

    for i, o in zip(arg_formats, compiled_apply.output_formats):
      self.assertEqual(i.layout.major_to_minor,
                       o.layout.major_to_minor[::-1])

    init_compiled = jax.jit(
        init, out_shardings=arg_formats).lower(sds1, sds2).compile()

    for i, o in zip(init_compiled.input_formats[0],
                    init_compiled.output_formats):
      self.assertEqual(i, o)

    arr1 = jax.device_put(np_inp1, s1)
    arr2 = jax.device_put(np_inp2, s2)

    with jtu.count_aot_jit_cpp_cache_miss() as init_count:
      init_out = init_compiled(arr1, arr2)
      init_compiled(arr1, arr2)
    self.assertEqual(init_count(), 1)

    self.assertEqual(init_out[0].format, init_compiled.output_formats[0])
    self.assertEqual(init_out[1].format, init_compiled.output_formats[1])

    with jtu.count_aot_jit_cpp_cache_miss() as apply_count:
      apply_out = compiled_apply(*init_out)
      compiled_apply(*init_out)
    self.assertEqual(apply_count(), 1)

    self.assertEqual(apply_out[0].format, compiled_apply.output_formats[0])
    self.assertEqual(apply_out[1].format, compiled_apply.output_formats[1])

    self.assertTupleEqual(apply_out[0].format.layout.major_to_minor,
                          init_out[0].format.layout.major_to_minor[::-1])
    self.assertTupleEqual(apply_out[1].format.layout.major_to_minor,
                          init_out[1].format.layout.major_to_minor[::-1])

    self.assertArraysEqual(init_out[0], np_inp1 * 2)
    self.assertArraysEqual(init_out[1], np_inp2 * 2)
    self.assertArraysEqual(apply_out[0], (np_inp1 * 2).T)
    self.assertArraysEqual(apply_out[1], (np_inp2 * 2).T)

  def test_default_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (4, 4, 2)
    np_inp = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    lowered = jax.jit(f, in_shardings=None, out_shardings=None).lower(sds)
    compiled = lowered.compile()
    out = compiled(arr)

    self.assertTupleEqual(
        compiled.input_formats[0][0].layout.major_to_minor[::-1],
        (2, 1, 0))
    self.assertTupleEqual(
        compiled.output_formats.layout.major_to_minor[::-1],
        (2, 1, 0))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, NamedSharding(mesh, P(None, 'y', 'x')))

    compiled_auto = jax.jit(f, in_shardings=Format(Layout.AUTO),
                            out_shardings=Format(Layout.AUTO)).lower(sds).compile()
    self.assertTupleEqual(
        compiled_auto.input_formats[0][0].layout.major_to_minor[::-1],
        (2, 1, 0))
    self.assertTupleEqual(
        compiled_auto.output_formats.layout.major_to_minor[::-1],
        (0, 1, 2))

    with self.assertRaisesRegex(
        ValueError, "jax.jit` does not accept device-local layouts directly"):
      jax.jit(f, in_shardings=Layout.AUTO,
              out_shardings=Layout.AUTO).lower(sds).compile()

  def test_in_layouts_out_layouts(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (8, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    compiled = jax.jit(f, in_shardings=Format(),
                       out_shardings=Format(Layout.AUTO)).lower(arr).compile()
    self.assertTupleEqual(
        compiled.input_formats[0][0].layout.major_to_minor[::-1],
        (1, 0))
    self.assertTupleEqual(
        compiled.output_formats.layout.major_to_minor[::-1],
        (0, 1))

    out = compiled(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.format, compiled.output_formats)
    self.assertEqual(out.sharding, NamedSharding(mesh, P('y', 'x')))

  def test_sharding_and_layouts(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (4, 8)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    s = NamedSharding(mesh, P('x', 'y'))

    compiled = jax.jit(lambda x: x.T, in_shardings=Format(Layout.AUTO, s),
                       out_shardings=Format(Layout.AUTO, s)).lower(np_inp).compile()
    out = compiled(np_inp)
    self.assertTupleEqual(
        compiled.input_formats[0][0].layout.major_to_minor[::-1],
        (1, 0))
    if not jtu.test_device_matches(['cpu']):
      self.assertTupleEqual(
          compiled.output_formats.layout.major_to_minor[::-1],
          (0, 1))
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.sharding, s)

  def test_dce_in_layouts(self):
    def f(x, y, z, a, b, c):
      return z * 2, b.T

    shape = (8, 2)
    inps = [np.arange(math.prod(shape)).reshape(shape)] * 6
    compiled = jax.jit(f, in_shardings=Format(Layout.AUTO),
                       out_shardings=Format(Layout.AUTO)).lower(*inps).compile()
    arg_formats, _ = compiled.input_formats
    out1, out2 = compiled(*inps)

    compiled2 = jax.jit(f, in_shardings=arg_formats).lower(*inps).compile()
    out3, out4 = compiled2(*inps)

    for l1, l2 in safe_zip(arg_formats, compiled2.input_formats[0]):
      self.assertEqual(l1, l2)

    self.assertArraysEqual(out1, out3)
    self.assertArraysEqual(out2, out4)

    arrs = [jax.device_put(i, l) for i, l in zip(inps, arg_formats)]
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

    jf = jax.jit(f, in_shardings=Format(Layout.AUTO, s),
                 out_shardings=Format(Layout.AUTO, s))
    compiled = jf.lower(np_inp, np_inp).compile()
    arg_formats, _ = compiled.input_formats
    arrs = [jax.device_put(i, l) for i, l in zip(arrs, arg_formats)]
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
    np_inp = np.arange(math.prod(shape), dtype=np.int32).reshape(shape)
    s = NamedSharding(mesh, P('x'))

    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return (x * 2).T

    with self.assertRaisesRegex(
        ValueError,
        'Layout passed to jit does not match the layout on the respective arg'):
      jax.jit(f, in_shardings=Format(Layout.AUTO)).lower(arr)

    compiled = jax.jit(f, in_shardings=Format(Layout.AUTO),
                       out_shardings=Format(Layout.AUTO)).lower(sds).compile()

    with self.assertRaisesRegex(
        ValueError,
        r'Computation was compiled for input layouts that disagree with the '
        r'layouts of arguments passed to it.'):
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
        lambda x: x * 2, out_shardings=Format(Layout.AUTO)).lower(arr).compile()
    col = compiled.output_formats

    out = jax.device_put(np_inp, col)
    self.assertEqual(out.format, col)
    self.assertArraysEqual(out, np_inp)
    for s in out.addressable_shards:
      self.assertEqual(out.format.layout,
                       s.data.format.layout)

  def test_device_put_non_concrete_layout_error(self):
    np_inp = np.arange(16).reshape(8, 2)

    l1 = Format(Layout.AUTO, make_single_device_sharding(jax.devices()[0]))
    with self.assertRaisesRegex(
        ValueError, 'sharding and layout.*should be concrete'):
      jax.device_put(np_inp, l1)

    l2 = Format(Layout.AUTO)
    with self.assertRaisesRegex(
        ValueError, 'sharding and layout.*should be concrete'):
      jax.device_put(np_inp, l2)

    l3 = Format(None, make_single_device_sharding(jax.devices()[0]))
    out = jax.device_put(np_inp, l3)
    self.assertArraysEqual(out, np_inp)
    self.assertTrue(out._committed)

  def invalid_layout_spec(self):
    x = np.arange(8)
    compiled = jax.jit(lambda x: x).lower(x).compile()
    with self.assertRaisesRegex(
        ValueError, 'Sharding has to be concrete when layout.*'):
      Format(compiled.output_formats[0], None)

  def test_layout_on_sds(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    out_format = jax.jit(jnp.sin, out_shardings=Format(Layout.AUTO)).lower(
        arr).compile().output_formats

    sds = jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=out_format)
    arg_format, _ = jax.jit(lambda x: x * 2).lower(sds).compile().input_formats
    self.assertEqual(arg_format[0], out_format)

    with self.assertRaisesRegex(
        TypeError,
        'Layout.AUTO` cannot be used in place of a device-local'
        ' layout in a `ShapeDtypeStruct`'):
      jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=Format(Layout.AUTO))

  def test_make_array_from_callback(self):
    mesh = jtu.create_mesh((2, 1), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16, dtype=np.int32).reshape(8, 2)
    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)

    format = jax.jit(lambda x: x * 2).lower(sds).compile().output_formats

    out = jax.make_array_from_callback(np_inp.shape, format,
                                       lambda idx: np_inp[idx])
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(out.format, format)

    with self.assertRaisesRegex(
        TypeError,
        '`Layout.AUTO` cannot be used in place of a device-local'
        ' layout'):
      jax.make_array_from_callback(np_inp.shape, Format(Layout.AUTO, s),
                                   lambda idx: np_inp[idx])

    with self.assertRaisesRegex(
        TypeError, 'sharding should be an instance of `jax.sharding`'):
      jax.make_array_from_callback(
          np_inp.shape, Format(None, None), lambda idx: np_inp[idx])

  def test_wsc_concrete_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    # Create a custom layout instead of using `arr.layout` to test the API.
    custom_dll = Layout(major_to_minor=(0, 1))

    @jax.jit
    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      return jax.lax.with_sharding_constraint(y, Format(custom_dll, s))

    out = f(arr)
    self.assertEqual(out.format.layout.major_to_minor,
                     custom_dll.major_to_minor)
    self.assertEqual(out.format, arr.format)
    self.assertArraysEqual(out, np_inp.T)

  def test_wsc_bfloat16_concrete_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (64, 128)
    s = NamedSharding(mesh, P('x'))
    inp = jnp.arange(math.prod(shape), dtype=jnp.bfloat16).reshape(shape)
    arr = jax.device_put(inp, s)

    # Create a custom layout instead of using `arr.layout` to test the API.
    custom_dll = Layout(major_to_minor=(0, 1))

    @jax.jit
    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      return jax.lax.with_sharding_constraint(y, Format(custom_dll, s))

    out = f(arr)
    self.assertEqual(out.format.layout.major_to_minor,
                     custom_dll.major_to_minor)
    self.assertEqual(out.format, arr.format)
    self.assertArraysEqual(out, inp.T)

  def test_device_put_user_concrete_layout(self):
    shape = (8, 128)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    dll = Layout(major_to_minor=(1, 0))
    s = make_single_device_sharding(jax.devices()[0])

    out = jax.device_put(np_inp, Format(dll, s))
    self.assertEqual(out.format.layout.major_to_minor,
                     dll.major_to_minor)
    self.assertArraysEqual(out, np_inp)

  def test_device_put_user_concrete_layout_multi_device(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    jnp_inp = jnp.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    custom_format = Format(Layout(major_to_minor=(0, 1)), s)
    out1 = jax.device_put(arr, custom_format)

    with jax.set_mesh(mesh):
      out2 = jax.device_put(arr, custom_format)
      out3 = jax.device_put(jnp_inp, custom_format)
      out4 = jax.device_put(np_inp, custom_format)

    for o in [out1, out2, out3, out4]:
      self.assertArraysEqual(o, np_inp)
      self.assertEqual(o.format.layout.major_to_minor,
                       custom_format.layout.major_to_minor)

  def test_concrete_layout_jit(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    def f(x):
      return x.T

    custom_dll = Layout(major_to_minor=(0, 1))
    f = jax.jit(f, out_shardings=Format(custom_dll, s))

    out = f(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.format.layout.major_to_minor,
                     custom_dll.major_to_minor)

  def test_compatible_aval_error(self):
    custom_dll = Layout(major_to_minor=(0, 1, 2))
    l = Format(custom_dll, make_single_device_sharding(jax.devices()[0]))
    inp = np.arange(8)

    @jax.jit(in_shardings=l)
    def f(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError,
        '.*Length of major_to_minor and the rank of the value should match.*'):
      f(inp)

  def test_incompatible_aval_error_device_put(self):
    custom_dll = Layout(major_to_minor=(0, 1, 2))
    l = Format(custom_dll, make_single_device_sharding(jax.devices()[0]))
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

    custom_dll = Layout(major_to_minor=(0, 1))

    @jax.jit(
             in_shardings=Format(custom_dll, s),
             out_shardings=Format(Layout.AUTO))
    def f(x):
      return x.T

    out = f(arr)
    self.assertArraysEqual(out, np_inp.T)
    self.assertEqual(out.format.layout.major_to_minor,
                     custom_dll.major_to_minor[::-1])

    custom_dll2 = Layout(major_to_minor=(1, 0))

    @jax.jit(in_shardings=Format(custom_dll2, s))
    def g(x):
      return x.T

    with self.assertRaisesRegex(
        ValueError,
        'Layout passed to jit does not match the layout on the respective arg'):
      g(arr)

  def test_in_layouts_jit_jnp_input(self):
    major_last_layout = Layout(major_to_minor=(1, 0))
    sharding = make_single_device_sharding(jax.devices()[0])

    f = jax.jit(lambda x: x + 1,
                in_shardings=Format(major_last_layout, sharding))

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

    custom_dll = Layout(major_to_minor=(0, 1))
    arr = jax.device_put(np_inp, Format(custom_dll, s))

    @jax.jit(in_shardings=Format(custom_dll, s), donate_argnums=0)
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

    @jax.jit(out_shardings=Format(Layout.AUTO), donate_argnums=0)
    def f(x):
      return x * x

    f(arr)
    self.assertTrue(arr.is_deleted())

  def test_layout_donation_matching_in_and_out(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (128, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)

    custom_dll = Layout(major_to_minor=(0, 1))
    l = Format(custom_dll, s)
    arr = jax.device_put(np_inp, l)

    @jax.jit(in_shardings=l, out_shardings=l, donate_argnums=0)
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

    tiling = ((8, 128), (2, 1))
    custom_dll1 = Layout(major_to_minor=(1, 0), tiling=tiling)
    l1 = Format(custom_dll1, s)
    arr = jax.device_put(np_inp, s)

    @jax.jit(out_shardings=l1, donate_argnums=0)
    def f(x):
      return x * x

    sds = jax.ShapeDtypeStruct(np_inp.shape, np_inp.dtype, sharding=s)
    f.lower(sds).compile()(arr)
    self.assertFalse(arr.is_deleted())

  def test_donation_error_on_auto(self):
    @jax.jit(donate_argnums=0, in_shardings=Format(Layout.AUTO))
    def f(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError, ".*Did you mean to set the.*output layout.*AUTO.*"):
      f(jnp.arange(8))

    @jax.jit(donate_argnums=0, out_shardings=Format(Layout.AUTO))
    def g(x):
      return x * 2

    with self.assertRaisesRegex(
        ValueError, ".*Did you mean to set the.*input layout.*AUTO.*"):
      g(jnp.arange(8))

  def test_cpp_layout_cache_miss(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (16, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    arr_m2m = arr.format.layout.major_to_minor
    custom_format = Format(Layout(major_to_minor=arr_m2m[::-1]), s)
    arr2 = jax.device_put(np_inp, custom_format)

    @jax.jit
    def f(x):
      return x @ x.T

    with jtu.count_pjit_cpp_cache_miss() as count:
      out = f(arr)
      out2 = f(arr2)
    self.assertEqual(count(), 2)

    self.assertArraysEqual(out, np_inp @ np_inp.T)
    self.assertArraysEqual(out2, np_inp @ np_inp.T)

  def test_layout_donation_with_default_layout(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (16, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)
    out_format = Format(arr.format.layout, s)

    @jax.jit(out_shardings=out_format, donate_argnums=0)
    def f(x):
      return x * 2

    lowered_text = f.lower(arr).as_text()
    self.assertIn('tf.aliasing_output = 0', lowered_text)
    self.assertNotIn('jax.buffer_donor', lowered_text)

    out = f(arr)
    self.assertArraysEqual(out, np_inp * 2)
    self.assertEqual(out.format, out_format)

  def test_with_layout_constraint(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Only works for TPU')
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    # Create a custom layout instead of using `arr.layout` to test the API.
    custom_dll = Layout(major_to_minor=arr.format.layout.major_to_minor[::-1])

    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      y = with_layout_constraint(y, custom_dll)
      return y * 2

    f(arr)  # doesn't crash

    f = jax.jit(f)
    out = f(arr)
    self.assertEqual(out.format.layout.major_to_minor,
                     custom_dll.major_to_minor)
    self.assertArraysEqual(out, np_inp.T * 2)

    lowered_text = f.lower(arr).as_text()
    self.assertIn('LayoutConstraint', lowered_text)

  def test_with_layout_constraint_with_tiling(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Only works for TPU')

    if not jtu.stablehlo_version_at_least('1.18.0'):
      self.skipTest('Requires stablehlo 1.18.0 or higher')

    shape = (64, 256)
    np_inp = np.arange(math.prod(shape), dtype=jnp.bfloat16).reshape(shape)
    arr = jax.device_put(np_inp)

    # Create a custom layout instead of using `arr.layout` to test the API.
    custom_dll = Layout(
        major_to_minor=arr.format.layout.major_to_minor[::-1],
        tiling=((16, 128), (2, 1)),
    )

    @jax.jit
    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      return with_layout_constraint(y, custom_dll) * 2

    out = f(arr)
    self.assertEqual(
        out.format.layout.major_to_minor, custom_dll.major_to_minor
    )
    self.assertArraysEqual(out, np_inp.T * 2)

    lowered_text = f.lower(arr).as_text()
    self.assertIn('LayoutConstraint', lowered_text)
    self.assertIn(
        'result_tilings = [[dense<[16, 128]> : tensor<2xindex>, dense<[2, 1]> :'
        ' tensor<2xindex>]]',
        lowered_text,
    )

  def test_with_layout_constraint_vmap(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Only works for TPU')
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    shape = (16, 128)
    s = NamedSharding(mesh, P('x'))
    np_inp = np.arange(math.prod(shape)).reshape(shape)
    arr = jax.device_put(np_inp, s)

    def f(x):
      y = x.T
      # Constrain `y` to the original layout of `arr` because without it,
      # the layout of `y` would be the transpose of `arr`.
      y = with_layout_constraint(y, Layout(major_to_minor=(0,)))
      return y * 2

    out = jax.jit(jax.vmap(f))(arr)
    self.assertEqual(out.format.layout.major_to_minor, (0, 1))

  def test_eval_shape_format(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = NamedSharding(mesh, P('x', 'y'))
    shape = (128, 16)
    np_inp = np.arange(math.prod(shape)).reshape(shape)

    custom_dll = Layout(major_to_minor=(0, 1))
    l = Format(custom_dll, s)
    arr = jax.device_put(np_inp, l)

    @jax.jit(in_shardings=l, out_shardings=l)
    def f(x):
      return x * x

    out = jax.eval_shape(f, arr)
    self.assertEqual(out.format, l)
    self.assertEqual(out.sharding, s)

  def test_valid_custom_layout_after_copy_across_clients(self):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Only works for TPU')

    custom_dll = Layout(major_to_minor=(1, 0))

    cpu_sharding = make_single_device_sharding(
        jax.local_devices(backend='cpu')[0])
    cpu_format = Format(custom_dll, cpu_sharding)
    cpu_array = jax.device_put(np.ones((128, 8)), cpu_format)

    mesh = jtu.create_mesh((1, 1), ('x', 'y'))
    tpu_sharding = jax.sharding.NamedSharding(mesh, P())
    tpu_format = Format(custom_dll, tpu_sharding)

    copied_tpu_array = jax.device_put(cpu_array, tpu_format.sharding)
    canonical_tpu_array = jax.device_put(np.ones((128, 8)), tpu_format)
    self.assertEqual(
        copied_tpu_array.format.layout, canonical_tpu_array.format.layout)

  @parameterized.named_parameters(
      ('device_to_pinned_host', 'device', 'pinned_host'),
      ('pinned_host_to_device', 'pinned_host', 'device'),
      ('device_to_unpinned_host', 'device', 'unpinned_host'),
      ('unpinned_host_to_device', 'unpinned_host', 'device'),
      ('pinned_host_to_unpinned_host', 'pinned_host', 'unpinned_host'),
      ('unpinned_host_to_pinned_host', 'unpinned_host', 'pinned_host'),
  )
  def test_valid_layout_after_copy_across_memories(
      self, src_memory_kind, dst_memory_kind):
    if not jtu.test_device_matches(['tpu']):
      self.skipTest('Only works for TPU')
    custom_dll = Layout(major_to_minor=(1, 0))

    mesh = jtu.create_mesh((1, 1), ('x', 'y'))
    src_tpu_sharding = jax.sharding.NamedSharding(
        mesh, P(), memory_kind=src_memory_kind)
    dst_tpu_sharding = jax.sharding.NamedSharding(
        mesh, P(), memory_kind=dst_memory_kind)

    # TPU unpinned_host memories do not support custom layouts.
    if src_memory_kind == 'unpinned_host':
      src_tpu_format = src_tpu_sharding
    else:
      src_tpu_format = Format(custom_dll, src_tpu_sharding)
    if dst_memory_kind == 'unpinned_host':
      dst_tpu_format = dst_tpu_sharding
    else:
      dst_tpu_format = Format(custom_dll, dst_tpu_sharding)

    tpu_array = jax.device_put(np.ones((128, 8)), src_tpu_format)

    copied_tpu_array = jax.device_put(tpu_array, dst_tpu_sharding)
    canonical_tpu_array = jax.device_put(np.ones((128, 8)), dst_tpu_format)
    self.assertEqual(
        copied_tpu_array.format.layout, canonical_tpu_array.format.layout)

  @jtu.run_on_devices('tpu')
  @jtu.with_explicit_mesh((2,), 'x')
  def test_reshard_layout_constraint(self, mesh):
    arr1 = jax.device_put(np.arange(128 * 256).reshape(128, 256), P(None, 'x'))
    arr2 = jax.device_put(np.arange(128 * 256).reshape(256, 128), P('x', None))

    @jax.jit
    def f(x, y):
      z = jnp.dot(x, y, out_sharding=P(unreduced={'x'}))
      z = with_layout_constraint(z, Layout(major_to_minor=(1, 0)))
      z = z + z
      return with_layout_constraint(z, Layout(major_to_minor=(1, 0)))

    out = f(arr1, arr2)
    self.assertArraysEqual(jax.reshard(out, P()),
                           jnp.dot(arr1, arr2, out_sharding=P()) * 2)
    self.assertEqual(out.sharding,
                     NamedSharding(mesh, P(None, None, unreduced={'x'})))
    self.assertEqual(out.format.layout.major_to_minor, (1, 0))
    self.assertNotIn('all-reduce(', f.lower(arr1, arr2).compile().as_text())

  def test_host_auto_layout(self):
    if jtu.test_device_matches(['cpu', 'gpu']):
      self.skipTest('This test does not work on CPU or GPU backends.')

    if not jtu.is_libtpu_at_least('0.0.49'):
      self.skipTest('Needs a newer libtpu')

    mesh = jtu.create_mesh((jax.device_count(),), ('data',))
    shape = (5, 2, 1004, 512, 36)
    layout = Layout((0, 1, 4, 3, 2))

    device_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    host_sharding = device_sharding.with_memory_kind('pinned_host')
    host_format = Format(layout=layout, sharding=host_sharding)

    @functools.partial(jax.jit, out_shardings=host_format)
    def test_fun(y):
      return with_layout_constraint(y * 2, host_format.layout)

    @functools.partial(
        jax.jit,
        out_shardings=Format(layout=Layout.AUTO, sharding=host_format.sharding),
    )
    def outer_fun(y):
      return test_fun(y)

    x = jnp.ones(shape, dtype=jnp.float32)
    compiled = outer_fun.lower(x).compile()

    hlo = compiled.as_text()
    match = re.search(r'->\s*f32\[5,2,1004,512,36\]\{([^}]+)\}', hlo)
    self.assertIsNotNone(match, 'Could not find output layout in HLO')
    layout_str = match.group(1)
    self.assertIn('2,3,4,1,0', layout_str)
    self.assertIn('S(5)', layout_str)

  def test_trace_cache_hit_default_layout_tracing_mode(self):
    x = jnp.arange(4.0)

    @jax.jit
    def f(x):
      return x

    with jtu.count_jit_tracing_cache_miss() as count:
      f(x)
      with use_layout_mode(LayoutMode.AUTO):
        f(x)
    self.assertEqual(count(), 1)


class LayoutInTypesTest(jtu.JaxTestCase):

  def test_unop_layout(self):
    if not jtu.is_libtpu_at_least('0.0.50'):
      self.skipTest('Requires libtpu >= 0.0.50')

    arr = jnp.arange(16.).reshape(2, 8)
    l = Layout.for_array(arr)
    ex_l = (Layout((0, 1), ((2, 128),)) if jtu.test_device_matches(['tpu'])
            else Layout((0, 1), ()))
    self.assertEqual(l, ex_l)

    @jax.jit
    @explicit_layout(in_layouts=l)
    def f(x):
      self.assertEqual(x.aval.layout, l)
      y = jnp.sin(x)
      self.assertEqual(y.aval.layout, l)
      return y

    out = f(arr)
    self.assertEqual(out.format, arr.format)
    self.assertArraysEqual(out, jnp.sin(arr))

  def test_naryop_layout(self):
    if not jtu.is_libtpu_at_least('0.0.50'):
      self.skipTest('Requires libtpu >= 0.0.50')

    arr1 = jnp.arange(16., dtype=np.float32).reshape(2, 8)
    arr2 = jnp.arange(16., dtype=np.float32).reshape(2, 8)
    l = Layout.for_array(arr1)
    ex_l = (Layout((0, 1), ((2, 128),)) if jtu.test_device_matches(['tpu'])
            else Layout((0, 1), ()))
    self.assertEqual(l, ex_l)

    @jax.jit
    @explicit_layout(in_layouts=(l, l))
    def f(x, y):
      self.assertEqual(x.aval.layout, l)
      self.assertEqual(y.aval.layout, l)
      z = x + y
      self.assertEqual(z.aval.layout, l)
      w = jax.lax.add(np.float32(1.0), z)
      self.assertEqual(w.aval.layout, l)
      return w

    out = f(arr1, arr2)
    self.assertEqual(out.format, arr1.format)
    self.assertArraysEqual(out, arr1 + arr2 + 1.0)

    l_transposed = Layout(l.major_to_minor[::-1], l.tiling)

    @jax.jit
    @explicit_layout(in_layouts=(l, l_transposed))
    def g(x, y):
      return x + y

    with self.assertRaisesRegex(
        ValueError, 'layout of all inputs passed to `add` must be the same'):
      g(arr1, arr2)

  def test_dot_2d_layout(self):
    arr1 = jnp.arange(64.).reshape(8, 8)
    arr2 = jnp.arange(64.).reshape(8, 8)
    l = arr1.format.layout
    self.assertEqual(l.major_to_minor, (0, 1))

    l_t = Layout((1, 0), l.tiling)
    arr2_t = jax.device_put(arr2, Format(l_t, arr2.sharding))

    @jax.jit
    @explicit_layout(in_layouts=(l, l_t))
    def f(x, y):
      self.assertEqual(x.aval.layout, Layout((0, 1), l.tiling))
      self.assertEqual(y.aval.layout, Layout((1, 0), l.tiling))
      z = x @ y
      self.assertEqual(z.aval.layout, Layout((0, 1), l.tiling))
      return z

    lowered_text = f.lower(arr1, arr2_t).as_text()
    self.assertEqual(lowered_text.count('LayoutConstraint'), 3)

    out = f(arr1, arr2_t)
    self.assertEqual(out.format.layout, Layout((0, 1), l.tiling))
    self.assertArraysAllClose(out, arr1 @ arr2)

  def test_dot_multiple_contracting_dims_layout(self):
    # lhs (M=8, K1=8, K2=8) with layout (1, 0, 2) -> K1 major, (M, K2) minor
    # rhs (K1=8, K2=8, N=8) with layout (0, 1, 2) or (0, 2, 1) -> K1 major,
    # (K2, N) or (N, K2) minor
    arr3 = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape(8, 8, 8)
    tiling = arr3.format.layout.tiling
    l_lhs = Layout((1, 0, 2), tiling)
    lhs_3d = jax.device_put(arr3, Format(l_lhs, arr3.sharding))

    for l_rhs in [Layout((0, 1, 2), tiling), Layout((0, 2, 1), tiling)]:
      rhs_3d = jax.device_put(arr3, Format(l_rhs, arr3.sharding))

      @jax.jit
      @explicit_layout(in_layouts=(l_lhs, l_rhs))
      def dot_2_contract(x, y):
        out = jnp.einsum('mab,abn->mn', x, y)
        self.assertEqual(out.aval.layout, Layout((0, 1), tiling))
        return out

      out_2c = dot_2_contract(lhs_3d, rhs_3d)
      self.assertEqual(out_2c.format.layout, Layout((0, 1), tiling))
      self.assertArraysAllClose(out_2c, jnp.einsum('mab,abn->mn', arr3, arr3))

  def test_dot_3d_output_layout(self):
    arr2 = jnp.arange(64., dtype=jnp.float32).reshape(8, 8)
    arr3 = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape(8, 8, 8)
    tiling = arr3.format.layout.tiling
    l_2d = Layout((0, 1), tiling)
    l_102 = Layout((1, 0, 2), tiling)
    l_012 = Layout((0, 1, 2), tiling)
    l_021 = Layout((0, 2, 1), tiling)

    lhs_102 = jax.device_put(arr3, Format(l_102, arr3.sharding))
    lhs_012 = jax.device_put(arr3, Format(l_012, arr3.sharding))
    rhs_021 = jax.device_put(arr3, Format(l_021, arr3.sharding))

    # 3D output from 2 non-contracting dims on lhs:
    # lhs (M=8, P=8, K=8) with layout (1, 0, 2) -> P major, (M, K) minor
    # rhs (K=8, N=8) with layout (0, 1) -> (K, N) minor
    # out (M=8, P=8, N=8) -> P (dim 1) stays major, (M, N) (dims 0, 2) minor -> (1, 0, 2)
    @jax.jit
    @explicit_layout(in_layouts=(l_102, l_2d))
    def dot_3d_out(x, y):
      out = jnp.einsum('mpk,kn->mpn', x, y)
      self.assertEqual(out.aval.layout, l_102)
      return out

    out_3d = dot_3d_out(lhs_102, arr2)
    self.assertEqual(out_3d.format.layout, l_102)
    self.assertArraysAllClose(out_3d, jnp.einsum('mpk,kn->mpn', arr3, arr2))

    # 3D output with batch dim:
    # lhs (B=8, M=8, K=8) with layout (0, 1, 2) -> B major, (M, K) minor
    # rhs (B=8, K=8, N=8) with layout (0, 2, 1) -> B major, (N, K) minor
    # out (B=8, M=8, N=8) -> B (dim 0) stays major, (M, N) (dims 1, 2) minor -> (0, 1, 2)
    @jax.jit
    @explicit_layout(in_layouts=(l_012, l_021))
    def dot_batch_3d_out(x, y):
      out = jnp.einsum('bmk,bkn->bmn', x, y)
      self.assertEqual(out.aval.layout, l_012)
      return out

    out_b3d = dot_batch_3d_out(lhs_012, rhs_021)
    self.assertEqual(out_b3d.format.layout, l_012)
    self.assertArraysAllClose(out_b3d, jnp.einsum('bmk,bkn->bmn', arr3, arr3))

  @jtu.with_explicit_mesh((2,), ('data',))
  def test_dot_4d_output_and_chained_layout(self, mesh):
    a, b, c, d, e, f, g = 4, 256, 256, 512, 4, 4, 256
    x = jax.device_put(
        jnp.arange(a * b * c, dtype=jnp.float32).reshape(a, b, c),
        P(None, None, 'data'))
    w1 = jax.device_put(
        jnp.ones((e, b, d), dtype=jnp.float32), P(None, 'data', None))
    w2 = jax.device_put(
        jnp.ones((e, d, b), dtype=jnp.float32), P(None, 'data', None))
    w3 = jax.device_put(
        jnp.ones((f, d, g), dtype=jnp.float32), P(None, 'data', None))
    tiling = x.format.layout.tiling
    l_012 = Layout((0, 1, 2), tiling)
    l_0213 = Layout((0, 2, 1, 3), tiling)
    l_02314 = Layout((0, 2, 3, 1, 4), tiling)

    # 1. Single matmul: (a, b, c) @ (e, b, d) -> (a, c, e, d)
    # lhs (a, b, c) layout (0, 1, 2) -> a major, (b, c) minor (c is non-contracting)
    # rhs (e, b, d) layout (0, 1, 2) -> e major, (b, d) minor (d is non-contracting)
    # out (a, c, e, d) -> (a, e) (dims 0, 2) major, (c, d) (dims 1, 3) minor -> (0, 2, 1, 3)
    @jax.jit
    def single_without(x, w1):
      return jnp.einsum('abc,ebd->aced', x, w1)

    @jax.jit
    @explicit_layout(in_layouts=(x.format.layout, w1.format.layout))
    def single_with(x, w1):
      out = jnp.einsum('abc,ebd->aced', x, w1)
      self.assertEqual(out.aval.layout, l_0213)
      return out

    out_s_without = single_without(x, w1)
    out_s_with = single_with(x, w1)
    self.assertEqual(out_s_with.format.layout, l_0213)
    self.assertArraysAllClose(out_s_with, out_s_without)

    # 2. Chained matmul contracting (e, d) back down:
    # (a, b, c) @ (e, b, d) -> (a, c, e, d) @ (e, d, b) -> (a, c, b)
    @jax.jit
    def chain_down_without(x, w1, w2):
      y = jnp.einsum('abc,ebd->aced', x, w1)
      return jnp.einsum('aced,edb->acb', y, w2, out_sharding=P(None, 'data'))

    @jax.jit
    @explicit_layout(
        in_layouts=(x.format.layout, w1.format.layout, w2.format.layout)
    )
    def chain_down_with(x, w1, w2):
      y = jnp.einsum('abc,ebd->aced', x, w1)
      self.assertEqual(y.aval.layout, l_0213)
      out = jnp.einsum('aced,edb->acb', y, w2, out_sharding=P(None, 'data'))
      self.assertEqual(out.aval.layout, l_012)
      return out

    out_cd_without = chain_down_without(x, w1, w2)
    out_cd_with = chain_down_with(x, w1, w2)
    self.assertEqual(out_cd_with.format, out_cd_without.format)
    self.assertArraysAllClose(out_cd_with, out_cd_without)

    # 3. Chained matmul contracting only d and growing to 5D:
    # (a, b, c) @ (e, b, d) -> (a, c, e, d) @ (f, d, g) -> (a, c, e, f, g)
    # y (a, c, e, d) layout (0, 2, 1, 3) -> (a, e) major, (c, d) minor
    # w3 (f, d, g) layout (0, 1, 2) -> f major, (d, g) minor
    # out (a, c, e, f, g) -> (a, e, f) (dims 0, 2, 3) major, (c, g) (dims 1, 4) minor -> (0, 2, 3, 1, 4)
    @jax.jit
    def chain_grow_without(x, w1, w3):
      y = jnp.einsum('abc,ebd->aced', x, w1)
      return jnp.einsum('aced,fdg->acefg', y, w3, out_sharding=P(None, 'data'))

    @jax.jit
    @explicit_layout(
        in_layouts=(x.format.layout, w1.format.layout, w3.format.layout)
    )
    def chain_grow_with(x, w1, w3):
      y = jnp.einsum('abc,ebd->aced', x, w1)
      self.assertEqual(y.aval.layout, l_0213)
      out = jnp.einsum('aced,fdg->acefg', y, w3, out_sharding=P(None, 'data'))
      self.assertEqual(out.aval.layout, l_02314)
      return out

    out_cg_without = chain_grow_without(x, w1, w3)
    out_cg_with = chain_grow_with(x, w1, w3)
    self.assertEqual(out_cg_with.format.layout, l_02314)
    self.assertArraysAllClose(out_cg_with, out_cg_without)

  def test_dot_layout_errors(self):
    arr2 = jnp.arange(64., dtype=jnp.float32).reshape(8, 8)
    arr3 = jnp.arange(8 * 8 * 8, dtype=jnp.float32).reshape(8, 8, 8)
    tiling = arr3.format.layout.tiling
    l_2d = Layout((0, 1), tiling)
    l_lhs_ok = Layout((1, 0, 2), tiling)
    l_rhs_ok = Layout((0, 1, 2), tiling)

    # Error 1: lhs 2 minor-most dims are both contracting ((1, 2) in (0, 1, 2))
    @jax.jit
    @explicit_layout(in_layouts=(Layout((0, 1, 2), tiling), l_rhs_ok))
    def bad_lhs_minor(x, y):
      return jnp.einsum('mab,abn->mn', x, y)

    with self.assertRaisesRegex(
        ValueError,
        'dot_general requires the 2 minor-most dims of lhs to be one'
        ' non-contracting and one contracting dim'):
      bad_lhs_minor(arr3, arr3)

    # Error 2: rhs 2 minor-most dims are both contracting ((0, 1) in (2, 0, 1))
    @jax.jit
    @explicit_layout(in_layouts=(l_lhs_ok, Layout((2, 0, 1), tiling)))
    def bad_rhs_minor(x, y):
      return jnp.einsum('mab,abn->mn', x, y)

    with self.assertRaisesRegex(
        ValueError,
        'dot_general requires the 2 minor-most dims of rhs to be one'
        ' non-contracting and one contracting dim'):
      bad_rhs_minor(arr3, arr3)

    # Error 3: lhs 2 minor-most dims are both non-contracting ((0, 1) in (2, 0, 1))
    @jax.jit
    @explicit_layout(in_layouts=(Layout((2, 0, 1), tiling), l_2d))
    def bad_lhs_both_nc(x, y):
      return jnp.einsum('mpk,kn->mpn', x, y)

    with self.assertRaisesRegex(
        ValueError,
        'dot_general requires the 2 minor-most dims of lhs to be one'
        ' non-contracting and one contracting dim'):
      bad_lhs_both_nc(arr3, arr2)

    # Error 4: mismatched relative order of contracting dims
    # lhs has (1, 0, 2) -> contracting order (K1, K2)
    # rhs has (1, 0, 2) -> contracting order (K2, K1)
    @jax.jit
    @explicit_layout(in_layouts=(l_lhs_ok, Layout((1, 0, 2), tiling)))
    def bad_contract_order(x, y):
      return jnp.einsum('mab,abn->mn', x, y)

    with self.assertRaisesRegex(
        ValueError,
        'dot_general requires lhs and rhs contracting dimensions to have the'
        ' same relative layout order'):
      bad_contract_order(arr3, arr3)

    # Error 5: batch dim not most major
    @jax.jit
    @explicit_layout(in_layouts=(Layout((1, 2, 0), tiling), l_rhs_ok))
    def bad_batch_major(x, y):
      return jnp.einsum('bmk,bkn->bmn', x, y)

    with self.assertRaisesRegex(
        ValueError,
        r'dot_general requires lhs batch dims \(0,\) to be most major'):
      bad_batch_major(arr3, arr3)

    # Error 6: mismatched relative order of batch dims
    arr4 = jnp.arange(8 * 8 * 8 * 8, dtype=jnp.float32).reshape(8, 8, 8, 8)
    @jax.jit
    @explicit_layout(
        in_layouts=(Layout((0, 1, 2, 3), tiling), Layout((1, 0, 2, 3), tiling))
    )
    def bad_batch_order(x, y):
      return jnp.einsum('abmk,abkn->abmn', x, y)

    with self.assertRaisesRegex(
        ValueError,
        'dot_general requires lhs and rhs batch dimensions to have the same'
        ' relative layout order'):
      bad_batch_order(arr4, arr4)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
