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
from jax._src.pjit import relayout
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

  @jax.default_matmul_precision("float32")
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

  def test_transpose_layout(self):
    arr = jnp.arange(64).reshape(4, 16)
    arr2 = jnp.arange(512).reshape(4, 8, 16)

    @jax.jit
    @explicit_layout(in_layouts=Layout((0, 1)))
    def f(w):
      w_t = w.T
      self.assertEqual(w_t.aval.layout.major_to_minor, (1, 0))
      return w_t

    lowered_text = f.lower(arr).as_text()
    self.assertIn('LayoutConstraint', lowered_text)

    w_t = f(arr)
    self.assertEqual(w_t.format.layout.major_to_minor, (1, 0))
    self.assertArraysAllClose(w_t, arr.T)

    @jax.jit
    @explicit_layout(in_layouts=(Layout((0, 1, 2))))
    def g(x):
      xT = jnp.transpose(x, (2, 0, 1))
      self.assertEqual(xT.aval.layout.major_to_minor, (1, 2, 0))
      return xT

    xT = g(arr2)
    self.assertEqual(xT.format.layout.major_to_minor, (1, 2, 0))
    self.assertArraysAllClose(xT, np.transpose(arr2, (2, 0, 1)))

    lowered_text = g.lower(arr2).as_text()
    self.assertEqual(lowered_text.count('LayoutConstraint'), 2)

  def test_convert_element_type_layout(self):
    arr = jnp.arange(64, dtype=jnp.float32).reshape(4, 16)
    arr_t = jax.device_put(arr, Format(Layout((1, 0)), arr.sharding))

    @jax.jit
    @explicit_layout(in_layouts=Layout((1, 0)))
    def f(x):
      y = jax.lax.convert_element_type(x, jnp.bfloat16)
      self.assertEqual(y.aval.layout.major_to_minor, (1, 0))
      return y

    lowered_text = f.lower(arr_t).as_text()
    self.assertEqual(lowered_text.count('LayoutConstraint'), 2)

    out = f(arr_t)
    self.assertEqual(out.format.layout.major_to_minor, (1, 0))
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertArraysAllClose(out, arr.astype(jnp.bfloat16))

    @jax.jit
    @explicit_layout(in_layouts=Layout((0, 1), tiling=((8, 128),)))
    def g(x):
      return jax.lax.convert_element_type(x, jnp.bfloat16)

    with self.assertRaises(NotImplementedError):
      g(arr)

  def test_relayout_transpose(self):
    arr1 = jnp.arange(64, dtype=jnp.float32).reshape(4, 16)
    arr2 = jnp.arange(64, dtype=jnp.float32).reshape(4, 16)

    @explicit_layout(in_layouts=(Layout((0, 1)), Layout((1, 0))))
    def f(x, y):
      y = relayout(y, Layout((0, 1)))
      z = x + y
      self.assertEqual(z.aval.layout.major_to_minor, (0, 1))
      return z

    primal_out, tangent_out = jax.jit(
        lambda x, y: jax.jvp(f, (x, y), (x, y))
    )(arr1, arr2)
    self.assertEqual(primal_out.format.layout.major_to_minor, (0, 1))
    self.assertEqual(tangent_out.format.layout.major_to_minor, (0, 1))
    self.assertArraysAllClose(primal_out, arr1 + arr2)
    self.assertArraysAllClose(tangent_out, arr1 + arr2)

    x_bar, y_bar = jax.jit(
        jax.grad(lambda x, y: f(x, y).sum(), argnums=(0, 1))
    )(arr1, arr2)
    self.assertEqual(x_bar.format.layout.major_to_minor, (0, 1))
    self.assertEqual(y_bar.format.layout.major_to_minor, (1, 0))
    self.assertArraysAllClose(x_bar, jnp.ones_like(arr1))
    self.assertArraysAllClose(y_bar, jnp.ones_like(arr1))

  @parameterized.parameters(
      (src_shape, dst_shape, src_m2m, dst_m2m, fun)
      for fun in [jnp.reshape, jax.lax.reshape]
      for src_shape, dst_shape, src_m2m, dst_m2m in [
          ((4, 8, 1), (1, 4, 8, 1), (2, 0, 1), (0, 3, 1, 2)),
          ((4, 8, 1), (1, 4, 8, 1), (2, 1, 0), (0, 3, 2, 1)),
          ((1, 4, 1, 8, 16, 1), (1, 4, 8, 16), (0, 2, 5, 3, 1, 4), (0, 2, 1, 3)),
          ((4, 8), (4, 8), (1, 0), (1, 0)),
          ((8, 4, 16), (4, 2, 4, 1, 1, 16), (0, 1, 2), (3, 4, 0, 1, 2, 5)),
          ((8, 4, 16), (8, 1, 1, 4, 16), (1, 0, 2), (1, 2, 3, 0, 4)),
          ((4, 8, 2, 1, 1, 16), (32, 2, 16), (0, 1, 3, 4, 2, 5), (0, 1, 2)),
          ((4, 1, 8, 1, 16), (4, 1, 2, 4, 1, 16), (1, 3, 2, 0, 4), (1, 4, 2, 3, 0, 5)),
          ((4, 1, 2, 4, 1, 16), (4, 1, 8, 1, 16), (1, 4, 2, 3, 0, 5), (1, 3, 2, 0, 4)),
      ]
  )
  def test_reshape_layout(self, src_shape, dst_shape, src_m2m, dst_m2m, fun):
    np_inp = np.arange(math.prod(src_shape), dtype=np.float32).reshape(src_shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout(src_m2m), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = fun(x, dst_shape)
      self.assertEqual(y.aval.layout.major_to_minor, dst_m2m)
      self.assertEqual(y.shape, dst_shape)
      return y

    out = f(arr)
    self.assertEqual(out.format.layout.major_to_minor, dst_m2m)
    self.assertArraysEqual(out, np_inp.reshape(dst_shape))

    lowered_text = f.lower(arr).as_text()
    self.assertIn('LayoutConstraint', lowered_text)

  @parameterized.parameters(
      # Splits on major dimensions
      ((6, 4, 8), (2, 3, 4, 8), (0, 1, 2), (0, 1, 2, 3), None),
      ((6, 4, 8), (2, 3, 4, 8), (0, 2, 1), (0, 1, 3, 2), None),
      ((4, 6, 8), (4, 2, 3, 8), (1, 0, 2), (1, 2, 0, 3), None),
      ((4, 6, 8), (4, 2, 3, 8), (1, 2, 0), (1, 2, 3, 0), None),
      ((4, 12, 8, 16), (4, 2, 2, 3, 8, 16), (0, 1, 2, 3), (0, 1, 2, 3, 4, 5), None),
      ((4, 12, 8, 16), (4, 2, 2, 3, 8, 16), (1, 0, 2, 3), (1, 2, 3, 0, 4, 5), None),
      ((10, 4, 8, 1), (2, 5, 4, 8, 1), (3, 0, 1, 2), (4, 0, 1, 2, 3), None),
      ((10, 4, 8, 1), (2, 5, 4, 8, 1, 1), (3, 0, 1, 2), (4, 5, 0, 1, 2, 3), None),
      ((10, 4, 8, 1, 1), (2, 5, 4, 8, 1, 1), (3, 4, 0, 1, 2), (4, 5, 0, 1, 2, 3), None),
      ((1, 10, 4, 8), (1, 2, 5, 4, 8), (0, 1, 2, 3), (0, 1, 2, 3, 4), None),
      # Merges on major dimensions
      ((2, 3, 4, 8), (6, 4, 8), (0, 1, 2, 3), (0, 1, 2), None),
      ((2, 3, 4, 8), (6, 4, 8), (0, 1, 3, 2), (0, 2, 1), None),
      ((4, 2, 3, 8), (4, 6, 8), (1, 2, 0, 3), (1, 0, 2), None),
      ((4, 2, 3, 8), (4, 6, 8), (1, 2, 3, 0), (1, 2, 0), None),
      ((4, 2, 2, 3, 8, 16), (4, 12, 8, 16), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3), None),
      ((4, 2, 2, 3, 8, 16), (4, 12, 8, 16), (1, 2, 3, 0, 4, 5), (1, 0, 2, 3), None),
      ((4, 2, 2, 8, 16), (4, 4, 8, 16), (1, 2, 0, 3, 4), (1, 0, 2, 3), None),
      ((2, 5, 4, 8, 1), (10, 4, 8, 1), (4, 0, 1, 2, 3), (3, 0, 1, 2), None),
      ((16, 8, 4, 2, 8), (16, 32, 1, 2, 8), (0, 1, 2, 3, 4), (2, 0, 1, 3, 4), None),
      ((16, 32, 1, 2, 8), (16, 8, 4, 2, 8), (2, 0, 1, 3, 4), (0, 1, 2, 3, 4), None),
      # Splitting or merging the 2 minor-most dimensions (errors on TPU, succeeds on CPU/GPU)
      ((4, 6, 8), (4, 2, 3, 8), (0, 1, 2), (0, 1, 2, 3),
       'cannot split the 2 minor-most dimensions'),
      ((4, 6, 8), (4, 6, 2, 2, 2), (0, 1, 2), (0, 1, 2, 3, 4),
       'cannot split the 2 minor-most dimensions'),
      ((4, 6, 8), (2, 2, 6, 8), (1, 0, 2), (2, 0, 1, 3),
       'cannot split the 2 minor-most dimensions'),
      ((4, 6, 8), (4, 6, 2, 4), (1, 0, 2), (1, 0, 2, 3),
       'cannot split the 2 minor-most dimensions'),
      ((4, 6, 8), (4, 48), (0, 1, 2), (0, 1),
       'cannot merge the 2 minor-most dimensions'),
      ((4, 2, 3, 8), (4, 6, 8), (0, 1, 2, 3), (0, 1, 2),
       'cannot merge the 2 minor-most dimensions'),
      # Singleton in the 2 minor-most dimensions (errors on TPU, succeeds on CPU/GPU)
      ((4, 8, 1), (1, 4, 8, 1), (0, 1, 2), (0, 3, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((4, 8, 1), (1, 4, 8, 1), (1, 0, 2), (0, 3, 2, 1),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((10, 4, 8, 1), (10, 4, 8), (0, 1, 2, 3), (0, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((10, 4, 8, 1), (2, 5, 4, 8, 1), (0, 1, 2, 3), (4, 0, 1, 2, 3),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((2, 5, 4, 8, 1), (10, 4, 8, 1), (0, 1, 2, 3, 4), (3, 0, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((4, 8, 2, 1, 1, 16), (32, 2, 16), (0, 1, 2, 3, 4, 5), (0, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      # Error cases: merging out-of-order or non-contiguous physical dims, or multi-axis splits/merges
      ((4, 2, 3, 8), (4, 6, 8), (2, 1, 0, 3), None,
       'is not possible without a physical copy'),
      ((4, 2, 3, 8), (8, 3, 8), (1, 0, 2, 3), None,
       'is not possible without a physical copy'),
      ((4, 6, 8, 16), (4, 2, 3, 4, 2, 16), (0, 1, 2, 3), None,
       'is not possible without a physical copy'),
      ((4, 6, 8, 16), (4, 4, 2, 6, 16), (0, 1, 2, 3), None,
       'is not possible without a physical copy'),
      ((4, 8, 9, 16), (4, 2, 2, 3, 3, 2, 16), (0, 1, 2, 3), None,
       'is not possible without a physical copy'),
      ((4, 2, 3, 2, 4, 16), (4, 6, 8, 16), (0, 1, 2, 3, 4, 5), None,
       'is not possible without a physical copy'),
      ((4, 2, 3, 8, 16), (4, 8, 6, 16), (0, 1, 2, 3, 4), None,
       'is not possible without a physical copy'),
  )
  def test_reshape_split_merge_one_axis_layout(
      self, src_shape, dst_shape, src_m2m, dst_m2m, error_msg):
    np_inp = np.arange(math.prod(src_shape), dtype=np.float32).reshape(src_shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout(src_m2m), s))
    should_error = (error_msg is not None and
                    (dst_m2m is None or jtu.test_device_matches(['tpu'])))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = jax.lax.reshape(x, dst_shape)
      if not should_error:
        self.assertEqual(y.aval.layout.major_to_minor, dst_m2m)
      return y

    if should_error:
      with self.assertRaisesRegex(ValueError, error_msg):
        f(arr)
    else:
      out = f(arr)
      self.assertEqual(out.format.layout.major_to_minor, dst_m2m)
      self.assertArraysEqual(out, np_inp.reshape(dst_shape))

      lowered_text = f.lower(arr).as_text()
      self.assertIn('LayoutConstraint', lowered_text)

  def test_reshape_dimensions_error_layout(self):
    arr = jnp.arange(4 * 8 * 16, dtype=np.float32).reshape(4, 8, 16)

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      return jax.lax.reshape(x, (4, 2, 4, 16), dimensions=(1, 0, 2))

    with self.assertRaisesRegex(
        ValueError, 'is not possible without a physical copy'):
      f(arr)

  @jtu.with_explicit_mesh((2,), ('data',))
  def test_reshape_sharded_layout(self, mesh):
    src_shape = (3, 2, 8, 16, 32)
    dst_shape = (2, 24, 16, 32)
    np_inp = np.arange(math.prod(src_shape), dtype=np.float32).reshape(src_shape)
    s = NamedSharding(mesh, P(None, 'data', None, None, None))
    arr = jax.device_put(np_inp, Format(Layout((0, 1, 2, 3, 4)), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = jax.lax.reshape(
          x, dst_shape, out_sharding=P('data', None, None, None))
      self.assertEqual(y.aval.layout.major_to_minor, (0, 1, 2, 3))
      return y

    out = f(arr)
    self.assertEqual(out.format.layout.major_to_minor, (0, 1, 2, 3))
    self.assertArraysEqual(out, np_inp.reshape(dst_shape))

  @parameterized.parameters(
      # Single major singleton dim squeezed
      ((1, 4, 8, 128), (0,), (0, 1, 2, 3), (0, 1, 2), None),
      ((10, 4, 8, 1), (3,), (3, 0, 1, 2), (0, 1, 2), None),
      ((10, 8, 128, 1), (3,), (3, 0, 1, 2), (0, 1, 2), None),
      ((4, 1, 8, 128), (1,), (1, 2, 0, 3), (1, 0, 2), None),
      # Multiple major singleton dims (squeezing all or a subset)
      ((1, 4, 1, 8, 128), (0, 2), (0, 2, 1, 3, 4), (0, 1, 2), None),
      ((1, 4, 1, 8, 128), (0,), (0, 1, 2, 3, 4), (1, 0, 2, 3), None),
      ((1, 4, 1, 8, 16, 1), (2, 5), (0, 2, 5, 3, 1, 4), (0, 2, 1, 3), None),
      # Singleton in the 2 minor-most dimensions (errors on TPU, succeeds on CPU/GPU)
      ((10, 4, 8, 1), (3,), (0, 1, 2, 3), (0, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((10, 8, 128, 1), (3,), (0, 1, 2, 3), (0, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((10, 3, 1, 128), (2,), (0, 1, 2, 3), (0, 1, 2),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
      ((1, 4, 8, 1), (0,), (0, 1, 2, 3), (2, 0, 1),
       'cannot reshape when any of the 2 minor-most dimensions has size 1'),
  )
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_squeeze_layout(
      self, src_shape, dims, src_m2m, dst_m2m, error_msg, mesh):
    np_inp = np.arange(math.prod(src_shape), dtype=np.float32).reshape(src_shape)
    arr = jax.device_put(np_inp, Format(Layout(src_m2m), NamedSharding(mesh, P())))
    should_error = (error_msg is not None and jtu.test_device_matches(['tpu']))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = jax.lax.squeeze(x, dimensions=dims)
      if not should_error:
        self.assertEqual(y.aval.layout.major_to_minor, dst_m2m)
      return y

    if should_error:
      with self.assertRaisesRegex(ValueError, error_msg):
        f(arr)
    else:
      out = f(arr)
      self.assertEqual(out.format.layout.major_to_minor, dst_m2m)
      self.assertArraysEqual(out, np.squeeze(np_inp, axis=dims))
      lowered_text = f.lower(arr).as_text()
      self.assertIn('LayoutConstraint', lowered_text)

  @parameterized.named_parameters(
      ('integer_pow', lambda x: jax.lax.integer_pow(x, 3)),
      ('reduce_precision', lambda x: jax.lax.reduce_precision(x, 5, 10)),
      ('slice', lambda x: jax.lax.slice(x, (0, 1, 0), (8, 5, 128))),
      ('dynamic_slice', lambda x: jax.lax.dynamic_slice(x, (0, 1, 0), (8, 4, 128))),
  )
  def test_noop_unop_layouts(self, op_fn):
    shape = (16, 8, 256)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout((1, 0, 2)), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = op_fn(x)
      self.assertEqual(y.aval.layout, arr.format.layout)
      return y

    out = f(arr)
    self.assertEqual(out.format.layout, arr.format.layout)
    self.assertArraysAllClose(out, jax.jit(op_fn)(np_inp))
    self.assertEqual(f.lower(arr).as_text().count('LayoutConstraint'), 2)

  def test_split_layout(self):
    shape = (8, 8, 16)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout((1, 0, 2)), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y0, y1 = jax.lax.split(x, (2, 6), axis=1)
      self.assertEqual(y0.aval.layout, arr.format.layout)
      self.assertEqual(y1.aval.layout, arr.format.layout)
      return y0, y1

    out0, out1 = f(arr)
    self.assertEqual(out0.format.layout, arr.format.layout)
    self.assertEqual(out1.format.layout, arr.format.layout)
    ref0, ref1 = np.split(np_inp, [2], axis=1)
    self.assertArraysEqual(out0, ref0)
    self.assertArraysEqual(out1, ref1)
    self.assertEqual(f.lower(arr).as_text().count('LayoutConstraint'), 3)

  @parameterized.named_parameters(
      ('clamp',
       lambda x, y: jax.lax.clamp(np.float32(0.0), x, y),
       (8, 8, 16)),
      ('select_n',
       lambda x, y: jax.lax.select_n(x > np.float32(10.0), x, y),
       (8, 8, 16)),
      ('concatenate',
       lambda x, y: jax.lax.concatenate([x, y], dimension=1),
       (8, 8, 16)),
      ('dynamic_update_slice',
       lambda x, y: jax.lax.dynamic_update_slice(x, y, (0, 1, 0)),
       (8, 4, 16)),
  )
  def test_noop_multi_operand_layouts(self, op_fn, y_shape):
    x_shape = (8, 8, 16)
    np_x = np.arange(math.prod(x_shape), dtype=np.float32).reshape(x_shape)
    np_y = np.ones(y_shape, dtype=np.float32)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr_x = jax.device_put(np_x, Format(Layout((1, 0, 2)), s))
    arr_y = jax.device_put(np_y, Format(Layout((1, 0, 2)), s))
    arr_y_bad = jax.device_put(np_y, Format(Layout((0, 1, 2)), s))

    @jax.jit
    @explicit_layout(in_layouts=(arr_x.format.layout, arr_y.format.layout))
    def f(x, y):
      out = op_fn(x, y)
      self.assertEqual(out.aval.layout, arr_x.format.layout)
      return out

    out = f(arr_x, arr_y)
    self.assertEqual(out.format.layout, arr_x.format.layout)
    self.assertArraysAllClose(out, jax.jit(op_fn)(np_x, np_y))
    self.assertIn('LayoutConstraint', f.lower(arr_x, arr_y).as_text())

    @jax.jit
    @explicit_layout(in_layouts=(arr_x.format.layout, arr_y_bad.format.layout))
    def f_bad(x, y):
      return op_fn(x, y)

    with self.assertRaisesRegex(
        ValueError, 'layout of all inputs passed to .* must be the same'):
      f_bad(arr_x, arr_y_bad)

  @jtu.run_on_devices('tpu')
  def test_minor_dim_tile_size_error(self):
    shape = (16, 128)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout((0, 1)), s))

    for op_fn in [
        lambda x: jax.lax.slice(x, (0, 0), (4, 128)),
        lambda x: jax.lax.dynamic_slice(x, (0, 0), (4, 128)),
        lambda x: jax.lax.split(x, (4, 12), axis=0)[0],
    ]:
      @jax.jit
      @explicit_layout(in_layouts=arr.format.layout)
      def f_bad(x):
        return op_fn(x)

      with self.assertRaisesRegex(
          ValueError,
          'cannot change the shard shape of a tiled minor dimension below its'
          ' tile size'):
        f_bad(arr)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_reshard_layout(self, mesh):
    shape = (16, 32, 256)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s_in = NamedSharding(mesh, P('x', None, None))
    s_out = NamedSharding(mesh, P(None, 'x', None))
    arr = jax.device_put(np_inp, Format(Layout((0, 1, 2)), s_in))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      # Minor-most dims are (1, 2); dim 1 shard shape changes 32 -> 16 (>= 8).
      y = jax.reshard(x, s_out)
      self.assertEqual(y.aval.layout, arr.format.layout)
      return y

    out = f(arr)
    self.assertEqual(out.format.layout, arr.format.layout)
    self.assertArraysEqual(out, np_inp)
    self.assertEqual(f.lower(arr).as_text().count('LayoutConstraint'), 2)

    if jtu.test_device_matches(['tpu']):
      small_shape = (16, 12, 128)
      np_small = np.ones(small_shape, dtype=np.float32)
      arr_small = jax.device_put(np_small, Format(Layout((0, 1, 2)), s_in))

      @jax.jit
      @explicit_layout(in_layouts=arr_small.format.layout)
      def f_bad(x):
        # Dim 1 shard shape changes 12 -> 6 (< 8 tile size on TPU).
        return jax.reshard(x, s_out)

      with self.assertRaisesRegex(
          ValueError,
          '`reshard` cannot change the shard shape of a tiled minor'
          ' dimension below its tile size'):
        f_bad(arr_small)

  def test_missing_custom_abstract_eval_layout_rules(self):
    shape = (8, 128)
    np_inp = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout((0, 1)), s))

    for op_fn in [
        lambda x: jax.lax.sort(x, dimension=0),
        lambda x: jax.lax.top_k(x, k=4, axis=0)[0],
        lambda x: jax.lax.approx_max_k(x, k=4, reduction_dimension=0)[0],
        lambda x: jax.lax.reduce_window(
            x, np.float32(0.0), jax.lax.add, (2, 2), (2, 2), 'VALID'),
    ]:
      @jax.jit
      @explicit_layout(in_layouts=arr.format.layout)
      def f(x):
        return op_fn(x)

      with self.assertRaisesRegex(
          NotImplementedError, 'Missing layout rule for'):
        f(arr)

  @parameterized.parameters(
      ((), (), (16, 128), (), (0, 1)),
      ((), (), (4, 16, 128), (), (0, 1, 2)),
      ((128,), (0,), (16, 128), (1,), (0, 1)),
      ((128,), (0,), (128, 16), (0,), (1, 0)),
      ((128,), (0,), (4, 128, 8), (1,), (0, 2, 1)),
      ((128, 16), (1, 0), (4, 128, 8, 16), (1, 3), (0, 2, 3, 1)),
      ((1, 16, 128), (0, 1, 2), (8, 16, 128), (0, 1, 2), (0, 1, 2)),
  )
  def test_broadcast_in_dim_layout(
      self, in_shape, in_m2m, out_shape, bcast_dims, expected_m2m):
    np_inp = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np_inp, Format(Layout(in_m2m), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = jax.lax.broadcast_in_dim(x, out_shape, bcast_dims)
      self.assertEqual(y.aval.layout.major_to_minor, expected_m2m)
      return y

    out = f(arr)
    self.assertEqual(out.format.layout.major_to_minor, expected_m2m)
    ref = jax.jit(
        lambda x: jax.lax.broadcast_in_dim(x, out_shape, bcast_dims))(np_inp)
    self.assertArraysEqual(out, ref)
    self.assertIn('LayoutConstraint', f.lower(arr).as_text())

  @parameterized.parameters(
      ((32, 256), (0, 1), (4, 32, 256), (1, 2), P(None, 'x', None), (0, 1, 2),
       None),
      ((8, 1, 128), (0, 1, 2), (8, 16, 128), (0, 1, 2), P(), (0, 1, 2),
       'cannot change the shard shape of a tiled minor dimension'),
      ((8, 16, 1), (0, 1, 2), (8, 16, 128), (0, 1, 2), P(), (0, 1, 2),
       'cannot change the shard shape of a tiled minor dimension'),
      ((12, 128), (0, 1), (4, 12, 128), (1, 2), P(None, 'x', None), (0, 1, 2),
       'cannot change the shard shape of a tiled minor dimension'),
  )
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_broadcast_in_dim_sharded_layout(
      self, in_shape, in_m2m, out_shape, bcast_dims, out_spec, expected_m2m,
      tpu_error, mesh):
    np_inp = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    arr = jax.device_put(np_inp, Format(Layout(in_m2m), NamedSharding(mesh, P())))
    s_out = NamedSharding(mesh, out_spec)
    should_error = tpu_error is not None and jtu.test_device_matches(['tpu'])

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      y = jax.lax.broadcast_in_dim(
          x, out_shape, bcast_dims, out_sharding=s_out)
      if not should_error:
        self.assertEqual(y.aval.layout.major_to_minor, expected_m2m)
      return y

    if should_error:
      with self.assertRaisesRegex(ValueError, tpu_error):
        f(arr)
    else:
      out = f(arr)
      self.assertEqual(out.format.layout.major_to_minor, expected_m2m)
      ref = jax.jit(
          lambda x: jax.lax.broadcast_in_dim(x, out_shape, bcast_dims))(np_inp)
      self.assertArraysEqual(out, ref)

  @parameterized.parameters(
      ((), (), P(), 0, (0,)),
      ((128,), (0,), P(), 0, (0, 1)),
      ((128,), (0,), P(), 1, (1, 0)),
      ((128, 16), (1, 0), P(), 0, (0, 2, 1)),
      ((128, 16), (1, 0), P(), 1, (1, 2, 0)),
      ((128, 16), (1, 0), P(), 2, (2, 1, 0)),
      ((32, 256), (1, 0), P('x', None), 1, (1, 2, 0)),
  )
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_stack_layout(
      self, in_shape, in_m2m, spec, axis, expected_m2m, mesh):
    np_a = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    np_b = np_a + 100.0
    sharding = NamedSharding(mesh, spec)
    arr_a = jax.device_put(np_a, Format(Layout(in_m2m), sharding))
    arr_b = jax.device_put(np_b, Format(Layout(in_m2m), sharding))

    @jax.jit
    @explicit_layout(in_layouts=(arr_a.format.layout, arr_b.format.layout))
    def f(x, y):
      out = jnp.stack([x, y], axis=axis)
      self.assertEqual(out.aval.layout.major_to_minor, expected_m2m)
      return out

    out = f(arr_a, arr_b)
    self.assertEqual(out.format.layout.major_to_minor, expected_m2m)
    self.assertArraysEqual(out, np.stack([np_a, np_b], axis=axis))
    self.assertIn('LayoutConstraint', f.lower(arr_a, arr_b).as_text())

  @parameterized.parameters(
      ((2, 128, 16), (0, 2, 1), P(), 0, (1, 0)),
      ((128, 2, 16), (1, 2, 0), P(), 1, (1, 0)),
      ((128, 16, 2), (2, 1, 0), P(), 2, (1, 0)),
      ((4, 2, 16, 128), (1, 0, 2, 3), P('x', None, None, None), 1, (0, 1, 2)),
  )
  @jtu.with_explicit_mesh((2,), ('x',))
  def test_unstack_layout(
      self, in_shape, in_m2m, spec, axis, expected_m2m, mesh):
    np_inp = np.arange(math.prod(in_shape), dtype=np.float32).reshape(in_shape)
    arr = jax.device_put(np_inp, Format(Layout(in_m2m), NamedSharding(mesh, spec)))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      u0, u1 = jnp.unstack(x, axis=axis)
      self.assertEqual(u0.aval.layout.major_to_minor, expected_m2m)
      return u0, u1

    u0, u1 = f(arr)
    self.assertEqual(u0.format.layout.major_to_minor, expected_m2m)
    self.assertEqual(u1.format.layout.major_to_minor, expected_m2m)
    self.assertArraysEqual(u0, np.take(np_inp, 0, axis=axis))
    self.assertArraysEqual(u1, np.take(np_inp, 1, axis=axis))
    self.assertIn('LayoutConstraint', f.lower(arr).as_text())

  def test_stack_layout_mismatch_error(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    m1 = jax.device_put(np.ones((16, 128), np.float32), Format(Layout((0, 1)), s))
    m2 = jax.device_put(np.ones((16, 128), np.float32), Format(Layout((1, 0)), s))

    @jax.jit
    @explicit_layout(in_layouts=(m1.format.layout, m2.format.layout))
    def f(x, y):
      return jax.lax.stack([x, y], axis=0)

    with self.assertRaisesRegex(
        ValueError, 'layout of all inputs passed to `stack` must be the same'):
      f(m1, m2)

  @parameterized.parameters(
      ((4, 3, 16, 128), (1, 0, 2, 3), 0, False,
       '`unstack` requires `axis` .* to be the major-most dimension'),
      ((4, 16, 128), (0, 1, 2), 1, False,
       '`unstack` requires `axis` .* to be the major-most dimension'),
      ((4, 16, 128), (0, 1, 2), 2, False,
       '`unstack` requires `axis` .* to be the major-most dimension'),
      ((16, 128), (0, 1), 0, True,
       '`unstack` along axis 0 cannot remove one of the 2 tiled'),
      ((128,), (0,), 0, True,
       '`unstack` along axis 0 cannot remove one of the 1 tiled'),
  )
  def test_unstack_layout_errors(
      self, shape, m2m, axis, tpu_only, error_regex):
    if tpu_only and not jtu.test_device_matches(['tpu']):
      self.skipTest('TPU tiling error test')
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np.ones(shape, np.float32), Format(Layout(m2m), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      return jax.lax.unstack(x, axis=axis)

    with self.assertRaisesRegex(ValueError, error_regex):
      f(arr)

  def test_stack_jvp_and_grad_layout(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    np_x = np.arange(128 * 16, dtype=np.float32).reshape(128, 16) * 0.01
    np_y = np_x + 0.05
    arr_x = jax.device_put(np_x, Format(Layout((1, 0)), s))
    arr_y = jax.device_put(np_y, Format(Layout((1, 0)), s))
    w = jax.device_put(
        np.arange(128 * 2 * 16, dtype=np.float32).reshape(128, 2, 16) * 0.01, s)

    def body(x, y):
      z = jax.lax.stack([x, 2.0 * y], axis=1)
      u0, u1 = jax.lax.unstack(z, axis=1)
      return z + jax.lax.stack([u0 + x, u1], axis=1)

    f = explicit_layout(body, in_layouts=(arr_x.format.layout, arr_y.format.layout))
    primal, tangent = jax.jit(lambda x, y: jax.jvp(f, (x, y), (x, y)))(arr_x, arr_y)
    ref_p, ref_t = jax.jvp(body, (np_x, np_y), (np_x, np_y))
    self.assertArraysAllClose(primal, ref_p)
    self.assertArraysAllClose(tangent, ref_t)

    gx, gy = jax.jit(
        jax.grad(lambda x, y: (f(x, y) * w).sum(), argnums=(0, 1))
    )(arr_x, arr_y)
    ref_gx, ref_gy = jax.grad(
        lambda x, y: (body(x, y) * np.asarray(w)).sum(), argnums=(0, 1)
    )(np_x, np_y)
    self.assertEqual(gx.format.layout, arr_x.format.layout)
    self.assertEqual(gy.format.layout, arr_y.format.layout)
    self.assertArraysAllClose(gx, ref_gx)
    self.assertArraysAllClose(gy, ref_gy)

  def test_unstack_grad_with_unused_output_layout(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    np_3d = np.arange(3 * 128 * 16, dtype=np.float32).reshape(3, 128, 16) * 0.01
    arr_3d = jax.device_put(np_3d, Format(Layout((0, 2, 1)), s))
    w = jax.device_put(np.ones((128, 16), dtype=np.float32), s)

    @explicit_layout(in_layouts=arr_3d.format.layout)
    def f(x):
      a, _, c = jax.lax.unstack(x, axis=0)
      return a + 3.0 * c

    g = jax.jit(jax.grad(lambda x: (f(x) * w).sum()))(arr_3d)
    ref_g = jax.grad(
        lambda x: ((x[0] + 3.0 * x[2]) * np.asarray(w)).sum()
    )(np_3d)
    self.assertEqual(g.format.layout, arr_3d.format.layout)
    self.assertArraysAllClose(g, ref_g)

  def test_iota_and_empty_layout(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(np.ones((16, 128), np.float32), Format(Layout((1, 0)), s))

    @jax.jit
    @explicit_layout(in_layouts=arr.format.layout)
    def f(x):
      idx = jax.lax.broadcasted_iota(np.int32, (4, 16, 128), 1)
      buf = jax.lax.empty((16, 128), np.float32)
      self.assertEqual(idx.aval.layout.major_to_minor, (0, 1, 2))
      self.assertEqual(buf.aval.layout.major_to_minor, (0, 1))
      return idx, buf

    idx, buf = f(arr)
    self.assertEqual(idx.format.layout.major_to_minor, (0, 1, 2))
    self.assertEqual(buf.format.layout.major_to_minor, (0, 1))
    self.assertArraysEqual(idx, jax.lax.broadcasted_iota(np.int32, (4, 16, 128), 1))
    self.assertIn('LayoutConstraint', f.lower(arr).as_text())

  def test_with_layout_constraint_assert(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arr = jax.device_put(
        np.arange(16 * 128, dtype=np.float32).reshape(16, 128),
        Format(Layout((1, 0)), s),
    )
    l = arr.format.layout
    l_transposed = Layout(l.major_to_minor[::-1], l.tiling)

    @jax.jit
    @explicit_layout(in_layouts=l)
    def f(x):
      x = with_layout_constraint(x, l)
      y = relayout(x, l_transposed)
      x, y = with_layout_constraint((x, y), (l, l_transposed))
      return y * 2

    self.assertNotIn('layout_constraint', str(jax.make_jaxpr(f)(arr)))
    out = f(arr)
    self.assertArraysEqual(out, arr * 2)

    f_id = jax.jit(
        explicit_layout(lambda x: with_layout_constraint(x, l), in_layouts=l)
    )
    self.assertEqual(f_id.lower(arr).as_text().count('LayoutConstraint'), 1)

    @jax.jit
    @explicit_layout(in_layouts=l)
    def g_wrong_m2m(x):
      return with_layout_constraint(x, l_transposed)

    with self.assertRaisesRegex(
        AssertionError,
        '`with_layout_constraint` acts as an assert when layouts exist in'
        ' types.*Please use `relayout`',
    ):
      g_wrong_m2m(arr)

    @jax.jit
    @explicit_layout(in_layouts=l)
    def g_none_tiling(x):
      return with_layout_constraint(x, Layout(l.major_to_minor, tiling=None))

    with self.assertRaisesRegex(
        AssertionError,
        '`with_layout_constraint` acts as an assert when layouts exist in'
        ' types',
    ):
      g_none_tiling(arr)

  def test_reshape_vjp_non_major_singleton_layout(self):
    s = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    np_inp = np.arange(2 * 1 * 2 * 16 * 128, dtype=jnp.float32)
    arr1 = jax.device_put(np_inp.reshape(2, 1, 2, 16, 128), s)

    @jax.jit
    def f1(x):
      @explicit_layout(in_layouts=Layout.for_array(x))
      def impl(x):
        return x.reshape(4, 16, 128)

      y, vjp_fn = jax.vjp(impl, x)
      return y, vjp_fn(y)[0]

    y1, x1_bar = f1(arr1)
    self.assertEqual(y1.shape, (4, 16, 128))
    self.assertArraysEqual(x1_bar, arr1)

    # 2. Interior singleton created by split followed by reshape + grad.
    np_inp2 = np.arange(4 * 3 * 16 * 128, dtype=jnp.float32).reshape(4, 3, 16, 128)
    arr2 = jax.device_put(np_inp2, s)

    def split_and_reshape(x):
      return jax.lax.split(x, (1, 2), axis=1)[0].reshape(4, 16, 128)

    f2 = explicit_layout(split_and_reshape, in_layouts=arr2.format.layout)


    g2 = jax.jit(jax.grad(lambda x: f2(x).sum()))(arr2)
    ref_g2 = jax.jit(jax.grad(lambda x: split_and_reshape(x).sum()))(arr2)
    self.assertArraysAllClose(g2, ref_g2)

  @jtu.with_explicit_mesh((2,), ('data',))
  def test_reshape_vjp_sharded_singleton_layout(self, mesh):
    np_inp = np.arange(4 * 2 * 4 * 16 * 128, dtype=jnp.float32)
    arr = jax.device_put(np_inp.reshape(4, 2, 4, 16, 128), P(None, 'data'))

    @jax.jit
    def f(x):
      @explicit_layout(in_layouts=Layout.for_array(x))
      def impl(x):
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        return x.reshape(16, 2, 16, 128, out_sharding=P(None, 'data'))

      y, vjp_fn = jax.vjp(impl, x)
      return y, vjp_fn(y)[0]

    y, x_bar = f(arr)
    self.assertEqual(y.shape, (16, 2, 16, 128))
    self.assertArraysEqual(x_bar, arr)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
