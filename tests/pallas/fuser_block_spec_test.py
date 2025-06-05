# Copyright 2025 The JAX Authors.
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
from absl.testing import parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.pallas.fuser import block_spec as block_spec_lib
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class PullBlockSpecTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if config.enable_x64.value:
      self.skipTest('x64 not supported')

  def test_identity(self):

    def f(x):
      return x

    in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs, in_block_spec), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(1, 1, 1),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, in_type)
    )
    # We should have no new values or scalar prefetch values.
    self.assertEmpty(scalar_prefetch_values)
    self.assertEmpty(value_block_specs)
    self.assertEqual(in_block_spec.block_shape, (128, 128))
    self.assertEqual(in_block_spec.index_map(0, 1, 2), (0, 1))

    x = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, new_values, x),
        x,
    )

  def test_const(self):

    x = np.ones((512, 512), dtype=np.float32)

    def f():
      return x

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    self.assertEmpty(scalar_prefetch_values)
    self.assertEqual(value_block_specs[0].block_shape, (128, 128))
    self.assertEqual(value_block_specs[0].index_map(0, 1, 2), (0, 1))

    x_block = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn(
            (0, 0, 0),
            scalar_prefetch_values,
            (np.ones((128, 128), dtype=np.float32),),
        ),
        x_block,
    )

  @parameterized.parameters([jnp.exp, jnp.tanh])
  def test_elementwise(self, fn):

    def f(x):
      return fn(x)

    in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs, in_block_spec), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(1, 1, 1),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(
                0
            ),
        )(new_values, in_type)
    )
    self.assertEmpty(value_block_specs)
    self.assertEqual(in_block_spec.block_shape, (128, 128))
    self.assertEqual(in_block_spec.index_map(0, 1, 2), (0, 1))

    x = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (), x),
        fn(x),
    )

  @parameterized.parameters([jnp.exp, jnp.tanh])
  def test_elementwise_bias(self, fn):

    b = np.ones((512, 512), dtype=np.float32)

    def f(x):
      return fn(x) + b

    in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs, in_block_spec), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(1, 1, 1),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, in_type)
    )
    self.assertLen(value_block_specs, 1)
    self.assertEqual(value_block_specs[0].block_shape, (128, 128))
    self.assertEqual(value_block_specs[0].index_map(0, 1, 2), (0, 1))
    self.assertEqual(in_block_spec.block_shape, (128, 128))
    self.assertEqual(in_block_spec.index_map(0, 1, 2), (0, 1))

    x = np.ones((128, 128), dtype=np.float32)
    b = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (b,), x),
        fn(x) + b,
    )

  @parameterized.product(
      fn=[lax.mul, lax.add, lax.sub, lax.div, lax.max, lax.lt, lax.eq, lax.gt],
  )
  def test_binop(self, fn):

    def f(x, y):
      return fn(x, y)

    in_type = (
        jax.ShapeDtypeStruct((512, 512), jnp.float32),
        jax.ShapeDtypeStruct((512, 512), jnp.float32),
    )
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, *in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs, *in_block_specs), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(1, 1, 1),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, *in_type)
    )
    self.assertEmpty(value_block_specs)
    self.assertLen(in_block_specs, 2)
    x_block_spec, y_block_spec = in_block_specs
    self.assertEqual(x_block_spec.block_shape, (128, 128))
    self.assertEqual(
        x_block_spec.index_map(0, 1, 2), block_spec.index_map(0, 1, 2)
    )
    self.assertEqual(y_block_spec.block_shape, (128, 128))
    self.assertEqual(
        y_block_spec.index_map(0, 1, 2), block_spec.index_map(0, 1, 2)
    )

    x = np.ones((128, 128), dtype=np.float32)
    y = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, new_values, x, y),
        fn(x, y),
    )

  def test_slice(self):
    x = jax.random.normal(jax.random.key(0), (4, 512, 512), dtype=np.float32)

    def f():
      return jax.lax.slice(x, (1, 0, 0), (2, 512, 512))

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((1, 128, 128), lambda i, j, k: (0, i, j))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(0),
    )(new_values)

    self.assertLen(value_block_specs, 1)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)
    x_block_spec = value_block_specs[0]
    self.assertEqual(x_block_spec.block_shape, (1, 128, 128))
    self.assertEqual(x_block_spec.index_map(4, 2, 3), (1, 4, 2))

    x = np.ones((1, 128, 128), dtype=np.float32)
    # Slice doesn't change value after pulling block spec.
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (x,)), x
    )

  def test_squeeze(self):

    x = jax.random.normal(jax.random.key(0), (1, 512, 512), dtype=np.float32)

    def f():
      return jnp.squeeze(x, axis=0)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)
    x_block_spec = value_block_specs[0]
    self.assertEqual(x_block_spec.block_shape, (None, 128, 128))
    self.assertEqual(x_block_spec.index_map(4, 2, 3), (0, 4, 2))

    x = np.ones((128, 128), dtype=np.float32)
    # Squeeze doesn't change value after pulling block spec and the squeezed
    # dimensions are removed.
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (x,)), x
    )

  def test_dynamic_slice_only(self):

    x = jax.random.normal(jax.random.key(0), (3, 4, 512, 512), dtype=np.float32)
    i = jnp.array(1, dtype=jnp.int32)
    j = jnp.array(2, dtype=jnp.int32)

    def f():
      return jax.lax.dynamic_slice(x, (i, j, 0, 0), (1, 1, 512, 512))

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertLen(scalar_prefetch_values, 2)

    block_spec = pl.BlockSpec(
        (1, 1, 128, 128), lambda i, j, k, *_: (0, 0, i, j)
    )
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    scalar_prefetch_values = jax.tree.map(
        lambda x: x[None], scalar_prefetch_values
    )
    self.assertLen(value_block_specs, 1)
    x_block_spec = value_block_specs[0]
    self.assertEqual(x_block_spec.block_shape, (1, 1, 128, 128))
    self.assertEqual(
        x_block_spec.index_map(0, 1, 2, *scalar_prefetch_values), (1, 2, 0, 1)
    )

    x = np.ones((1, 1, 128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (x,)), x
    )

  def test_dynamic_slice_squeeze(self):
    x = jax.random.normal(jax.random.key(0), (3, 4, 512, 512), dtype=np.float32)
    i = jnp.array(1, dtype=jnp.int32)
    j = jnp.array(2, dtype=jnp.int32)

    def f():
      return jnp.squeeze(
          jax.lax.dynamic_slice(x, (i, j, 0, 0), (1, 1, 512, 512)), axis=(0, 1)
      )

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertLen(scalar_prefetch_values, 2)
    scalar_prefetch_values = jax.tree.map(
        lambda x: x[None], scalar_prefetch_values
    )

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k, *_: (i, j))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    x_block_spec = value_block_specs[0]
    self.assertEqual(x_block_spec.block_shape, (None, None, 128, 128))
    self.assertEqual(
        x_block_spec.index_map(0, 1, 2, *scalar_prefetch_values), (1, 2, 0, 1)
    )

    x = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (x,)), x
    )

  def test_concatenate_spanning_blocks(self):
    x = jax.random.normal(jax.random.key(0), (256, 256), dtype=np.float32)
    y = jax.random.normal(jax.random.key(1), (256, 256), dtype=np.float32)

    def f(x, y):
      return jnp.concatenate([x, y], axis=0)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, x, y
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, *_: (i, j))
    kernel_fn, (value_block_specs, *in_block_specs), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(4, 2),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(
                0
            ),
        )(new_values, x, y)
    )
    self.assertEmpty(value_block_specs)
    self.assertLen(in_block_specs, 2)
    x_block_spec, y_block_spec = in_block_specs
    self.assertEqual(x_block_spec.block_shape, (128, 128))
    self.assertEqual(y_block_spec.block_shape, (128, 128))
    # Indices in this concatenate should be clamped depending on if the
    # block is OOB.
    self.assertEqual(
        x_block_spec.index_map(0, 0, scalar_prefetch_values), (0, 0)
    )
    self.assertEqual(
        x_block_spec.index_map(1, 0, scalar_prefetch_values), (1, 0)
    )
    self.assertEqual(
        x_block_spec.index_map(2, 0, scalar_prefetch_values), (1, 0)
    )
    self.assertEqual(
        x_block_spec.index_map(3, 0, scalar_prefetch_values), (1, 0)
    )
    self.assertEqual(
        y_block_spec.index_map(0, 0, scalar_prefetch_values), (0, 0)
    )
    self.assertEqual(
        y_block_spec.index_map(1, 0, scalar_prefetch_values), (0, 0)
    )
    self.assertEqual(
        y_block_spec.index_map(2, 0, scalar_prefetch_values), (0, 0)
    )
    self.assertEqual(
        y_block_spec.index_map(3, 0, scalar_prefetch_values), (1, 0)
    )

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=np.float32)
    y = jax.random.normal(jax.random.key(1), (128, 128), dtype=np.float32)
    # Evaluating should just select the valid block.
    np.testing.assert_array_equal(
        kernel_fn((0, 0), scalar_prefetch_values, (), x, y), x
    )
    np.testing.assert_array_equal(
        kernel_fn((1, 0), scalar_prefetch_values, (), x, y), x
    )
    np.testing.assert_array_equal(
        kernel_fn((2, 0), scalar_prefetch_values, (), x, y), y
    )
    np.testing.assert_array_equal(
        kernel_fn((3, 0), scalar_prefetch_values, (), x, y), y
    )

  def test_concatenate_full_blocks(self):
    x = jax.random.normal(jax.random.key(0), (256, 256), dtype=np.float32)
    y = jax.random.normal(jax.random.key(1), (256, 256), dtype=np.float32)

    def f(x, y):
      return jnp.concatenate([x, y], axis=0)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, x, y
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((512, 128), lambda i, j, *_: (i, j))
    kernel_fn, (value_block_specs, *in_block_specs), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(1, 2),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(
                0
            ),
        )(new_values, x, y)
    )
    self.assertEmpty(value_block_specs)
    self.assertLen(in_block_specs, 2)
    x_block_spec, y_block_spec = in_block_specs
    self.assertEqual(x_block_spec.block_shape, (256, 128))
    self.assertEqual(y_block_spec.block_shape, (256, 128))
    # Indices in this concatenate always just fetch the whole block on that
    # dimension.
    self.assertEqual(
        x_block_spec.index_map(0, 0, scalar_prefetch_values), (0, 0)
    )
    self.assertEqual(
        x_block_spec.index_map(0, 1, scalar_prefetch_values), (0, 1)
    )
    self.assertEqual(
        y_block_spec.index_map(0, 0, scalar_prefetch_values), (0, 0)
    )
    self.assertEqual(
        y_block_spec.index_map(0, 1, scalar_prefetch_values), (0, 1)
    )

    x = jax.random.normal(jax.random.key(0), (256, 128), dtype=np.float32)
    y = jax.random.normal(jax.random.key(1), (256, 128), dtype=np.float32)
    xy = jnp.concatenate([x, y], axis=0)
    # Evaluating should just select the valid block.
    np.testing.assert_array_equal(
        kernel_fn((0, 0), scalar_prefetch_values, (), x, y), xy
    )
    np.testing.assert_array_equal(
        kernel_fn((1, 0), scalar_prefetch_values, (), x, y), xy
    )

  def test_transpose_minor(self):
    x = jax.random.normal(jax.random.key(0), (512, 256), dtype=np.float32)

    def f():
      return jax.lax.transpose(x, (1, 0))

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 128), lambda i, j, k: (i, j))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    x_block_spec = value_block_specs[0]
    self.assertEqual(x_block_spec.block_shape, (128, 128))
    self.assertEqual(
        x_block_spec.index_map(0, 1, 2, *scalar_prefetch_values), (1, 0)
    )

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0), scalar_prefetch_values, (x,)), x.T
    )

  def test_transpose_major(self):
    x = jax.random.normal(jax.random.key(0), (2, 3, 512, 256), dtype=np.float32)

    def f():
      return jax.lax.transpose(x, (1, 0, 2, 3))

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec(
        (None, None, 128, 128), lambda i, j, k, l: (i, j, k, l)
    )
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    x_block_spec = value_block_specs[0]
    self.assertEqual(x_block_spec.block_shape, (None, None, 128, 128))
    self.assertEqual(
        x_block_spec.index_map(0, 1, 2, 3, *scalar_prefetch_values),
        (1, 0, 2, 3),
    )

    x = jax.random.normal(jax.random.key(0), (128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 0), scalar_prefetch_values, (x,)), x
    )

  def test_iota(self):

    def f():
      return jax.lax.broadcasted_iota(jnp.int32, (2, 2, 512, 512), 2)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec(
        (None, None, 128, 128), lambda i, j, k, l: (i, j, k, l)
    )
    kernel_fn, ((),), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(2, 2, 4, 4),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)

    x = jax.lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((1, 1, 0, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 1), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 1, 0), scalar_prefetch_values, ()), x + 128
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 3, 0), scalar_prefetch_values, ()), x + 128 * 3
    )

  def test_broadcast_scalar(self):

    def f():
      return jnp.full((1, 1, 512, 512), fill_value=1.2345, dtype=jnp.float32)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec(
        (None, None, 128, 128), lambda i, j, k, l: (i, j, k, l)
    )
    kernel_fn, ((),), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(2, 2, 4, 4),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)

    x = jnp.full((128, 128), fill_value=1.2345, dtype=jnp.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((1, 1, 0, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 1), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 1, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 3, 0), scalar_prefetch_values, ()), x
    )

  def test_broadcast_scalar_with_prefetch(self):

    a = jnp.array(1.2345)

    def f():
      return jnp.full((1, 1, 512, 512), fill_value=a, dtype=jnp.float32)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertEmpty(new_values)
    self.assertLen(scalar_prefetch_values, 1)

    block_spec = pl.BlockSpec(
        (None, None, 128, 128), lambda i, j, k, l, *_: (i, j, k, l)
    )
    kernel_fn, ((),), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(2, 2, 4, 4),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    scalar_prefetch_values = jax.tree.map(
        lambda x: x[None], scalar_prefetch_values
    )

    x = jnp.full((128, 128), fill_value=a, dtype=jnp.float32)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((1, 1, 0, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 1), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 1, 0), scalar_prefetch_values, ()), x
    )
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 3, 0), scalar_prefetch_values, ()), x
    )

  @parameterized.parameters(
      (False, False), (False, True), (True, False), (True, True)
  )
  def test_broadcast_array(self, bcast0, bcast1):

    x = jnp.ones((1 if bcast0 else 512, 1 if bcast1 else 512))

    def f():
      return jax.lax.broadcast_in_dim(x, (2, 2, 512, 512), (2, 3))

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_shape = (None, 1, 128, 128)
    block_spec = pl.BlockSpec(block_shape, lambda i, j, k, l: (i, j, k, l))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(2, 2, 4, 4),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    x_index_map = value_block_specs[0].index_map
    self.assertEqual(
        x_index_map(0, 0, 1, 2), (0 if bcast0 else 1, 0 if bcast1 else 2)
    )
    self.assertEqual(
        x_index_map(1, 2, 3, 3), (0 if bcast0 else 3, 0 if bcast1 else 3)
    )

    block_shape = (1 if bcast0 else 128, 1 if bcast1 else 128)
    self.assertEqual(block_shape, value_block_specs[0].block_shape)
    x = jnp.full(block_shape, fill_value=1.2345, dtype=jnp.float32)
    y = jax.lax.broadcast_in_dim(x, (1, 128, 128), (1, 2))
    np.testing.assert_array_equal(kernel_fn((0, 0, 0, 0), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((1, 1, 0, 0), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((0, 0, 0, 1), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((0, 0, 1, 0), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((0, 0, 3, 0), (), (x,)), y)

  @parameterized.parameters(0, 1, 2, 3)
  def test_broadcast_1d_array(self, bcast_dim):
    full_shape = (2, 2, 512, 512)
    x = jnp.ones((full_shape[bcast_dim],))

    def f():
      return jax.lax.broadcast_in_dim(x, full_shape, (bcast_dim,))

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    block_shape = (None, 1, 128, 128)
    block_spec = pl.BlockSpec(block_shape, lambda i, j, k, l: (i, j, k, l))
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(2, 2, 4, 4),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    x_index_map = value_block_specs[0].index_map
    self.assertEqual(x_index_map(0, 0, 1, 2), ((0, 0, 1, 2)[bcast_dim],))
    self.assertEqual(x_index_map(1, 2, 3, 3), ((1, 2, 3, 3)[bcast_dim],))

    if block_shape[bcast_dim] is None:
      x = jnp.ones(())
      y = jax.lax.broadcast_in_dim(x, (1, 128, 128), ())
    else:
      x = jnp.arange(block_shape[bcast_dim] or 1, dtype=jnp.float32)
      y = jax.lax.broadcast_in_dim(x, (1, 128, 128), (bcast_dim - 1,))

    np.testing.assert_array_equal(kernel_fn((0, 0, 0, 0), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((1, 1, 0, 0), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((0, 0, 0, 1), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((0, 0, 1, 0), (), (x,)), y)
    np.testing.assert_array_equal(kernel_fn((0, 0, 3, 0), (), (x,)), y)

  def test_element_indexing(self):

    x = np.zeros((512, 512), dtype=np.float32)

    def f():
      return x

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertEmpty(scalar_prefetch_values)

    # Block spec with an offset on the first dimension
    block_spec = pl.BlockSpec(
        (pl.Element(128, (0, 16)), 128), lambda i, j, k: (128 * i + 16, j)
    )
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(1, 1, 1),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    self.assertEmpty(scalar_prefetch_values)
    self.assertEqual(
        value_block_specs[0].block_shape, (pl.Element(128, (0, 16)), 128)
    )
    self.assertEqual(value_block_specs[0].index_map(0, 1, 2), (16, 1))
    self.assertEqual(value_block_specs[0].index_map(1, 1, 2), (128 + 16, 1))

    x_block = np.ones((128, 128), dtype=np.float32)
    np.testing.assert_array_equal(
        kernel_fn(
            (0, 0, 0),
            scalar_prefetch_values,
            (np.ones((128, 128), dtype=np.float32),),
        ),
        x_block,
    )

  def test_basic_reshape_sublanes_to_lanes(self):

    def f(x):
      return x.reshape((512, 2048))

    in_type = jax.ShapeDtypeStruct((512, 16, 128), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((256, 1024), lambda i, j, k: (i, k))
    kernel_fn, (value_block_specs, x_block_spec), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(2, 3, 4),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, in_type)
    )
    self.assertEmpty(value_block_specs)
    self.assertEqual(x_block_spec.index_map(0, 1, 2), (0, 2, 0))
    self.assertEqual(x_block_spec.index_map(3, 2, 1), (3, 1, 0))

    x = jnp.arange((256 * 1024), dtype=jnp.float32).reshape((256, 8, 128))
    y = kernel_fn((0, 1, 2), scalar_prefetch_values, (), x)
    np.testing.assert_array_equal(y, x.reshape((256, 1024)))

  def test_basic_reshape_lanes_to_sublanes(self):

    def f(x):
      return x.reshape((512, 32, 128))

    in_type = jax.ShapeDtypeStruct((512, 4096), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((256, 8, 128), lambda i, j, k: (i, k, 0))
    kernel_fn, (value_block_specs, x_block_spec), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(2, 3, 4),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, in_type)
    )
    self.assertEmpty(value_block_specs)
    self.assertEqual(x_block_spec.index_map(0, 1, 2), (0, 2))
    self.assertEqual(x_block_spec.index_map(3, 2, 1), (3, 1))

    x = jnp.arange((256 * 1024), dtype=jnp.float32).reshape((256, 1024))
    y = kernel_fn((0, 1, 2), scalar_prefetch_values, (), x)
    np.testing.assert_array_equal(y, x.reshape((256, 8, 128)))

    block_spec = pl.BlockSpec((256, 4, 256), lambda i, j, k: (i, j, k))
    with self.assertRaises(NotImplementedError):
      _ = block_spec_lib.pull_block_spec(
          f2,
          block_spec,
          grid=(2, 3, 4),
          scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
      )(new_values, in_type)

  def test_basic_swap(self):
    value = jnp.arange((512 * 1024), dtype=jnp.int32).reshape((512, 1024)) * 2
    x = jnp.zeros((256, 512), dtype=jnp.int32)

    def outer(refs):
      ref, y_ref = refs

      def f(x):
        return ref.swap(x)

      in_type = jax.ShapeDtypeStruct((512, 1024), jnp.int32)
      f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
          f, in_type
      )
      self.assertLen(new_values, 1)  # Captures Ref
      self.assertEmpty(scalar_prefetch_values)

      block_spec = pl.BlockSpec((256, 512), lambda i, j, k: (i, k))
      kernel_fn, (value_block_specs, x_block_spec), _ = (
          block_spec_lib.pull_block_spec(
              f2,
              block_spec,
              grid=(2, 3, 4),
              scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
          )(new_values, in_type)
      )
      self.assertLen(value_block_specs, 1)
      self.assertEqual(x_block_spec.index_map(0, 1, 2), (0, 2))
      self.assertEqual(x_block_spec.index_map(3, 2, 1), (3, 1))

      y_ref[...] = kernel_fn((0, 1, 1), scalar_prefetch_values, (ref,), x)

    y = jnp.zeros((256, 512), jnp.int32)
    _, y = pl.run_state(outer)((value, y))
    np.testing.assert_array_equal(y, value[:256, 512:1024])

  def test_basic_get(self):
    value = jnp.arange((512 * 1024), dtype=jnp.int32).reshape((512, 1024)) * 2

    def outer(refs):
      ref, y_ref = refs

      def f():
        return ref.get()

      block_spec = pl.BlockSpec((256, 512), lambda i, j, k: (i, k))
      kernel_fn, (), _ = block_spec_lib.pull_block_spec(
          f,
          block_spec,
          grid=(2, 3, 4),
          scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
      )()
      y_ref[...] = kernel_fn((0, 1, 1), ())

    y = jnp.zeros((256, 512), jnp.int32)
    _, y = pl.run_state(outer)((value, y))
    np.testing.assert_array_equal(y, value[:256, 512:1024])

  def test_get_with_squeezed_block_spec(self):
    value = (
        jnp.arange((4 * 512 * 1024), dtype=jnp.int32).reshape((4, 512, 1024))
        * 2
    )

    def outer(refs):
      ref, y_ref = refs

      def f():
        return ref.get()

      block_spec = pl.BlockSpec(
          (pl.Squeezed(), 256, 512), lambda i, j, k: (j, i, k)
      )
      kernel_fn, (), _ = block_spec_lib.pull_block_spec(
          f,
          block_spec,
          grid=(2, 3, 4),
          scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
      )()
      y_ref[...] = kernel_fn((0, 3, 1), ())

    y = jnp.zeros((256, 512), jnp.int32)
    _, y = pl.run_state(outer)((value, y))
    np.testing.assert_array_equal(y, value[3, :256, 512:1024])

  def test_get_with_squeezed_indexer(self):
    value = (
        jnp.arange((4 * 512 * 1024), dtype=jnp.int32).reshape((4, 512, 1024))
        * 2
    )

    def outer(refs):
      ref, y_ref = refs

      def f():
        return ref[3]

      block_spec = pl.BlockSpec((256, 512), lambda i, j, k: (i, k))
      kernel_fn, (), _ = block_spec_lib.pull_block_spec(
          f,
          block_spec,
          grid=(2, 3, 4),
          scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
      )()
      y_ref[...] = kernel_fn((0, 2, 1), ())

    y = jnp.zeros((256, 512), jnp.int32)
    _, y = pl.run_state(outer)((value, y))
    np.testing.assert_array_equal(y, value[3, :256, 512:1024])

  def test_random_noise(self):
    key = jax.random.key(0, impl='threefry2x32')

    def f(key):
      return jax.random.uniform(key, (512, 512), dtype=jnp.float32)

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, key
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec((128, 256), lambda i, j: (i, j))
    kernel_fn, (value_block_specs, key_block_spec), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(4, 2),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, key)
    )
    self.assertEmpty(value_block_specs)
    self.assertEqual(key_block_spec.memory_space, pl.MemorySpace.KEY)
    self.assertIsNone(key_block_spec.block_shape)

    @jax.jit
    def gen(idx):
      k = key
      for i in idx:
        k = jax.random.fold_in(k, i)
      return jax.random.uniform(k, (128, 256), dtype=jnp.float32)

    for i in range(4):
      for j in range(2):
        out = kernel_fn((i, j), scalar_prefetch_values, (), key)
        out_ref = gen((i, j))
        np.testing.assert_array_equal(out, out_ref)


class PullBlockSpecHOPTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if config.enable_x64.value:
      self.skipTest('x64 not supported')

  def test_jit(self):
    def f(x):
      return jax.jit(jnp.sin)(x)

    in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec(
        (None, 1, 128, 128), lambda i, j, k, l, _: (i, j, k, l)
    )
    kernel_fn, (value_block_specs, *in_block_specs), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(2, 2, 4, 4),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, in_type)
    )
    self.assertEmpty(value_block_specs)
    x_block_spec = in_block_specs[0]
    self.assertEqual(x_block_spec.index_map(0, 0, 1, 2, ()), (0, 0, 1, 2))
    self.assertEqual(x_block_spec.index_map(1, 2, 3, 3, ()), (1, 2, 3, 3))

    x = jax.random.normal(jax.random.key(0), (1, 128, 128), dtype=np.float32)
    sin_x = jnp.sin(x)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 0), scalar_prefetch_values, (), x), sin_x
    )

  def test_custom_jvp(self):
    def f(x):
      return jax.nn.relu(x)

    in_type = jax.ShapeDtypeStruct((512, 512), jnp.float32)
    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(
        f, in_type
    )
    self.assertEmpty(new_values)
    self.assertEmpty(scalar_prefetch_values)

    block_spec = pl.BlockSpec(
        (None, 1, 128, 128), lambda i, j, k, l, _: (i, j, k, l)
    )
    kernel_fn, (value_block_specs, *in_block_specs), _ = (
        block_spec_lib.pull_block_spec(
            f2,
            block_spec,
            grid=(2, 2, 4, 4),
            scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
        )(new_values, in_type)
    )
    self.assertEmpty(value_block_specs)
    x_block_spec = in_block_specs[0]
    self.assertEqual(x_block_spec.index_map(0, 0, 1, 2, ()), (0, 0, 1, 2))
    self.assertEqual(x_block_spec.index_map(1, 2, 3, 3, ()), (1, 2, 3, 3))

    x = jax.random.normal(jax.random.key(0), (1, 128, 128), dtype=np.float32)
    relu_x = jax.nn.relu(x)
    np.testing.assert_array_equal(
        kernel_fn((0, 0, 0, 0), scalar_prefetch_values, (), x), relu_x
    )

  def test_pull_block_spec_handles_closed_over_constants(self):
    x = jnp.ones((2, 512, 512))
    i = jnp.array(1)

    def f():
      return x[i]

    f2, new_values, scalar_prefetch_values = block_spec_lib.get_fusion_values(f)
    self.assertLen(new_values, 1)
    self.assertLen(scalar_prefetch_values, 1)

    block_spec = pl.BlockSpec(
        (None, 1, 128, 128), lambda i, j, k, l, _: (i, j, k, l)
    )
    kernel_fn, (value_block_specs,), _ = block_spec_lib.pull_block_spec(
        f2,
        block_spec,
        grid=(2, 2, 4, 4),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values)
    self.assertLen(value_block_specs, 1)
    scalar_prefetch_values = jax.tree.map(
        lambda x: x[None], scalar_prefetch_values
    )
    fn = lambda x: kernel_fn((0, 0, 0, 0), scalar_prefetch_values, x)
    new_values_type = (jax.ShapeDtypeStruct((1, 128, 128), jnp.float32),)
    # Try pulling again
    # This should not raise an error.
    _ = block_spec_lib.pull_block_spec(
        fn,
        block_spec,
        grid=(1,),
        scalar_prefetch_handler=block_spec_lib.make_scalar_prefetch_handler(),
    )(new_values_type)


class PushBlockSpecTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if config.enable_x64.value:
      self.skipTest('x64 not supported')

  def test_jit(self):

    def f(x):
      return jax.jit(jnp.sin)(x)

    block_spec = pl.BlockSpec(
        (None, 1, 128, 128), lambda i, j, k, l, _: (i, l, k, j)
    )
    x_type = jax.ShapeDtypeStruct((1, 1, 512, 512), jnp.float32)
    out_block_spec = block_spec_lib.push_block_spec(f, block_spec)(x_type)
    self.assertEqual(out_block_spec.block_shape, block_spec.block_shape)

  def test_custom_jvp(self):
    def f(x):
      return jax.nn.relu(x)

    x_type = jax.ShapeDtypeStruct((1, 1, 512, 512), jnp.float32)
    block_spec = pl.BlockSpec(
        (None, 1, 128, 128), lambda i, j, k, l, _: (i, l, k, j)
    )
    out_block_spec = block_spec_lib.push_block_spec(f, block_spec)(x_type)
    self.assertEqual(out_block_spec.block_shape, block_spec.block_shape)

  def test_push_reshape_lanes_to_sublanes(self):
    def f(x):
      return x.reshape((512, 32, 128))

    x_type = jax.ShapeDtypeStruct((512, 4096), jnp.float32)
    block_spec = pl.BlockSpec(
        (256, 1024), lambda i, j, k: (i, k)
    )
    out_block_spec = block_spec_lib.push_block_spec(f, block_spec)(x_type)
    self.assertEqual(out_block_spec.block_shape, (256, 8, 128))
    self.assertTupleEqual(out_block_spec.index_map(0, 1, 2), (0, 2, 0))
    self.assertEqual(out_block_spec.index_map(3, 2, 1), (3, 1, 0))

    def f(x):
      return x.reshape((512, 16, 256))

    x_type = jax.ShapeDtypeStruct((512, 4096), jnp.float32)
    block_spec = pl.BlockSpec(
        (256, 1024), lambda i, j, k: (i, k)
    )
    out_block_spec = block_spec_lib.push_block_spec(f, block_spec)(x_type)
    self.assertEqual(out_block_spec.block_shape, (256, 4, 256))
    self.assertTupleEqual(out_block_spec.index_map(0, 1, 2), (0, 2, 0))
    self.assertEqual(out_block_spec.index_map(3, 2, 1), (3, 1, 0))



if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
