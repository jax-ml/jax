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

"""Fusible matmul test."""

import functools
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
from jax._src.pallas import fuser
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()

jit_no_excess_precision = functools.partial(
    jax.jit, compiler_options={'xla_allow_excess_precision': False}
)


@jit_no_excess_precision
def mm_ref(x, y):
  return jnp.dot(x, y, preferred_element_type=jnp.float32)


def matmul_kernel(
    x_scalar_prefetch,
    y_scalar_prefetch,
    z_scalar_prefetch,
    x_value_refs,
    y_value_refs,
    z_value_refs,
    o_ref,
    acc_ref,
    *,
    x_fn: Any,
    y_fn: Any,
    z_fn: Any,
    out_dtype: jnp.dtype,
):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  pids = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

  x_values = jax.tree.map(lambda ref: ref.get(), x_value_refs)
  x = x_fn(pids, scalar_prefetch, x_values)
  y_values = jax.tree.map(lambda ref: ref.get(), y_value_refs)
  y = y_fn(pids, scalar_prefetch, y_values)
  acc_ref[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)

  @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
  def _():
    acc = acc_ref[...].astype(out_dtype)
    z_values = jax.tree.map(lambda ref: ref.get(), z_value_refs)
    out = z_fn(pids, scalar_prefetch, z_values, acc)
    jax.tree.map(lambda ref, x: ref.set(x), o_ref, out)


def _fusible_matmul(
    x: fuser.Fusion[[], jax.Array],  # pytype: disable=invalid-annotation
    y: fuser.Fusion[[], jax.Array],  # pytype: disable=invalid-annotation
    z: fuser.Fusion[[jax.Array], jax.Array] | None,  # pytype: disable=invalid-annotation
    *,
    bm: int,
    bk: int,
    bn: int,
    interpret: bool,
    debug: bool,
) -> jax.Array:
  m, k = x.shape
  k_, n = y.shape
  out_dtype = jnp.float32
  z_type = jax.ShapeDtypeStruct((m, n), dtype=out_dtype)
  if not z:
    z = lambda x: x
  if k != k_:
    raise ValueError(f'X and Y shapes must be compatible. Got {k} != {k_}')

  assert m % bm == 0
  assert k % bk == 0
  assert n % bn == 0
  grid = (m // bm, n // bn, k // bk)

  def x_index_map(i, j, k, *_):
    del j
    return i, k

  x_block_spec = pl.BlockSpec(block_shape=(bm, bk), index_map=x_index_map)

  def y_index_map(i, j, k, *_):
    del i
    return k, j

  y_block_spec = pl.BlockSpec(block_shape=(bk, bn), index_map=y_index_map)

  def z_index_map(i, j, k, *_):
    del k
    return i, j

  z_block_spec = pl.BlockSpec(block_shape=(bm, bn), index_map=z_index_map)
  dimension_semantics = (pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)

  z_out_type = jax.eval_shape(z, z_type)

  # First thing we do is extract the values from the fusions. These will be
  # values that are passed in directly and values that are passed in via
  # scalar prefetch.
  x_fn, x_values, x_scalar_prefetch = fuser.get_fusion_values(x)
  y_fn, y_values, y_scalar_prefetch = fuser.get_fusion_values(y)
  z_fn, z_values, z_scalar_prefetch = fuser.get_fusion_values(z, z_type)

  # We construct the set of scalar prefetch arguments that will be passed to
  # the kernel.
  scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

  x_fn, (x_value_block_specs,), _ = fuser.pull_block_spec(
      x_fn,
      x_block_spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
      grid=grid,
  )(x_values)

  y_fn, (y_value_block_specs,), _ = fuser.pull_block_spec(
      y_fn,
      y_block_spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(1),
      grid=grid,
  )(y_values)

  z_out_block_spec = fuser.push_block_spec(z, z_block_spec)(z_type)
  z_fn, (z_value_block_specs, _), _ = fuser.pull_block_spec(
      z_fn,
      z_out_block_spec,
      scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(2),
      grid=grid,
  )(z_values, z_type)

  # TODO(sharadmv): This is a hack. We should be able to pass in the scalar
  # prefetch arguments directly to the kernel but don't have Mosaic support atm.
  scalar_prefetch = jax.tree.map(lambda x: x[None], scalar_prefetch)

  return pl.pallas_call(
      functools.partial(
          matmul_kernel,
          x_fn=x_fn,
          y_fn=y_fn,
          z_fn=z_fn,
          out_dtype=out_dtype,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetch),
          grid=grid,
          scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
          in_specs=[
              x_value_block_specs,
              y_value_block_specs,
              z_value_block_specs,
          ],
          out_specs=[z_out_block_spec],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=dimension_semantics,
      ),
      out_shape=[z_out_type],
      interpret=interpret,
      debug=debug,
  )(
      *scalar_prefetch,
      x_values,
      y_values,
      z_values,
  )[0]


def fusible_matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    debug: bool = False,
    interpret: bool = False,
) -> jax.Array:
  return fuser.fusible(
      functools.partial(
          _fusible_matmul,
          bm=bm,
          bk=bk,
          bn=bn,
          interpret=interpret,
          debug=debug,
      )
  )(x, y)


class FusibleMatmulTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works with TPU v4+')
    super().setUp()

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)
    np.testing.assert_allclose(
        jax.jit(fusible_matmul)(x, y), mm_ref(x, y), atol=5e-5
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_activation(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_relu(x, y):
      x = fusible_matmul(x, y)
      x = jnp.maximum(x, 0.0)
      return x

    np.testing.assert_allclose(
        matmul_relu(x, y), jax.nn.relu(mm_ref(x, y)), atol=5e-5
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_bias(self, dtype):
    if dtype == 'bfloat16' and not jtu.is_device_tpu_at_least(5):
      self.skipTest('bfloat16 bcast not supported on TPU generations < 5')
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)
    b = jax.random.normal(k2, (1, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_bias(x, y, b):
      x = fusible_matmul(x, y).astype(dtype) + b
      x = jnp.maximum(x, 0.0)
      return x

    np.testing.assert_allclose(
        matmul_bias(x, y, b),
        jax.nn.relu(mm_ref(x, y).astype(dtype) + b),
        atol=5e-5 if dtype == 'float32' else 0.5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_slice(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (2, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y):
      x = fusible_matmul(x, y[1])
      return x

    np.testing.assert_allclose(matmul_slice(x, y), mm_ref(x, y[1]), atol=5e-5)

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_dynamic_slice(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (2, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y, i):
      x = fusible_matmul(x, y[i])
      return x

    np.testing.assert_allclose(
        matmul_slice(x, y, 1), mm_ref(x, y[1]), atol=5e-5
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_dynamic_slice_bias(self, dtype):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (3, 512, 512), dtype)
    b = jax.random.normal(k2, (2, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y, b, i, j):
      x = fusible_matmul(x, y[j]).astype(dtype) + b[i]
      return x

    np.testing.assert_allclose(
        matmul_slice(x, y, b, 1, 2),
        mm_ref(x, y[2]).astype(dtype) + b[1],
        atol=5e-5 if dtype == 'float32' else 0.5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_multi_slice(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (2, 3, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y):
      x = fusible_matmul(x, y[1, 1])
      return x

    np.testing.assert_allclose(
        matmul_slice(x, y), mm_ref(x, y[1, 1]), atol=5e-5
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_multiple_slices(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (2, 3, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y):
      x = fusible_matmul(x, y[1][1])
      return x

    np.testing.assert_allclose(
        matmul_slice(x, y), mm_ref(x, y[1, 1]), atol=5e-5
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_multiple_dynamic_slices(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (2, 3, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y, i, j):
      x = fusible_matmul(x, y[i][j])
      return x

    for i in range(2):
      for j in range(3):
        np.testing.assert_allclose(
            matmul_slice(x, y, i, j), mm_ref(x, y[i, j]), atol=5e-5
        )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_mixed_slices(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (4, 2, 3, 512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y, i, j):
      x = fusible_matmul(x, y[2][i, j])
      return x

    for i in range(2):
      for j in range(3):
        np.testing.assert_allclose(
            matmul_slice(x, y, i, j), mm_ref(x, y[2, i, j]), atol=5e-5
        )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_with_multiple_mixed_slices_and_bias(self, dtype):
    if dtype == 'bfloat16' and not jtu.is_device_tpu_at_least(5):
      self.skipTest('bfloat16 bcast not supported on TPU generations < 5')
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (4, 4, 512, 512), dtype)
    y = jax.random.normal(k1, (4, 2, 3, 512, 512), dtype)
    b = jax.random.normal(k2, (4, 4, 1, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_slice(x, y, b, i, j, k):
      x = fusible_matmul(x[k][3], y[2][i, j]).astype(dtype)
      return x + b[i, j]

    @jit_no_excess_precision
    def matmul_slice_ref(x, y, b, i, j, k):
      x = mm_ref(x[k][3], y[2][i, j]).astype(dtype)
      return x + b[i, j]

    for i in range(2):
      for j in range(3):
        for k in range(4):
          np.testing.assert_allclose(
              matmul_slice(x, y, b, i, j, k),
              matmul_slice_ref(x, y, b, i, j, k),
              atol=5e-5 if dtype == 'float32' else 0.5,
          )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_input_concat_output(self, dtype):
    self.skipTest('select_n does not support more than 3 elements')
    # TODO(sharadmv): fix this test
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k0, (128, 128), dtype)
    y1 = jax.random.normal(k1, (128, 256), dtype)
    y2 = jax.random.normal(k2, (128, 256), dtype)
    y3 = jax.random.normal(k3, (128, 256), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_concat(x, ys):
      y = jnp.concatenate(ys, axis=1)
      x = fusible_matmul(x, y)
      return x

    @jax.jit
    def matmul_concat_ref(x, ys):
      y = jnp.concatenate(ys, axis=1)
      return mm_ref(x, y)

    ys = [y1, y2, y3]
    np.testing.assert_allclose(
        matmul_concat(x, ys),
        matmul_concat_ref(x, ys),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_input_concat_contract(self, dtype):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (128, 256), dtype)
    y1 = jax.random.normal(k1, (128, 128), dtype)
    y2 = jax.random.normal(k2, (128, 128), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_concat(x, ys):
      y = jnp.concatenate(ys, axis=0)
      x = fusible_matmul(x, y)
      return x

    @jit_no_excess_precision
    def matmul_concat_ref(x, ys):
      y = jnp.concatenate(ys, axis=0)
      return mm_ref(x, y)

    ys = [y1, y2]
    np.testing.assert_allclose(
        matmul_concat(x, ys),
        matmul_concat_ref(x, ys),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_double_concat(self, dtype):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k0, (128, 256), dtype)
    y1 = jax.random.normal(k1, (128, 128), dtype)
    y2 = jax.random.normal(k2, (128, 128), dtype)
    y3 = jax.random.normal(k3, (256, 128), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_concat(x, ys, y3):
      y = jnp.concatenate(ys, axis=0)
      y = jnp.concatenate([y, y3], axis=1)
      x = fusible_matmul(x, y)
      return x

    @jit_no_excess_precision
    def matmul_concat_ref(x, ys, y3):
      y = jnp.concatenate(ys, axis=0)
      y = jnp.concatenate([y, y3], axis=1)
      return mm_ref(x, y)

    ys = [y1, y2]
    np.testing.assert_allclose(
        matmul_concat(x, ys, y3),
        matmul_concat_ref(x, ys, y3),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_slice_concat(self, dtype):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (128, 256), dtype)
    y1 = jax.random.normal(k1, (128, 128), dtype)
    y2 = jax.random.normal(k2, (4, 128, 128), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_concat(x, y1, y2):
      y = jnp.concatenate([y1, y2[3]], axis=0)
      x = fusible_matmul(x, y)
      return x

    @jit_no_excess_precision
    def matmul_concat_ref(x, y1, y2):
      y = jnp.concatenate([y1, y2[3]], axis=0)
      return mm_ref(x, y)

    np.testing.assert_allclose(
        matmul_concat(x, y1, y2),
        matmul_concat_ref(x, y1, y2),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_slice_concat_slice(self, dtype):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (128, 256), dtype)
    y1 = jax.random.normal(k1, (3, 128, 128), dtype)
    y2 = jax.random.normal(k2, (4, 3, 128, 128), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_concat(x, y1, y2):
      y = jnp.concatenate([y1, y2[3]], axis=1)[1]
      x = fusible_matmul(x, y)
      return x

    @jit_no_excess_precision
    def matmul_concat_ref(x, y1, y2):
      y = jnp.concatenate([y1, y2[3]], axis=1)[1]
      return mm_ref(x, y)

    np.testing.assert_allclose(
        matmul_concat(x, y1, y2),
        matmul_concat_ref(x, y1, y2),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_dynamic_slice_concat(self, dtype):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k0, (128, 256), dtype)
    y1 = jax.random.normal(k1, (2, 128, 128), dtype)
    y2 = jax.random.normal(k2, (4, 2, 128, 128), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_concat(x, y1, y2, i, j):
      y = jnp.concatenate([y1, y2[i]], axis=1)[j]
      x = fusible_matmul(x, y)
      return x

    @jit_no_excess_precision
    def matmul_concat_ref(x, y1, y2, i, j):
      y = jnp.concatenate([y1, y2[i]], axis=1)[j]
      return mm_ref(x, y)

    np.testing.assert_allclose(
        matmul_concat(x, y1, y2, 3, 1),
        matmul_concat_ref(x, y1, y2, 3, 1),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_no_following_ops(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (512, 256), dtype)

    def matmul(impl, x, y):
      # This transpose gets fused in
      z = impl(x, y.T)
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    np.testing.assert_allclose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_then_add(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (512, 256), dtype)

    def matmul(impl, x, y):
      # This transpose gets fused in as well (special handling for this case)
      z = impl(x, y.T + 1.0)
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_after_slice(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (2, 512, 256), dtype)

    def matmul(impl, x, y):
      # This transpose gets fused in as well
      z = impl(x, y[1].T)
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_before_slice(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (2, 512, 256), dtype)

    def matmul(impl, x, y):
      # This transpose gets fused in as well
      z = impl(x, y.swapaxes(-1, -2)[1])
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_major_dim(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (2, 3, 256, 512), dtype)

    def matmul(impl, x, y):
      # This transpose gets fused in but the block spec takes care of it
      z = impl(x, y.swapaxes(0, 1)[1, 0])
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_transpose(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (256, 512), dtype)

    def matmul(impl, x, y):
      # This transpose gets converted into a no-op.
      z = impl(x, y.T.T)
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_rhs_transpose_add_transpose(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (128, 256), dtype)
    y = jax.random.normal(k1, (256, 512), dtype)

    def matmul(impl, x, y):
      z = impl(x, jnp.tanh((y.T + 1).T))
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jit_no_excess_precision(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_lhs_transpose(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (256, 512), dtype)
    y = jax.random.normal(k1, (256, 384), dtype)

    def matmul(impl, x, y):
      # This transpose gets converted into a no-op.
      z = impl(x.T, y)
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bm=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_lhs_transpose_add(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (256, 512), dtype)
    y = jax.random.normal(k1, (256, 384), dtype)

    def matmul(impl, x, y):
      # This transpose gets converted into a no-op.
      z = impl(x.T + 1, y)
      return z

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bm=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_out_transpose(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (256, 256), dtype)
    y = jax.random.normal(k1, (256, 512), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      # This transpose is executed.
      return z.T

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )

  @parameterized.parameters('float32', 'bfloat16')
  def test_matmul_out_transpose_mul(self, dtype):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (256, 256), dtype)
    y = jax.random.normal(k1, (256, 512), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      # This transpose is executed.
      return z.T * 2

    impl = fuser.fuse(
        functools.partial(matmul, functools.partial(fusible_matmul, bn=256))
    )
    ref = functools.partial(matmul, mm_ref)

    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=5e-5,
    )


def dot_ref(x, y, *, bm=128, bk=128, bn=128):
  # Meant to precisely mimic the numerics of the kernel
  out_dtype = jnp.float32
  m, k = x.shape
  n = y.shape[-1]
  out = jnp.zeros((m, n), dtype=out_dtype)
  assert m % bm == 0
  assert k % bk == 0
  assert n % bn == 0

  def body_m(i, out):
    def body_n(j, out):
      acc = jnp.zeros((bm, bn), dtype=jnp.float32)

      def body_k(k, acc):
        lhs = jax.lax.dynamic_slice(x, (i * bm, k * bk), (bm, bk))
        rhs = jax.lax.dynamic_slice(y, (k * bk, j * bn), (bk, bn))
        acc += jnp.dot(lhs, rhs, preferred_element_type=jnp.float32)
        return acc

      acc = jax.lax.fori_loop(0, k // bk, body_k, acc)
      return jax.lax.dynamic_update_slice(
          out, acc.astype(out.dtype), (i * bm, j * bn)
      )

    return jax.lax.fori_loop(0, n // bn, body_n, out)

  return jax.lax.fori_loop(0, m // bm, body_m, out)


class ExcessPrecisionTest(jtu.JaxTestCase):

  def setUp(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('Only works with TPU v4+')
    super().setUp()

  def test_matmul_bf16_out(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('TPU v4+ required')
    dtype = jnp.bfloat16
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 256), dtype)
    y = jax.random.normal(k1, (256, 384), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      return z

    impl = fuser.fuse(
        functools.partial(
            matmul,
            fusible_matmul,
        )
    )
    ref = functools.partial(matmul, dot_ref)

    # XLA should be bitwise equivalent.
    self.assertAllClose(
        jax.jit(impl)(x, y),
        jax.jit(ref)(x, y),
        atol=0,
    )

  def test_matmul_bf16_activation(self):
    dtype = jnp.bfloat16
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 256), dtype)
    y = jax.random.normal(k1, (256, 384), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      return jnp.exp(jnp.tanh(z))

    ref = functools.partial(matmul, dot_ref)

    out_ref = jit_no_excess_precision(ref)(x, y)

    impl = fuser.fuse(functools.partial(matmul, fusible_matmul))
    out = jax.jit(impl)(x, y)

    self.assertAllClose(out, out_ref, atol=0)

  def test_matmul_f32_out_simple(self):
    dtype = jnp.bfloat16
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (512, 256), dtype)
    y = jax.random.normal(k1, (256, 384), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      return z

    ref = functools.partial(
        matmul, mm_ref
    )

    out_ref = jit_no_excess_precision(ref)(x, y)

    impl = fuser.fuse(
        functools.partial(
            matmul,
            functools.partial(fusible_matmul, bk=256, bn=128),
        )
    )
    out = jax.jit(impl)(x, y)

    atol = 0
    if jtu.is_device_tpu_at_least(6):
      # 256 MXU changes some tols.
      atol = 1e-5
    self.assertAllClose(out, out_ref, atol=atol)

  def test_matmul_f32_out_fused_downcast(self):
    dtype = jnp.bfloat16
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (2048, 2048), dtype)
    y = jax.random.normal(k1, (2048, 2048), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      return z.astype(x.dtype)

    bm = 512
    bk = 256
    bn = 1024

    ref = functools.partial(
        matmul,
        functools.partial(dot_ref, bm=bm, bk=bk, bn=bn),
    )

    out_ref = jit_no_excess_precision(ref)(x, y)

    impl = fuser.fuse(
        functools.partial(
            matmul,
            functools.partial(
                fusible_matmul,
                bm=bm,
                bk=bk,
                bn=bn,
            ),
        )
    )
    out = jax.jit(impl)(x, y)
    self.assertArraysEqual(out, out_ref)

  def test_matmul_out_bf16_with_f32_activation(self):
    if not jtu.is_device_tpu_at_least(4):
      self.skipTest('TPU v4+ required')
    dtype = jnp.bfloat16
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (2048, 2048), dtype)
    y = jax.random.normal(k1, (2048, 2048), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      return jnp.exp(jnp.tanh(z)).astype(x.dtype)

    bm = 512
    bk = 256
    bn = 1024

    ref = functools.partial(
        matmul,
        functools.partial(dot_ref, bm=bm, bk=bk, bn=bn),
    )

    out_ref = jit_no_excess_precision(ref)(x, y)

    impl = fuser.fuse(
        functools.partial(
            matmul,
            functools.partial(
                fusible_matmul,
                bm=bm,
                bk=bk,
                bn=bn,
            ),
        )
    )
    out = jax.jit(impl)(x, y)
    self.assertArraysEqual(out, out_ref)

  def test_matmul_out_bf16_with_bf16_activation(self):
    dtype = jnp.bfloat16
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(k0, (2048, 2048), dtype)
    y = jax.random.normal(k1, (2048, 2048), dtype)

    def matmul(impl, x, y):
      z = impl(x, y)
      return jnp.exp(jnp.tanh(z)).astype(x.dtype)

    bm = 512
    bk = 256
    bn = 1024

    ref = functools.partial(
        matmul,
        functools.partial(dot_ref, bm=bm, bk=bk, bn=bn),
    )

    out_ref = jit_no_excess_precision(ref)(x, y)

    impl = fuser.fuse(
        functools.partial(
            matmul,
            functools.partial(
                fusible_matmul,
                bm=bm,
                bk=bk,
                bn=bn,
            ),
        )
    )
    out = jax.jit(impl)(x, y)
    self.assertArraysEqual(out, out_ref)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
