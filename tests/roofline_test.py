# Copyright 2024 The JAX Authors.
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
from __future__ import annotations

from functools import partial
from typing import Sequence

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental import roofline
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

_VERY_LARGE_NUMBER = 512 * 1024


def create_inputs(
  *shardings: P,
  dtype: jnp.dtype = jnp.float32,
  mesh_shape: tuple[int, ...] = (2, 2, 2),
) -> tuple[jax.sharding.Mesh, tuple[jax.ShapeDtypeStruct, ...]]:
  mesh = jtu.create_mesh(mesh_shape, ("x", "y", "z"))
  arrays = []
  for sharding in shardings:
    array = jax.ShapeDtypeStruct(
      (8, 8), dtype, sharding=jax.sharding.NamedSharding(mesh, sharding)
    )
    arrays.append(array)
  return mesh, tuple(arrays)


class RooflineTest(jtu.JaxTestCase):

  def setUp(self):
    self._bytes_per_word = 8 if jax.config.read("jax_enable_x64") else 4

  def test_scalar_collectives(self):
    a_spec = P("z", ("x", "y"))
    b_spec = P(("x", "y"), "z")
    mesh, (a, b) = create_inputs(a_spec, b_spec)

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=(a_spec, b_spec),
      out_specs=(P("z", None), P(("x", "y"), None)),
    )
    def scalar_collectives(a, b):
      a = lax.pmin(a, ("x", "y"))
      b = lax.pmax(b, "z")
      return a, b

    _, results = scalar_collectives(a, b)

    itemsize = 4

    axis_size = 2
    axis_size_m1 = axis_size - 1

    xy_num_axes = 2
    xy_ici_bytes = int(
      itemsize
      # 2 phases.
      * (
        (1 / xy_num_axes * axis_size_m1) + (1 * axis_size / xy_num_axes * axis_size_m1)
      )
    )
    # 2 phases times 2 hops.
    xy_ici_latency = 2 * 2

    z_ici_bytes = int(itemsize * 1 * axis_size_m1)
    # 2 hops.
    z_ici_latency = 2
    expected = roofline.RooflineResult(
      ici_bytes={"x": xy_ici_bytes, "y": xy_ici_bytes, "z": z_ici_bytes},
      ici_latency={"x": xy_ici_latency, "y": xy_ici_latency, "z": z_ici_latency},
      peak_hbm_bytes=itemsize * 2 * 4 * 2,
    )
    self.assertDataclassEqual(results, expected)

  def test_collective_matmul(self):
    a_spec = P(None, "x")
    b_spec = P(None, "x")
    c_spec = P("x", None)
    mesh, (a, b, c) = create_inputs(a_spec, b_spec, c_spec, dtype=jnp.int8)

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=(a_spec, b_spec, c_spec),
      out_specs=a_spec,
    )
    def collective_matmul(a, b, c):
      a = lax.all_gather(a, "x", axis=1, tiled=True)
      # Test broadcasting and slicing works.
      a = a[None, :, :]
      b = b[:, None, :]
      ab = jnp.einsum("bij,jbk->ikb", a, b).astype(jnp.int8)[..., 0]
      abc = jnp.einsum("ik,kc->ic", ab, c).astype(jnp.int8)
      abc = lax.psum_scatter(abc, "x", scatter_dimension=1, tiled=True)
      return abc

    _, results = collective_matmul(a, b, c)

    itemsize = 1
    m, k, n = 8, 4, 8
    mk = m * k
    kn = k * n
    mn = m * n

    axis_size = 2
    axis_size_m1 = axis_size - 1
    sharded_mk = mk

    # Times 2 for ag + rs.
    ici_bytes = 2 * int(itemsize * sharded_mk * axis_size_m1)
    ici_latency = 2 * 2
    expected = roofline.RooflineResult(
        flops=2 * 2 * m * k * n,
        unfused_flops=2 * 2 * m * k * n,
        ici_bytes={"x": ici_bytes},
        ici_latency={"x": ici_latency},
        hbm_bytes=2 * itemsize * (mk + kn + mn),
        unfused_hbm_bytes=2 * itemsize * (mk + kn + mn),
        # Right after all_gather.
        peak_hbm_bytes=itemsize * (mk * axis_size + mk + kn),
    )
    self.assertDataclassEqual(results, expected)

  def test_matmul_psum(self):
    a_spec = P("z", ("x", "y"))
    b_spec = P(("x", "y"), None)
    mesh, (a, b) = create_inputs(a_spec, b_spec)

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=(a_spec, b_spec),
      out_specs=P("z", None),
    )
    def matmul_psum(a, b):
      c = a @ b
      c = lax.psum(c, ("x", "y"))
      return c

    _, results = matmul_psum(a, b)

    itemsize = 4
    m, k, n = 4, 2, 8
    mk = m * k
    kn = k * n
    mn = m * n

    axis_size = 2
    axis_size_m1 = axis_size - 1
    num_axes = 2
    sharded_mn = mn / axis_size / num_axes

    # Times 2 for ag + rs.
    ici_bytes = 2 * int(
      itemsize
      # 2 phases.
      * (
        (sharded_mn / num_axes * axis_size_m1)
        + (sharded_mn * axis_size / num_axes * axis_size_m1)
      )
    )
    ici_latency = 2 * 2 * 2
    expected = roofline.RooflineResult(
        flops=2 * m * k * n,
        unfused_flops=2 * m * k * n,
        ici_bytes={axis: ici_bytes for axis in ("x", "y")},
        ici_latency={axis: ici_latency for axis in ("x", "y")},
        hbm_bytes=itemsize * (mk + kn + mn),
        unfused_hbm_bytes=itemsize * (mk + kn + mn),
        peak_hbm_bytes=itemsize * (mn),
    )
    self.assertDataclassEqual(results, expected)

  def test_all_to_all(self):
    a_spec = P("z", ("x", "y"))
    b_spec = P(("x", "y"), "z")
    mesh, (a, b) = create_inputs(a_spec, b_spec)

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=(a_spec, b_spec),
      out_specs=(P(("z", "x", "y"), None), P(("x", "y", "z"), None)),
    )
    def all_to_all(a, b):
      a = lax.all_to_all(a, ("x", "y"), split_axis=0, concat_axis=1, tiled=True)
      b = lax.all_to_all(b, "z", split_axis=0, concat_axis=1, tiled=True)
      return a, b

    _, results = all_to_all(a, b)

    itemsize = 4

    xy_size = itemsize * 8 * 8 / 2
    # Half the data over 2 links.
    xy_ici_bytes = int(xy_size / 2 / 2)
    # 2 hops.
    xy_ici_latency = 2

    z_size = itemsize * 8 * 8 / 2 / 2
    # Half the data over 1 link.
    z_ici_bytes = int(z_size / 2)
    # 1 hop.
    z_ici_latency = 1
    expected = roofline.RooflineResult(
      ici_bytes={"x": xy_ici_bytes, "y": xy_ici_bytes, "z": z_ici_bytes},
      ici_latency={"x": xy_ici_latency, "y": xy_ici_latency, "z": z_ici_latency},
      peak_hbm_bytes=itemsize * 2 * 4 * 2,
    )
    self.assertDataclassEqual(results, expected)

  def test_ppermute(self):
    a_spec = P("z", ("x", "y"))
    b_spec = P(("x", "y"), "z")
    mesh, (a, b) = create_inputs(a_spec, b_spec)

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=(a_spec, b_spec),
      out_specs=(a_spec, b_spec),
    )
    def ppermute(a, b):
      a = lax.ppermute(a, ("x", "y"), perm=((0, 3), (3, 0), (1, 2), (2, 1)))
      b = lax.ppermute(b, "z", perm=((1, 0), (0, 1)))
      return a, b

    _, results = ppermute(a, b)

    itemsize = 4
    shard_size = itemsize * 4 * 2

    # At most 2 shards contend for 1 link.
    xy_ici_bytes = int(shard_size * 2)
    # 2 hops.
    xy_ici_latency = 2

    # No contention but there is a single link.
    z_ici_bytes = int(shard_size * 2)
    # 1 hop.
    z_ici_latency = 1
    expected = roofline.RooflineResult(
      ici_bytes={"x": xy_ici_bytes, "y": xy_ici_bytes, "z": z_ici_bytes},
      ici_latency={"x": xy_ici_latency, "y": xy_ici_latency, "z": z_ici_latency},
      peak_hbm_bytes=itemsize * 2 * 4 * 2,
    )
    self.assertDataclassEqual(results, expected)

  def test_grad_matmuls(self):
    a_spec = P(None, "x")
    b_spec = P(None, None)
    mesh, (a, b) = create_inputs(a_spec, b_spec, dtype=jnp.int8)

    @partial(
      roofline.roofline_and_grad,
      mesh=mesh,
      in_specs=(a_spec, b_spec),
      # Numerically incorrect AD, but tests that we handle it properly.
      out_specs=P("x", None),
    )
    def collective_matmul(a, b):
      a = lax.all_gather(a, "x", axis=1, tiled=True)
      return a @ b

    c, fwd_results, bwd_results = collective_matmul(a, b)

    itemsize = 1
    m, k, n = 8, 8, 8
    mk = m * k
    kn = k * n
    mn = m * n

    axis_size = 2
    axis_size_m1 = axis_size - 1
    sharded_mk = mk // axis_size

    ici_bytes = int(itemsize * sharded_mk * axis_size_m1)
    ici_latency = 2
    expected = roofline.RooflineResult(
        flops=2 * m * k * n,
        unfused_flops=2 * m * k * n,
        ici_bytes={"x": ici_bytes},
        ici_latency={"x": ici_latency},
        hbm_bytes=itemsize * (mk + kn + mn),
        unfused_hbm_bytes=itemsize * (mk + kn + mn),
        peak_hbm_bytes=itemsize * (mk + kn),
    )
    self.assertDataclassEqual(fwd_results, expected)

    bwd_itemsize = 2
    # 2 for psum + 1 for rs.
    bwd_ici_bytes = 3 * int(bwd_itemsize * sharded_mk * axis_size_m1)
    expected = roofline.RooflineResult(
        flops=2 * 2 * m * k * n,
        unfused_flops=2 * 2 * m * k * n,
        ici_bytes={"x": bwd_ici_bytes},
        ici_latency={"x": 3 * ici_latency},
        hbm_bytes=2 * bwd_itemsize * (mk + kn + mn),
        unfused_hbm_bytes=2 * bwd_itemsize * (mk + kn + mn),
        # Residuals + cotangents.
        peak_hbm_bytes=bwd_itemsize * (mk + kn + mn),
    )
    self.assertDataclassEqual(bwd_results, expected)

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=c.sharding.spec,
      out_specs=c.sharding.spec,
    )
    def mul_2(c):
      return c * 2

    results = mul_2(c)
    self.assertLen(results, 2)

  def test_one_sized_axis_collectives(self):
    a_spec = P("x")
    mesh, (a,) = create_inputs(a_spec, mesh_shape=(1, 2, 4))

    @partial(
      roofline.roofline,
      mesh=mesh,
      in_specs=a_spec,
      out_specs=a_spec,
    )
    def one_sized_axis_collectives(a):
      a = lax.pmin(a, "x")
      a = lax.all_gather(a, "x", axis=1, tiled=True)
      a = lax.psum_scatter(a, "x", scatter_dimension=1, tiled=True)
      a = lax.psum(a, "x")
      a = lax.all_to_all(a, "x", split_axis=0, concat_axis=1, tiled=True)
      a = lax.ppermute(a, "x", perm=((1, 0), (0, 1)))
      return a

    _, results = one_sized_axis_collectives(a)
    expected = roofline.RooflineResult(
      ici_bytes={"x": 0},
      ici_latency={"x": 0},
      peak_hbm_bytes=4 * 8 * 8,
    )
    self.assertDataclassEqual(results, expected)

  def test_remat(self):
    a_spec = P("x", None)
    b_spec = P("x", None)
    mesh, (a, b) = create_inputs(a_spec, b_spec)

    def fsdp_checkpoint_policy(prim, *args, **kwargs):
      if prim == lax.all_gather_p and kwargs["axis_name"] == "x":
        return True
      return False

    @partial(
      roofline.roofline_and_grad,
      mesh=mesh,
      in_specs=(a_spec, b_spec),
      out_specs=P("x", None),
    )
    @partial(jax.checkpoint, policy=fsdp_checkpoint_policy)
    def collective_matmul(a, b):
      b = lax.all_gather(b, "x", axis=0, tiled=True)
      return a @ b

    _, fwd_results, bwd_results = collective_matmul(a, b)

    itemsize = 4
    m, k, n = 4, 8, 8
    mk = m * k
    kn = k * n
    mn = m * n

    axis_size = 2
    axis_size_m1 = axis_size - 1
    sharded_kn = kn // axis_size

    ici_bytes = int(itemsize * sharded_kn * axis_size_m1)
    ici_latency = 2
    expected = roofline.RooflineResult(
        flops=2 * m * k * n,
        unfused_flops=2 * m * k * n,
        ici_bytes={"x": ici_bytes},
        ici_latency={"x": ici_latency},
        hbm_bytes=itemsize * (mk + kn + mn),
        unfused_hbm_bytes=itemsize * (mk + kn + mn),
        peak_hbm_bytes=itemsize * (mk + kn),
    )
    self.assertDataclassEqual(fwd_results, expected)

    bwd_itemsize = 2
    # Remat ag + rs.
    bwd_ici_bytes = 2 * int(bwd_itemsize * sharded_kn * axis_size_m1)
    expected = roofline.RooflineResult(
        flops=2 * 2 * m * k * n,
        unfused_flops=2 * 2 * m * k * n,
        ici_bytes={"x": bwd_ici_bytes},
        ici_latency={"x": 2 * ici_latency},
        hbm_bytes=2 * bwd_itemsize * (mk + kn + mn),
        unfused_hbm_bytes=2 * bwd_itemsize * (mk + kn + mn),
        # Residuals + cotangents.
        # We gather kn while computing the kn cotangents.
        peak_hbm_bytes=bwd_itemsize * (kn + kn + mn),
    )
    self.assertDataclassEqual(bwd_results, expected)

  @jtu.parameterized.named_parameters(
      ("abs", lax.abs, float),
      ("acos", lax.acos, float),
      ("asin", lax.asin, float),
      ("atan", lax.atan, float),
      ("cbrt", lax.cbrt, float),
      ("ceil", lax.ceil, float),
      ("conj", lax.conj, complex),
      ("cos", lax.cos, float),
      ("cosh", lax.cosh, float),
      ("exp", lax.exp, float),
      ("expm1", lax.expm1, float),
      ("floor", lax.floor, float),
      ("imag", lax.imag, complex),
      ("integer_pow", lambda a: lax.integer_pow(a, 5), int),
      ("is_finite", lax.is_finite, float),
      ("log", lax.log, float),
      ("log1p", lax.log1p, float),
      ("logistic", lax.logistic, float),
      ("neg", lax.neg, float),
      ("not", lax.bitwise_not, bool),
      ("real", lax.real, complex),
      ("round", lax.round, float),
      ("rsqrt", lax.rsqrt, float),
      ("sign", lax.sign, float),
      ("sin", lax.sin, float),
      ("sinh", lax.sinh, float),
      ("sqrt", lax.sqrt, float),
      ("square", lax.square, float),
      ("tan", lax.tan, float),
      ("bessel_i0e", lax.bessel_i0e, float),
      ("bessel_i1e", lax.bessel_i1e, float),
      ("digamma", lax.digamma, float),
      ("erf_inv", lax.erf_inv, float),
      ("erf", lax.erf, float),
      ("erfc", lax.erfc, float),
      ("lgamma", lax.lgamma, float),
  )
  def test_unary_ops(self, f, dtype):
    data = jnp.zeros((3, 8), dtype=dtype)
    out, result = roofline.roofline(f)(data)
    with self.subTest("flops"):
      self.assertEqual(result.unfused_flops, 3 * 8)
    with self.subTest("hbm_bytes"):
      self.assertEqual(
          result.unfused_hbm_bytes,
          data.dtype.itemsize * 3 * 8 + out.dtype.itemsize * 3 * 8,
      )

  def test_binary_ops(self):
    for f in [
        lambda a, b: a ^ b,
        lambda a, b: a | b,
        lambda a, b: a & b,
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a < b,
        lambda a, b: a <= b,
        lambda a, b: a > b,
        lambda a, b: a >= b,
        lambda a, b: a == b,
        lambda a, b: jnp.minimum(a, b),
        lambda a, b: jnp.maximum(a, b),
    ]:
      out, result = roofline.roofline(f)(
          jnp.zeros((3, 8), dtype=int), jnp.ones((3, 8), dtype=int)
      )
      self.assertEqual(result.unfused_flops, 3 * 8)
      self.assertEqual(
          result.unfused_hbm_bytes,
          2 * self._bytes_per_word * 3 * 8 + out.dtype.itemsize * 3 * 8,
      )

  def test_broadcast(self):
    for left, right in [
        (jnp.zeros((3, 8)), jnp.ones((1, 1))),
        (jnp.zeros((1, 1)), jnp.ones((3, 8))),
        (jnp.zeros((3, 8)), jnp.ones((3, 8))),
        (2.0, jnp.ones((3, 8))),
        (jnp.zeros((3, 8)), 2.0),
    ]:
      _, result = roofline.roofline(lambda a, b: a + b)(left, right)
      self.assertEqual(result.unfused_flops, 3 * 8)

  def test_nested(self):
    def f(x, y):
      @jax.jit
      def g(x):
        return x * y

      return g(x) + g(y)

    _, result = roofline.roofline(f)(
        jnp.zeros((11, 4), dtype=int), jnp.ones((11, 4), dtype=int)
    )
    self.assertEqual(result.unfused_flops, 3 * (11 * 4))

  def test_no_mesh(self):
    _, result = roofline.roofline(lambda a, b: a + b)(
        jnp.zeros((3, 8), dtype=int), jnp.ones((3, 8), dtype=int)
    )
    self.assertEqual(result.unfused_flops, 3 * 8)

  def test_no_specs(self):
    _, result = roofline.roofline(lambda a, b: a + b)(
        jnp.zeros((3, 8), dtype=int), jnp.ones((3, 8), dtype=int)
    )
    self.assertEqual(result.unfused_flops, 3 * 8)

  def test_no_mesh_and_no_specs(self):
    _, result = roofline.roofline(
        lambda a, b: a + b,
    )(jnp.zeros((3, 8), dtype=int), jnp.ones((3, 8), dtype=int))
    self.assertEqual(result.unfused_flops, 3 * 8)

  def test_dot_general(self):
    _, result = roofline.roofline(lambda a, b: a @ b)(
        jnp.zeros((3, 7), dtype=int), jnp.ones((7, 5), dtype=int)
    )
    self.assertEqual(result.unfused_flops, 2 * 3 * 7 * 5)
    self.assertEqual(
        result.unfused_hbm_bytes, self._bytes_per_word * (3 * 7 + 7 * 5 + 3 * 5)
    )

  def get_conv_output_dim(self, i, k, pad_low, pad_high, stride) -> int:
    return jnp.floor((i - k + pad_low + pad_high) / stride) + 1

  def get_conv_num_output_channels(
      self, batch_group_count: int, feature_group_count: int
  ) -> int:
    if batch_group_count > 1:
      return batch_group_count
    elif feature_group_count > 1:
      return feature_group_count
    else:
      return 1

  @jtu.parameterized.product(
      window_strides=[(1, 1), (2, 2)],
      padding=[((0, 0), (0, 0)), ((1, 2), (3, 4))],
      # batch must be divisible by batch_group_count, so we only include factors
      # of batch_group_count.
      batch=[6, 12],
      batch_group_count=[1, 3],
      # num_input_channels must be divisible by feature_group_count, so we only
      # include factors of feature_group_count.
      num_input_channels=[6, 12],
      feature_group_count=[1, 3],
  )
  def test_conv_general_dilated_unfused_hbm_bytes(
      self,
      window_strides: Sequence[int, int],
      padding: Sequence[int, int],
      batch: int,
      batch_group_count: int,
      num_input_channels: int,
      feature_group_count: int,
  ):
    if batch_group_count > 1 and feature_group_count > 1:
      self.skipTest(
          "batch_group_count and feature_group_count cannot both be > 1"
      )

    num_output_channels = self.get_conv_num_output_channels(
        batch_group_count, feature_group_count
    )

    num_input_features = int(num_input_channels / feature_group_count)
    iw, ih = 100, 200
    kw, kh = 7, 7
    input_data = jnp.zeros((batch, num_input_channels, iw, ih), dtype=int)
    kernel_data = jnp.ones(
        (num_output_channels, num_input_features, kw, kh), dtype=int
    )
    conv = lambda a, b: lax.conv_general_dilated(
        lhs=a,
        rhs=b,
        window_strides=window_strides,
        padding=padding,
        batch_group_count=batch_group_count,
        feature_group_count=feature_group_count,
    )

    _, result = roofline.roofline(conv)(input_data, kernel_data)

    expected_input_size = batch * num_input_channels * iw * ih
    expected_kernel_size = num_output_channels * num_input_features * kw * kh

    ow = self.get_conv_output_dim(
        iw, kw, padding[0][0], padding[0][1], window_strides[0]
    )
    oh = self.get_conv_output_dim(
        ih, kh, padding[1][0], padding[1][1], window_strides[1]
    )
    expected_output_shape = jnp.array(
        (batch / batch_group_count, num_output_channels, ow, oh)
    )
    expected_output_size = jnp.prod((expected_output_shape))
    # Bytes accessed is sum of inputs and output.
    expected_unfused_hbm_bytes = self._bytes_per_word * (
        expected_input_size + expected_kernel_size + expected_output_size
    )
    self.assertEqual(result.unfused_hbm_bytes, expected_unfused_hbm_bytes)

  @jtu.parameterized.named_parameters(
      dict(
          testcase_name="same",
          padding="SAME",
      ),
      dict(
          testcase_name="same_lower",
          padding="SAME_LOWER",
      ),
  )
  def test_conv_general_dilated_padding_string(
      self, padding: str
  ):
    input_data = jnp.zeros((1, 1, 3, 3), dtype=int)
    kernel_data = jnp.ones((1, 1, 3, 3), dtype=int)
    conv = lambda a, b: lax.conv_general_dilated(
        lhs=a, rhs=b, window_strides=(1, 1), padding=padding
    )

    _, result = roofline.roofline(conv)(input_data, kernel_data)

    # Test hbm bytes.
    expected_input_size = 1 * 1 * 3 * 3
    expected_kernel_size = 1 * 1 * 3 * 3
    # Because of same{_lower} padding, output shape should equal to input shape.
    # This may not be true for other `{feature, batch}`_group_count`s.
    expected_output_size = expected_input_size
    # Bytes accessed is sum of inputs and output.
    expected_unfused_hbm_bytes = self._bytes_per_word * (
        expected_input_size + expected_kernel_size + expected_output_size
    )
    self.assertEqual(result.unfused_hbm_bytes, expected_unfused_hbm_bytes)

    # Test flops.
    # For spatial_valid_position_counts, we have 3x3 output with the following
    # flops for each element:
    #    4 6 4
    #    6 9 6
    #    4 6 4
    # Non_spatial_dims_factor = 1 because `{batch, feature}_group_count` are
    # both equal to 1.
    # Each FMA is 2 flops.
    self.assertEqual(
        result.unfused_flops,
        2 * (4 + 6 + 4 + 6 + 9 + 6 + 4 + 6 + 4),
    )

  def test_conv_general_dilated_padding_string_valid(self):
    input_data = jnp.zeros((1, 1, 10, 20), dtype=int)
    kernel_data = jnp.ones((1, 1, 3, 3), dtype=int)
    conv = lambda a, b: lax.conv_general_dilated(
        lhs=a, rhs=b, window_strides=(1, 1), padding="VALID"
    )

    _, result = roofline.roofline(conv)(input_data, kernel_data)

    expected_input_size = 1 * 1 * 10 * 20
    expected_kernel_size = 1 * 1 * 3 * 3
    # Valid padding is same as 0 padding.
    expected_output_size = (
        1
        * 1
        * self.get_conv_output_dim(10, 3, 0, 0, 1)
        * self.get_conv_output_dim(20, 3, 0, 0, 1)
    )

    # Bytes accessed is sum of inputs and output.
    expected_unfused_hbm_bytes = self._bytes_per_word * (
        expected_input_size + expected_kernel_size + expected_output_size
    )
    self.assertEqual(result.unfused_hbm_bytes, expected_unfused_hbm_bytes)

    # Output shape is [1x1x8x18] and each output element requires (3x3) FMAs,
    # and each FMA is 2 flops.
    self.assertEqual(
        result.unfused_flops, 2 * expected_output_size * 3 * 3
    )


  @jtu.parameterized.named_parameters(
      dict(
          testcase_name="padding",
          input_spatial_dim=1,
          window_strides=[1],
          padding=[(_VERY_LARGE_NUMBER - 1, _VERY_LARGE_NUMBER - 1)],
          lhs_dilation=[1],
      ),
      dict(
          testcase_name="input",
          input_spatial_dim=_VERY_LARGE_NUMBER,
          window_strides=[_VERY_LARGE_NUMBER - 1],
          padding=[(0, 0)],
          lhs_dilation=[_VERY_LARGE_NUMBER],
      ),
  )
  def test_conv_general_dilated_flops_very_large(
      self, input_spatial_dim, window_strides, padding, lhs_dilation
  ):
    input_data = jnp.zeros((1, 1, input_spatial_dim), dtype=int)
    kernel_data = jnp.ones((1, 1, _VERY_LARGE_NUMBER), dtype=int)
    conv = lambda a, b: lax.conv_general_dilated(
        lhs=a,
        rhs=b,
        window_strides=window_strides,
        padding=padding,
        lhs_dilation=lhs_dilation,
    )
    _, result = roofline.roofline(conv)(input_data, kernel_data)

    self.assertEqual(result.unfused_flops, 2 * _VERY_LARGE_NUMBER)

  def test_conv_general_dilated_flops_feature_group_count(self):
    feature_group_count = 120
    input_data = jnp.zeros((1, feature_group_count, 10, 20), dtype=int)
    kernel_data = jnp.ones((feature_group_count, 1, 3, 3), dtype=int)
    conv = lambda a, b: lax.conv_general_dilated(
        lhs=a,
        rhs=b,
        window_strides=(1, 1),
        padding=((0, 0), (0, 0)),
        feature_group_count=feature_group_count,
    )
    _, result = roofline.roofline(conv)(input_data, kernel_data)

    # Output shape is [1x120x8x18] and each output element requires (3x3)
    # FMAs and one FMA is 2 flops.
    self.assertEqual(
        result.unfused_flops, 2 * 120 * 8 * 18 * 3 * 3
    )

  def test_conv_general_dilated_flops_batch_group_count(self):
    batch_group_count = 120
    input_data = jnp.zeros((batch_group_count, 1, 10, 20), dtype=int)
    kernel_data = jnp.ones((batch_group_count, 1, 3, 3), dtype=int)
    conv = lambda a, b: lax.conv_general_dilated(
        lhs=a,
        rhs=b,
        window_strides=(1, 1),
        padding=((0, 0), (0, 0)),
        batch_group_count=batch_group_count,
    )
    _, result = roofline.roofline(conv)(input_data, kernel_data)

    # Output shape is [120x1x8x18] and each output element requires (3x3)
    # FMAs and one FMA is 2 flops.
    self.assertEqual(
        result.unfused_flops, 2 * 120 * 8 * 18 * 3 * 3
    )

  def test_reduce_sum_no_axis(self):
    _, result = roofline.roofline(lambda x: jnp.sum(x))(jnp.zeros((11, 4)))
    self.assertEqual(result.unfused_flops, 11 * 4 - 1)
    self.assertEqual(
        result.unfused_hbm_bytes, self._bytes_per_word * (11 * 4 + 1)
    )

  def test_reduce_sum_with_axis(self):
    for axis, expected_flops, expected_memory in [
        (0, (11 - 1) * 4, 11 * 4 + 4),
        (1, (4 - 1) * 11, 11 * 4 + 11),
        ([0, 1], 11 * 4 - 1, 11 * 4 + 1),
        ([], 0, 11 * 4 + 11 * 4),
    ]:
      _, result = roofline.roofline(lambda x: jnp.sum(x, axis=axis))(
          jnp.zeros((11, 4))
      )
      self.assertEqual(result.unfused_flops, expected_flops)
      self.assertEqual(
          result.unfused_hbm_bytes, self._bytes_per_word * expected_memory
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
