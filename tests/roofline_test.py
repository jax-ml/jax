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

from absl.testing import absltest
import jax
from jax._src import mesh
from jax._src import test_util as jtu
from jax.experimental import roofline
import jax.lax as lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


jax.config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


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
        # Residuals + cotangents.
        # We gather kn while computing the kn cotangents.
        peak_hbm_bytes=bwd_itemsize * (kn + kn + mn),
    )
    self.assertDataclassEqual(bwd_results, expected)

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
    ]:
      _, result = roofline.roofline(
          f,
          mesh=mesh.AbstractMesh(()),
          in_specs=(P(), P()),
          out_specs=P(),
      )(jnp.zeros((3, 8), dtype=int), jnp.ones((3, 8), dtype=int))
      self.assertEqual(result.unfused_flops, 3 * 3 * 8)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
