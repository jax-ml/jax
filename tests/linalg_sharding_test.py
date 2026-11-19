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

import functools

from absl.testing import absltest, parameterized
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec as P

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex


# These functions are only supported on CPU.
CPU_ONLY_FUN_AND_SHAPES = [
    (lax.linalg.hessenberg, ((6, 6),)),
    (lax.linalg.schur, ((6, 6),)),
]

CPU_AND_GPU_FUN_AND_SHAPES = [
    (lax.linalg.cholesky, ((6, 6),)),
    (lax.linalg.eig, ((6, 6),)),
    (lax.linalg.eigh, ((6, 6),)),
    (lax.linalg.lu, ((10, 6),)),
    (lax.linalg.qr, ((6, 6),)),
    (lax.linalg.svd, ((10, 6),)),
    (lax.linalg.triangular_solve, ((6, 6), (4, 6))),
    (lax.linalg.tridiagonal, ((6, 6),)),
    (lax.linalg.tridiagonal_solve, ((6,), (6,), (6,), (6, 4))),
]

ALL_FUN_AND_SHAPES = CPU_ONLY_FUN_AND_SHAPES + CPU_AND_GPU_FUN_AND_SHAPES


class LinalgShardingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() < 2:
      self.skipTest("Requires multiple devices")

  def get_fun_and_shapes(self, fun_and_shapes, grad=False):
    if jtu.test_device_matches(["gpu"]):
      if fun_and_shapes not in CPU_AND_GPU_FUN_AND_SHAPES:
        self.skipTest(
            f"Partitioning {fun_and_shapes[0].__name__} not supported on GPU.")
      if fun_and_shapes[0] == lax.linalg.tridiagonal_solve:
        self.skipTest(
            f"Partitioning {fun_and_shapes[0].__name__} on GPU, requires a "
            "more recent jaxlib version.")
    if not grad:
      return fun_and_shapes

    fun, shapes = fun_and_shapes
    if fun in (lax.linalg.schur, lax.linalg.hessenberg, lax.linalg.tridiagonal):
      self.skipTest(f"{fun.__name__} does not support differentation")

    if fun == lax.linalg.eig:
      fun = functools.partial(
          fun,
          compute_left_eigenvectors=False,
          compute_right_eigenvectors=False,
      )
    if fun == lax.linalg.svd:
      fun = functools.partial(fun, full_matrices=False)

    return fun, shapes

  def get_args(self, shapes, dtype, batch_size=None):
    rng = jtu.rand_default(self.rng())
    def arg_maker(shape):
      if batch_size is not None:
        x = rng((batch_size, *shape), dtype)
      else:
        x = rng(shape, dtype)
      if len(shape) == 2 and shape[0] == shape[1]:
        x = np.matmul(x, np.swapaxes(np.conj(x), -1, -2))
      return x
    return tuple(arg_maker(shape) for shape in shapes)

  @parameterized.product(
      fun_and_shapes=ALL_FUN_AND_SHAPES, dtype=float_types + complex_types
  )
  @jtu.run_on_devices("gpu", "cpu")
  def test_batch_axis_sharding(self, fun_and_shapes, dtype):
    fun, shapes = self.get_fun_and_shapes(fun_and_shapes)
    args = self.get_args(shapes, dtype, batch_size=8)

    mesh = jtu.create_mesh((2,), ("i",))
    sharding = jax.NamedSharding(mesh, P("i"))
    args_sharded = jax.device_put(args, sharding)

    fun_jit = jax.jit(fun)
    expected = fun(*args)
    actual = fun_jit(*args_sharded)
    self.assertAllClose(actual, expected)
    self.assertNotIn("all-", fun_jit.lower(*args_sharded).compile().as_text())

    vmap_fun = jax.vmap(fun)
    vmap_fun_jit = jax.jit(vmap_fun)
    actual = vmap_fun_jit(*args_sharded)
    self.assertAllClose(actual, expected)
    self.assertNotIn(
        "all-", vmap_fun_jit.lower(*args_sharded).compile().as_text())

  @parameterized.product(
      fun_and_shapes=ALL_FUN_AND_SHAPES, dtype=float_types + complex_types
  )
  @jtu.run_on_devices("gpu", "cpu")
  def test_non_batch_axis_sharding(self, fun_and_shapes, dtype):
    fun, shapes = self.get_fun_and_shapes(fun_and_shapes)
    args = self.get_args(shapes, dtype)

    mesh = jtu.create_mesh((2,), ("i",))
    sharding = jax.NamedSharding(mesh, P("i"))
    args_sharded = jax.device_put(args, sharding)

    fun_jit = jax.jit(fun)
    expected = fun(*args)
    actual = fun_jit(*args_sharded)
    self.assertAllClose(actual, expected)
    self.assertIn(
        "all-gather", fun_jit.lower(*args_sharded).compile().as_text())

  @parameterized.product(
      fun_and_shapes=ALL_FUN_AND_SHAPES, dtype=float_types + complex_types
  )
  @jtu.run_on_devices("gpu", "cpu")
  @jax.default_matmul_precision("highest")
  def test_batch_axis_sharding_jvp(self, fun_and_shapes, dtype):
    fun, shapes = self.get_fun_and_shapes(fun_and_shapes, grad=True)
    primals = self.get_args(shapes, dtype, batch_size=8)
    tangents = tuple(map(jnp.ones_like, primals))

    def jvp_fun(primals, tangents):
      return jax.jvp(fun, primals, tangents)

    mesh = jtu.create_mesh((2,), ("i",))
    sharding = jax.NamedSharding(mesh, P("i"))
    primals_sharded = jax.device_put(primals, sharding)
    tangents_sharded = jax.device_put(tangents, sharding)

    jvp_fun_jit = jax.jit(jvp_fun)
    _, expected = jvp_fun(primals, tangents)
    for args in [
        (primals_sharded, tangents_sharded),
        (primals, tangents_sharded),
        (primals_sharded, tangents),
    ]:
      _, actual = jvp_fun_jit(*args)
      self.assertAllClose(
          actual,
          expected,
          atol={np.complex64: 2e-4},
          rtol={
              np.float32: 1e-4,
              np.float64: 2e-11,
              np.complex64: 1e-4,
              np.complex128: 1e-11,
          },
      )
      hlo = jvp_fun_jit.lower(primals_sharded, tangents_sharded).compile()
      self.assertNotIn("all-", hlo.as_text())

  @parameterized.product(
      fun_and_shapes=ALL_FUN_AND_SHAPES, dtype=float_types + complex_types
  )
  @jtu.run_on_devices("gpu", "cpu")
  @jax.default_matmul_precision("highest")
  def test_batch_axis_sharding_vjp(self, fun_and_shapes, dtype):
    fun, shapes = self.get_fun_and_shapes(fun_and_shapes, grad=True)
    primals = self.get_args(shapes, dtype, batch_size=8)
    out, vjp_fun = jax.vjp(fun, *primals)
    tangents = jax.tree.map(jnp.ones_like, out)

    mesh = jtu.create_mesh((2,), ("i",))
    sharding = jax.NamedSharding(mesh, P("i"))
    tangents_sharded = jax.device_put(tangents, sharding)

    vjp_fun_jit = jax.jit(vjp_fun)
    expected = vjp_fun(tangents)
    actual = vjp_fun_jit(tangents_sharded)
    self.assertAllClose(
        actual,
        expected,
        atol={np.complex64: 1e-4},
        rtol={
            np.float32: 1e-4,
            np.float64: 1e-11,
            np.complex64: 1e-4,
            np.complex128: 1e-11,
        },
    )
    hlo = vjp_fun_jit.lower(tangents_sharded).compile()
    self.assertNotIn("all-", hlo.as_text())

  @jtu.with_explicit_mesh((2, 2), ("x", "y"))
  def test_explicit_batch_axis_sharding_qr_cholesky(self, mesh):
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((4, 4, 5, 3), dtype=np.float32)
    t_np = rng.standard_normal((4, 4, 5, 3), dtype=np.float32)
    mat_sharding = jax.NamedSharding(mesh, P("x", "y", None, None))
    vec_sharding = jax.NamedSharding(mesh, P("x", "y", None))
    a = jax.device_put(a_np, mat_sharding)
    t = jax.device_put(t_np, mat_sharding)

    qr_jit = jax.jit(functools.partial(lax.linalg.qr, full_matrices=False))
    q, r = qr_jit(a)
    self.assertEqual(q.sharding, mat_sharding)
    self.assertEqual(r.sharding, mat_sharding)
    self.assertAllClose(q @ r, a_np, atol=1e-4, rtol=1e-4)
    self.assertNotIn("all-", qr_jit.lower(a).compile().as_text())

    qr_piv_jit = jax.jit(
        functools.partial(lax.linalg.qr, full_matrices=False, pivoting=True)
    )
    qp, rp, p = qr_piv_jit(a)
    self.assertEqual(qp.sharding, mat_sharding)
    self.assertEqual(rp.sharding, mat_sharding)
    self.assertEqual(p.sharding, vec_sharding)
    self.assertAllClose(
        qp @ rp,
        np.take_along_axis(a_np, np.asarray(p)[..., None, :], axis=-1),
        atol=1e-4,
        rtol=1e-4,
    )
    self.assertNotIn("all-", qr_piv_jit.lower(a).compile().as_text())

    a_empty = jax.device_put(a_np[:, :, :0, :], mat_sharding)
    q0, r0 = qr_jit(a_empty)
    self.assertEqual(q0.sharding, mat_sharding)
    self.assertEqual(r0.sharding, mat_sharding)
    q0p, r0p, p0 = qr_piv_jit(a_empty)
    self.assertEqual(q0p.sharding, mat_sharding)
    self.assertEqual(r0p.sharding, mat_sharding)
    self.assertEqual(p0.sharding, vec_sharding)

    a_short = jax.device_put(a_np[0], jax.NamedSharding(mesh, P("x")))
    _, _, p_short = qr_piv_jit(a_short)
    self.assertEqual(p_short.sharding, jax.NamedSharding(mesh, P("x", None)))

    (_, _), (dq, dr) = jax.jit(
        lambda x, dx: jax.jvp(
            functools.partial(lax.linalg.qr, full_matrices=False), (x,), (dx,)
        )
    )(a, t)
    self.assertEqual(dq.sharding, mat_sharding)
    self.assertEqual(dr.sharding, mat_sharding)

    b_np = rng.standard_normal((4, 4, 4, 4), dtype=np.float32)
    with self.assertRaisesRegex(
        core.ShardingTypeError, "unsharded on non-batch dimensions"):
      qr_piv_jit(jax.device_put(
          b_np, jax.NamedSharding(mesh, P("x", None, "y", None))))
    with self.assertRaisesRegex(
        core.ShardingTypeError, "unsharded on non-batch dimensions"):
      qr_piv_jit(jax.device_put(
          b_np, jax.NamedSharding(mesh, P("x", None, None, "y"))))

    spd_np = (
        b_np @ np.swapaxes(b_np, -1, -2) + 4.0 * np.eye(4, dtype=np.float32))
    spd = jax.device_put(spd_np, mat_sharding)
    chol_jit = jax.jit(lax.linalg.cholesky)
    l = chol_jit(spd)
    self.assertEqual(l.sharding, mat_sharding)
    self.assertAllClose(
        l @ np.swapaxes(l, -1, -2), spd_np, atol=1e-4, rtol=1e-4)
    self.assertNotIn("all-", chol_jit.lower(spd).compile().as_text())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
