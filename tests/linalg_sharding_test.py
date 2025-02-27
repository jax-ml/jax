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

from absl.testing import absltest
import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax.sharding import PartitionSpec as P

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

float_types = jtu.dtypes.floating
complex_types = jtu.dtypes.complex


CPU_ONLY_FUN_AND_SHAPES = [
    # These functions are supported on GPU, but partitioning support will
    # require updates to GSPMD, since they are lowered directly to HLO ops
    # instead of custom calls on GPU.
    (lax.linalg.cholesky, ((6, 6),)),
    (lax.linalg.triangular_solve, ((6, 6), (4, 6))),

    # The GPU kernel for this function still uses an opaque descriptor to
    # encode the input shapes so it is not partitionable.
    # TODO(danfm): Update the kernel and enable this test on GPU.
    (lax.linalg.tridiagonal_solve, ((6,), (6,), (6,), (6, 4))),

    # These functions are only supported on CPU.
    (lax.linalg.hessenberg, ((6, 6),)),
    (lax.linalg.schur, ((6, 6),)),
]

CPU_AND_GPU_FUN_AND_SHAPES = [
    (lax.linalg.eig, ((6, 6),)),
    (lax.linalg.eigh, ((6, 6),)),
    (lax.linalg.lu, ((10, 6),)),
    (lax.linalg.qr, ((6, 6),)),
    (lax.linalg.svd, ((10, 6),)),
    (lax.linalg.tridiagonal, ((6, 6),)),
]

ALL_FUN_AND_SHAPES = CPU_ONLY_FUN_AND_SHAPES + CPU_AND_GPU_FUN_AND_SHAPES


class LinalgShardingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if jtu.test_device_matches(["gpu"]):
      self.skipTest("TODO(danfm): Enable this test on GPU.")
    if jax.device_count() < 2:
      self.skipTest("Requires multiple devices")

  def get_fun_and_shapes(self, fun_and_shapes, grad=False):
    if (jtu.test_device_matches(["gpu"])
        and fun_and_shapes not in CPU_AND_GPU_FUN_AND_SHAPES):
      self.skipTest(f"{fun_and_shapes[0].__name__} not supported on GPU")
    if not grad:
      return fun_and_shapes

    fun, shapes = fun_and_shapes
    if fun in (lax.linalg.schur, lax.linalg.hessenberg, lax.linalg.tridiagonal):
      self.skipTest(f"{fun.__name__} does not support differentation")
    if jtu.test_device_matches(["gpu"]) and fun in (
        lax.linalg.eig, lax.linalg.lu, lax.linalg.qr
    ):
      self.skipTest(
          f"JVP of {fun.__name__} uses triangular solve on GPU, which doesn't "
          "support batch partitioning yet")

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

  @jtu.sample_product(
      fun_and_shapes=ALL_FUN_AND_SHAPES,
      dtype=float_types + complex_types,
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

  @jtu.sample_product(
      fun_and_shapes=ALL_FUN_AND_SHAPES,
      dtype=float_types + complex_types,
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

  @jtu.sample_product(
      fun_and_shapes=ALL_FUN_AND_SHAPES,
      dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("gpu", "cpu")
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
      self.assertAllClose(actual, expected)
      hlo = jvp_fun_jit.lower(primals_sharded, tangents_sharded).compile()
      self.assertNotIn("all-", hlo.as_text())

  @jtu.sample_product(
      fun_and_shapes=ALL_FUN_AND_SHAPES,
      dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("gpu", "cpu")
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
    self.assertAllClose(actual, expected)
    hlo = vjp_fun_jit.lower(tangents_sharded).compile()
    self.assertNotIn("all-", hlo.as_text())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
