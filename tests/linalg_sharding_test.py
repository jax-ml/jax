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
import numpy as np

import jax
from jax import lax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lib import xla_extension_version
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
    (lax.linalg.qr, ((10, 6),)),
    (lax.linalg.svd, ((10, 6),)),
    (lax.linalg.tridiagonal, ((6, 6),)),
]

ALL_FUN_AND_SHAPES = CPU_ONLY_FUN_AND_SHAPES + CPU_AND_GPU_FUN_AND_SHAPES


class LinalgShardingTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if xla_extension_version < 313:
      self.skipTest("Requires XLA extension version >= 313")
    if jax.device_count() < 2:
      self.skipTest("Requires multiple devices")

  def check_fun_and_shapes(self, fun_and_shapes):
    if (jtu.test_device_matches(["gpu"])
        and fun_and_shapes not in CPU_AND_GPU_FUN_AND_SHAPES):
      self.skipTest(f"{fun_and_shapes[0].__name__} not supported on GPU")

  @jtu.sample_product(
      fun_and_shapes=ALL_FUN_AND_SHAPES,
      dtype=float_types + complex_types,
  )
  @jtu.run_on_devices("gpu", "cpu")
  def test_batch_axis_sharding(self, fun_and_shapes, dtype):
    self.check_fun_and_shapes(fun_and_shapes)
    fun, shapes = fun_and_shapes
    batch_size = 8
    rng = jtu.rand_default(self.rng())
    def arg_maker(shape):
      x = rng((batch_size, *shape), dtype)
      if len(shape) == 2 and shape[0] == shape[1]:
        x = np.matmul(x, np.swapaxes(np.conj(x), -1, -2))
      return x
    args = tuple(arg_maker(shape) for shape in shapes)
    expected = fun(*args)

    mesh = jtu.create_mesh((2,), ("i",))
    sharding = jax.NamedSharding(mesh, P("i"))
    args_sharded = jax.device_put(args, sharding)

    fun_jit = jax.jit(fun)
    actual = fun_jit(*args_sharded)
    self.assertAllClose(actual, expected)
    self.assertNotIn("all-", fun_jit.lower(*args_sharded).compile().as_text())

    vmap_fun = jax.vmap(fun)
    vmap_fun_jit = jax.jit(vmap_fun)
    actual = vmap_fun_jit(*args_sharded)
    self.assertAllClose(actual, expected)
    self.assertNotIn(
        "all-", vmap_fun_jit.lower(*args_sharded).compile().as_text())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
