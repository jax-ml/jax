# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test different parameterizations of a matmul."""

import os

from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax.experimental.mosaic.gpu import dialect as mgpu_dialect  # pylint: disable=g-importing-member
import jax.numpy as jnp
import numpy as np

import hypothesis as hp
import hypothesis.strategies as hps

try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  matmul = None
else:
  from jax.experimental.mosaic.gpu.examples import matmul
  from jax.experimental.mosaic.gpu.examples import matmul_blackwell


config.parse_flags_with_absl()
jtu.setup_hypothesis()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


def seed_hypothesis(f):
  def wrapper(self, seed):
    return hp.seed(seed)(f)(self)
  return wrapper


@jtu.with_config(jax_traceback_filtering="off")
@jtu.thread_unsafe_test_class()  # hypothesis is not thread safe
class MatmulTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if matmul is None:
      self.skipTest("Mosaic GPU not available.")
    if not jtu.test_device_matches(["cuda"]):
      self.skipTest("Test needs a GPU device")
    self.context = mlir.make_ir_context()
    mgpu_dialect.register_dialect(self.context)
    self.enter_context(config.traceback_filtering("off"))
    self.enter_context(self.context)
    self.enter_context(ir.Location.unknown())

  @parameterized.named_parameters(
      (f"_shard{i}", i) for i in range(5)
  )
  @seed_hypothesis
  @hp.settings(max_examples=100)  # Add verbosity=hp.Verbosity.verbose to debug
  @hp.given(hps.data())
  def test_matmul_sm90(self, data):
    if not jtu.is_cuda_compute_capability_equal("9.0"):
      self.skipTest("Only works on GPU with capability sm90a")

    in_dtype = data.draw(
        hps.sampled_from([jnp.float16, jnp.bfloat16, jnp.float32]),
        label="in_dtype",
    )
    out_dtype = jnp.float32
    if in_dtype != jnp.float32:
      out_dtype = data.draw(
          hps.sampled_from([in_dtype, jnp.float32]),
          label="out_dtype",
      )
    bytewidth = jnp.dtype(in_dtype).itemsize
    m, n, k = (
        data.draw(hps.sampled_from([128, 256, 512, 2048]), label=d)
        for d in "mnk"
    )
    stages = data.draw(hps.integers(2, 5), label="stages")
    swizzle = data.draw(hps.sampled_from([32, 64, 128]), label="swizzle")
    tile_m = data.draw(
        hps.sampled_from([t for t in [64, 128, 256] if t <= m]), label="tile_m"
    )
    tile_n = data.draw(
        hps.sampled_from([t for t in [64, 128, 256] if t <= n]), label="tile_n"
    )
    grid_m, grid_n = m // tile_m, n // tile_n
    grid_tile_n = data.draw(hps.sampled_from([1, 2, 4, 8, 16]), label="grid_tile_n")
    hp.assume(grid_n % grid_tile_n == 0)
    cluster_m = data.draw(hps.sampled_from([1, 2, 4]), label="cluster_m")
    hp.assume(grid_m % cluster_m == 0)
    cluster_n = data.draw(hps.sampled_from([1, 2, 4]), label="cluster_n")
    hp.assume(grid_n % cluster_n == 0)
    # TODO(apaszke): Non-portable clusters (16 blocks) sometimes deadlock.
    hp.assume(cluster_m * cluster_n <= 8)
    if bytewidth == 4:
      rhs_transpose = True
    else:
      rhs_transpose = data.draw(hps.booleans(), label="rhs_transpose")

    try:
      matmul.verify(
          m,
          k,
          n,
          stages=stages,
          tile_m=tile_m,
          tile_n=tile_n,
          in_dtype=in_dtype,
          out_dtype=out_dtype,
          cluster_m=cluster_m,
          cluster_n=cluster_n,
          grid_tile_n=grid_tile_n,
          swizzle=swizzle,
          rhs_transpose=rhs_transpose,
      )
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" in str(e):
        hp.assume(False)
      raise e

  @parameterized.named_parameters(
      # TODO(apaszke): Increase shard count once we have more B200s in CI.
      (f"_shard{i}", i) for i in range(1)
  )
  @seed_hypothesis
  @hp.settings(max_examples=100)  # Add verbosity=hp.Verbosity.verbose to debug
  @hp.given(hps.data())
  def test_matmul_sm100(self, data):
    if not jtu.is_cuda_compute_capability_equal("10.0"):
      self.skipTest("Only works on GPU with capability sm100a")

    dtype = data.draw(
        hps.sampled_from([jnp.float16, jnp.bfloat16]),
        label="dtype",
    )
    m, n, k = (
        data.draw(hps.sampled_from([128, 256, 512, 2048, 8192]), label=d) for d in "mnk"
    )
    max_concurrent_steps = data.draw(
        hps.integers(2, 5), label="max_concurrent_steps"
    )
    collective = data.draw(hps.booleans(), label="collective")
    num_ctas = 2 if collective else 1
    hp.assume(not (m == 128 and collective))  # Too small for collective MMA.
    tile_m = data.draw(
        hps.sampled_from([t for t in [128] if t * num_ctas <= m]), label="tile_m"
    )
    tmem_cols = 512
    tile_n = data.draw(
        hps.sampled_from([
            t
            for t in [64, 128, 256]
            # We're double buffering TMEM in the kernel, hence the 2x.
            if t * num_ctas <= n and 2 * t * num_ctas <= tmem_cols
        ]),
        label="tile_n",
    )
    grid_m = m // (num_ctas * tile_m)
    grid_tile_m = data.draw(hps.sampled_from([1, 2, 4, 8, 16]), label="grid_tile_m")
    hp.assume(grid_m % grid_tile_m == 0)

    try:
      kernel = matmul_blackwell.build_kernel(
          m,
          k,
          n,
          dtype=dtype,
          tile_m=tile_m,
          tile_n=tile_n,
          grid_tile_m=grid_tile_m,
          max_concurrent_steps=max_concurrent_steps,
          collective=collective,
      )
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" in str(e):
        hp.assume(False)
      raise

    ka, kb = jax.random.split(jax.random.key(0), 2)
    a = jax.random.normal(key=ka, shape=(m, k), dtype=dtype)
    b = jax.random.normal(key=kb, shape=(n, k), dtype=dtype)
    out = kernel(a, b)
    out_ref = jnp.dot(a, b.T)
    np.testing.assert_allclose(
        out, out_ref, atol=1e-3, rtol=1e-3 if k < 512 else 1e-2
    )


if __name__ == "__main__":
  absltest.main(argv=["python"], testLoader=jtu.JaxTestLoader())
