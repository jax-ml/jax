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
import unittest

from absl.testing import absltest, parameterized
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
try:
  # We only import this to see if Mosaic is available.
  import jax.experimental.mosaic.gpu  # noqa: F401
except ImportError:
  matmul = None
else:
  from jax.experimental.mosaic.gpu.examples import matmul
try:
  import hypothesis as hp
  import hypothesis.strategies as hps
except (ModuleNotFoundError, ImportError):
  raise unittest.SkipTest("these tests require hypothesis")


config.parse_flags_with_absl()
jtu.setup_hypothesis()
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0")


def seed_hypothesis(f):
  def wrapper(self, seed):
    return hp.seed(seed)(f)(self)
  return wrapper


@jtu.with_config(jax_traceback_filtering="off")
class MatmulTestCase(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if matmul is None:
      self.skipTest("Mosaic GPU not available.")
    if (not jtu.test_device_matches(["cuda"]) or
        not jtu.is_cuda_compute_capability_at_least("9.0")):
      self.skipTest("Only works on GPU with capability >= sm90")

  @parameterized.named_parameters(
      (f"_shard{i}", i) for i in range(5)
  )
  @seed_hypothesis
  @hp.settings(max_examples=100)  # Add verbosity=hp.Verbosity.verbose to debug
  @hp.given(hps.data())
  def test_matmul(self, data):
    m, n, k = (
        data.draw(hps.sampled_from([128, 256, 512, 2048]), label=d)
        for d in "mnk"
    )
    stages = data.draw(hps.integers(2, 5), label="stages")
    tile_m = data.draw(
        hps.sampled_from([t for t in [64, 128, 256] if t <= m]), label="tile_m"
    )
    tile_n = data.draw(
        hps.sampled_from([t for t in [64, 128, 256] if t <= n]), label="tile_n"
    )
    in_dtype = data.draw(
        hps.sampled_from([jnp.float16, jnp.bfloat16, jnp.float32]),
        label="dtype",
    )
    cluster_m = data.draw(hps.sampled_from([1, 2, 4]), label="cluster_m")
    hp.assume((m // tile_m) % cluster_m == 0)
    cluster_n = data.draw(hps.sampled_from([1, 2, 4]), label="cluster_n")
    hp.assume((n // tile_n) % cluster_n == 0)
    # TODO(apaszke): Non-portable clusters (16 blocks) sometimes deadlock.
    hp.assume(cluster_m * cluster_n <= 8)
    if jnp.dtype(in_dtype).itemsize == 4:
      rhs_transpose = True
    else:
      rhs_transpose = data.draw(hps.booleans(), label="rhs_transpose")

    try:
      matmul.verify(
          m,
          k,
          n,
          stages,
          tile_m=tile_m,
          tile_n=tile_n,
          in_dtype=in_dtype,
          cluster_m=cluster_m,
          cluster_n=cluster_n,
          rhs_transpose=rhs_transpose,
      )
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" in str(e):
        hp.assume(False)
      raise e


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
