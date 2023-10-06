# Copyright 2020 The JAX Authors.
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
"""Tests for multi-platform and cross-platform JAX export."""

import re
from typing import Literal

from absl import logging
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental.export import export
# TODO(necula): Move the primitive harness out of jax2tf so that we can move
# this whole test out of jax2tf.
from jax.experimental.jax2tf.tests import primitive_harness


def make_disjunction_regexp(*parts: str) -> re.Pattern[str]:
  return re.compile("(" + "|".join(parts) + ")")


# TODO(necula): Failures to be investigated (on multiple platforms)
_known_failures = make_disjunction_regexp(
    "cumsum_",
    "cumprod_",
)


# TODO(necula): Failures to be investigated (on GPU).
_known_failures_gpu = make_disjunction_regexp(
    # Failures due to failure to export custom call targets for GPU, these
    # targets do not have backwards compatibility tests.
    "custom_linear_solve_",
    "lu_",
    "svd_",
    "tridiagonal_solve_",
)

# Some primitive lowering rules need the GPU backend to be able to create
# CUDA lowering.
_skip_cuda_lowering_unless_have_gpus = make_disjunction_regexp(
    "svd_", "lu_", "eigh_", "qr_", "custom_linear_", "tridiagonal_solve_",
    "random_",
)


class PrimitiveTest(jtu.JaxTestCase):

  @classmethod
  def setUpClass(cls):
    # Pick one device from each available platform
    cls.devices = []
    cls.platforms = []
    for backend in ["cpu", "gpu", "tpu"]:
      try:
        devices = jax.devices(backend)
      except RuntimeError:
        devices = []

      for d in devices:
        if d.platform not in cls.platforms:
          cls.platforms.append(d.platform)
          cls.devices.append(d)
    super(PrimitiveTest, cls).setUpClass()

  # For each primitive we export for all platforms that are available and
  # compare the results of running the exported code and running the native
  # code.
  # If you want to run this test for only one harness, add parameter
  # `one_containing="foo"` to parameterized below.
  @primitive_harness.parameterized(
      primitive_harness.all_harnesses,
      include_jax_unimpl=False,
      #one_containing="",
  )
  def test_prim(self, harness: primitive_harness.Harness):
    if (
        _known_failures.search(harness.fullname)
        or (
            jtu.device_under_test() == "gpu"
            and _known_failures_gpu.search(harness.fullname)
        )
    ):
      self.skipTest("failure to be investigated")

    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())

    unimplemented_platforms: set[str] = set()
    for l in harness.jax_unimplemented:
      if l.filter(dtype=harness.dtype):
        unimplemented_platforms = unimplemented_platforms.union(l.devices)
    logging.info("Harness is not implemented on %s", unimplemented_platforms)

    devices = [
        d
        for d in self.__class__.devices
        if d.platform not in unimplemented_platforms
    ]
    logging.info(
        "Using devices %s",
        [
            (str(d), d.platform, d.device_kind, d.client.platform)
            for d in devices
        ],
    )
    # lowering_platforms uses "cuda" instead of "gpu"
    lowering_platforms: list[str] = [
        p if p != "gpu" else "cuda"
        for p in {"cpu", "gpu", "tpu"} - unimplemented_platforms
    ]
    if (
        "cuda" in lowering_platforms
        and _skip_cuda_lowering_unless_have_gpus.search(harness.fullname)
        and all(d.platform != "gpu" for d in devices)
    ):
      lowering_platforms.remove("cuda")

    if len(lowering_platforms) <= 1:
      self.skipTest(
          "Harness is uninteresting with fewer than 2 platforms"
      )

    logging.info("Exporting harness for %s", lowering_platforms)
    exp = export.export(func_jax, lowering_platforms=lowering_platforms)(*args)

    for device in devices:
      device_args = jax.tree_util.tree_map(
          lambda x: jax.device_put(x, device), args
      )
      logging.info("Running harness natively on %s", device)
      native_res = func_jax(*device_args)
      logging.info("Running exported harness on %s", device)
      exported_res = export.call_exported(exp)(*device_args)
      self.assertAllClose(native_res, exported_res)
      # TODO(necula): Check HLO equivalence for the ultimate test.


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
