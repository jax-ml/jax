# Copyright 2026 The JAX Authors.
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

"""Tests for Pallas lowering determinism."""

import json
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
from jax._src.lib.mlir import ir
from jax.experimental import pallas
import jax.numpy as jnp

jax.config.parse_flags_with_absl()


@jax.jit
def nested_jit_func(x):
  return jax.jit(lambda x: x + 1.0)(x)


def pallas_kernel(x_ref, y_ref):
  y_ref[...] = nested_jit_func(x_ref[...])


def pallas_kernel_duplicate(x_ref, y_ref):
  y_ref[...] = nested_jit_func(x_ref[...])


@jax.jit
def stable_jit_func(x):
  return pallas.pallas_call(
      pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      name="stable_kernel_name",
  )(x)


@jax.jit
def stable_jit_func_duplicate(x):
  return pallas.pallas_call(
      pallas_kernel_duplicate,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      name="stable_kernel_name_duplicate",
  )(x)


def extract_pallas_body(lowered):
  """Extracts Pallas kernel body from lowered object."""
  module = lowered.compiler_ir()
  bodies = []

  def _find_backend_configs(op):
    is_tpu_custom_call = (
        "call_target_name" in op.attributes
        and "backend_config" in op.attributes
        and ir.StringAttr(op.attributes["call_target_name"]).value
        == "tpu_custom_call"
    )
    if is_tpu_custom_call:
      backend_config = op.attributes["backend_config"]
      config = json.loads(ir.StringAttr(backend_config).value)
      if "custom_call_config" in config:
        bodies.append(config["custom_call_config"]["body"])
    for region in op.regions:
      for block in region:
        for nested_op in block:
          _find_backend_configs(nested_op)

  _find_backend_configs(module.operation)
  assert len(bodies) == 1
  return bodies[0]


class PallasLoweringDeterminismTest(jtu.JaxTestCase):

  @jtu.run_on_devices("tpu")
  def testCallsiteAgnostic(self):
    if jaxlib_extension_version < 399:
      self.skipTest("TracebackScope requires jaxlib >= 399")

    def get_lowered():
      x = jnp.ones((8,), dtype=jnp.float32)
      return stable_jit_func.lower(x)

    jax.clear_caches()

    lowered0 = get_lowered()
    body0 = extract_pallas_body(lowered0)

    jax.clear_caches()

    def wrapper():
      return get_lowered()

    lowered1 = wrapper()
    body1 = extract_pallas_body(lowered1)

    self.assertEqual(body0, body1)

  def testOrderAgnostic(self):
    # TODO(b/476232048): Reenable once fixed.
    self.skipTest("Fix debug info in jit cache.")

    def get_pallas_body(f):
      x = jnp.ones((8,), dtype=jnp.float32)
      return extract_pallas_body(f.lower(x))

    jax.clear_caches()

    body_a0 = get_pallas_body(stable_jit_func)
    body_b0 = get_pallas_body(stable_jit_func_duplicate)

    jax.clear_caches()

    body_b1 = get_pallas_body(stable_jit_func_duplicate)
    body_a1 = get_pallas_body(stable_jit_func)

    self.assertEqual(body_a0, body_a1)
    self.assertEqual(body_b0, body_b1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
