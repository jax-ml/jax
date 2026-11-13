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

import base64
import importlib.util
import json
import os
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config as jax_config
from jax._src import test_util as jtu
from jax._src import tpu_custom_call
from jax._src.interpreters import mlir
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
      out_shape=jax.ShapeDtypeStruct.like(x),
      name="stable_kernel_name",
  )(x)


@jax.jit
def stable_jit_func_duplicate(x):
  return pallas.pallas_call(
      pallas_kernel_duplicate,
      out_shape=jax.ShapeDtypeStruct.like(x),
      name="stable_kernel_name_duplicate",
  )(x)


def extract_custom_call_config(lowered):
  """Extracts Pallas kernel custom_call_config from lowered object."""
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
        bodies.append(config["custom_call_config"])
    for region in op.regions:
      for block in region:
        for nested_op in block:
          _find_backend_configs(nested_op)

  _find_backend_configs(module.operation)
  assert len(bodies) == 1
  return bodies[0]


def extract_pallas_body(lowered):
  """Extracts Pallas kernel body from lowered object."""
  return extract_custom_call_config(lowered)["body"]


_KERNEL_SRC = """\
import jax
from jax.experimental import pallas as pl

def kernel(x_ref, y_ref):
  y_ref[...] = x_ref[...] + 1.0

@jax.jit
def f(x):
  return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct.like(x))(x)
"""


def lower_kernel_installed_at(root):
  """Lowers a Pallas kernel defined in `root`/pkg/mod.py."""
  path = os.path.join(root, "pkg", "mod.py")
  os.makedirs(os.path.dirname(path))
  with open(path, "w") as f:
    f.write(_KERNEL_SRC)
  spec = importlib.util.spec_from_file_location("mod", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  x = jax.ShapeDtypeStruct((8, 128), jnp.float32)
  return extract_custom_call_config(
      mod.f.trace(x).lower(lowering_platforms=("tpu",)))


class PallasLoweringDeterminismTest(jtu.JaxTestCase):

  @jtu.run_on_devices("tpu")
  def testCallsiteAgnostic(self):

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

  @jtu.run_on_devices("tpu")
  def testOrderAgnostic(self):
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

  @parameterized.parameters(True, False)
  @jtu.thread_unsafe_test()  # Mutates sys.path.
  def testInstallDirAgnostic(self, unzip):
    configs = []
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        jax_config.jax_mosaic_unzip_debug_locations(unzip),
    ):
      for name in ("install_a", "install_b"):
        root = os.path.join(tmp_dir, name)
        with mock.patch.object(sys, "path", [root, *sys.path]):
          configs.append(lower_kernel_installed_at(root))
    self.assertEqual(configs[0], configs[1])
    locs = base64.b64decode(configs[0]["debug_locations" if unzip else "body"])
    self.assertIn(b"pkg/mod.py", locs)
    self.assertNotIn(b"install_", locs)

  @jtu.thread_unsafe_test()  # Mutates sys.path.
  def testInstallDirAgnosticPrebuiltModule(self):
    asms = []
    for root in ("/install_a", "/install_b"):
      with mlir.make_ir_context(), ir.Location.unknown():
        module = ir.Module.parse(f'module {{}} loc("{root}/pkg/mod.py":1:2)')
        with mock.patch.object(sys, "path", [root, *sys.path]):
          asm, _, _ = tpu_custom_call._lower_mosaic_module_to_asm(
              module, unzip_debug_locations=False)
      asms.append(asm)
    self.assertEqual(asms[0], asms[1])


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
