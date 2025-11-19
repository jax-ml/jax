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

"""Tests whether the frontend attributes added by the context manager are

correctly propagated to the jaxpr and mlir.
"""

from absl.testing import absltest, parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import lax
from jax.experimental.xla_metadata import set_xla_metadata
import jax.numpy as jnp
import numpy as np

config.parse_flags_with_absl()


class XlaMetadataTest(jtu.JaxTestCase):

  def _assert_metadata_appears_once_per_op(
      self,
      hlo_text: str,
      expected_tagged_ops: list[str],
      metadata: dict[str, str],
  ):
    attribute_strings = [f'{k}="{v}"' for k, v in metadata.items()]
    op_with_metadata_count = {op: 0 for op in expected_tagged_ops}

    for line in hlo_text.splitlines():
      for op in expected_tagged_ops:
        if (str(op + "(") in line and all(attr in line for attr in attribute_strings)
            and "frontend_attributes=" in line):
          op_with_metadata_count[op] += 1

    for op in op_with_metadata_count:
      self.assertEqual(
          op_with_metadata_count[op],
          1,
          f"Expected op '{op}' to have the metadata exactly once,"
          f" but found it {op_with_metadata_count[op]} times\n"
          f"Metadata: {metadata}\n"
          f"HLO Graph:\n\n{hlo_text}",
      )

  def test_f_jitted(self):
    @jax.jit
    def f(a, b):
      with set_xla_metadata(a="b"):
        return a + b

    f_jaxpr = jax.make_jaxpr(f)(1, 2)
    eqns = f_jaxpr.eqns
    for eq in eqns[1:]:
      self.assertDictEqual(eq.ctx.attributes, {"a": "b"})

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "b"}', f_lowered_text)

  def test_f_jitted_bool_attributes(self):
    @jax.jit
    def f(a, b):
      with set_xla_metadata(a=True):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "true"}', f_lowered_text)

  def test_f_jitted_int_attributes(self):
    @jax.jit
    def f(a, b):
      with set_xla_metadata(a=10):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "10"}', f_lowered_text)

  def test_decorator(self):
    @set_xla_metadata(a="b")
    @jax.jit
    def f(a, b):
      return a + b

    f_jaxpr = jax.make_jaxpr(f)(1, 2)
    eqns = f_jaxpr.eqns
    for eq in eqns[1:]:
      self.assertDictEqual(eq.ctx.attributes, {"a": "b"})

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {a = "b"}', f_lowered_text)

  def test_decorator_and_context_manager_nested(self):
    @set_xla_metadata(a="b")
    @jax.jit
    def f(a, b):
      with set_xla_metadata(c="d"):
        return a + b

    f_lowered_text = f.lower(1.0, 2.0).as_text()
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b", c = "d"}',
        f_lowered_text,
    )

  def test_f_nonjitted(self):
    def f_add(a, b):
      return lax.add(a, b)

    arg1 = jnp.arange(2)
    with set_xla_metadata(a="b"):
      self.assertIn(
          'mhlo.frontend_attributes = {a = "b"}',
          jax.jit(f_add).lower(arg1, arg1).as_text(),
      )

  def test_f_attributes_overwrite(self):
    @jax.jit
    def g(a, b):
      return a * b

    with set_xla_metadata(a="b"):

      @jax.jit
      def f(a, b):
        with set_xla_metadata(a="c"):
          return a + b

      f_lowered_text = f.lower(1.0, 2.0).as_text()
      self.assertIn('mhlo.frontend_attributes = {a = "c"}', f_lowered_text)
      self.assertIn(
          'mhlo.frontend_attributes = {a = "b"}', g.lower(1.0, 2.0).as_text()
      )
    self.assertNotIn("mhlo.frontend_attributes", g.lower(1.0, 2.0).as_text())

  def test_f_attributes_merge(self):
    with set_xla_metadata(key1="val1"):

      @jax.jit
      def f(a, b):
        with set_xla_metadata(key2="val2"):
          return a + b

      f_lowered_text = f.lower(1.0, 2.0).as_text()
      self.assertIn(
          'mhlo.frontend_attributes = {key1 = "val1", key2 = "val2"}',
          f_lowered_text,
      )

  def test_attr_caching_jit(self):
    @jax.jit
    def f_add_jit(a, b):
      return a + b

    with set_xla_metadata(b="c"):
      f_add_lowered1 = f_add_jit.lower(2.0, 3.0).as_text()
    # Expect no attributes in the mlir.
    f_add_lowered2 = f_add_jit.lower(1.0, 2.0).as_text()
    with set_xla_metadata(c="d"):
      f_add_lowered3 = f_add_jit.lower(4.0, 5.0).as_text()
    self.assertIn('mhlo.frontend_attributes = {b = "c"}', f_add_lowered1)
    self.assertNotIn("mhlo.frontend_attributes = {}", f_add_lowered2)
    self.assertNotIn('mhlo.frontend_attributes = {b = "c"}', f_add_lowered2)
    self.assertNotIn('mhlo.frontend_attributes = {c = "d"}', f_add_lowered2)
    self.assertIn('mhlo.frontend_attributes = {c = "d"}', f_add_lowered3)

  def test_attr_caching_nonjit(self):
    def f_add(a, b):
      return lax.add(a, b)

    arg1 = jnp.arange(2)
    arg2 = jnp.arange(2) + 1
    arg3 = jnp.arange(2) + 2
    with set_xla_metadata(b="c"):
      self.assertIn(
          'mhlo.frontend_attributes = {b = "c"}',
          jax.jit(f_add).lower(arg1, arg1).as_text(),
      )
    # Expect no attributes in the jaxpr.
    self.assertNotIn(
        "mhlo.frontend_attributes",
        jax.jit(f_add).lower(arg2, arg2).as_text(),
    )

    with set_xla_metadata(c="d"):
      self.assertIn(
          'mhlo.frontend_attributes = {c = "d"}',
          jax.jit(f_add).lower(arg3, arg3).as_text(),
      )

  def test_axpy(self):
    @jax.jit
    def axpy(a, x, y):
      with set_xla_metadata(a="b"):
        return a * x + y

    for line in axpy.lower(1.0, 2.0, 3.0).as_text().split("\n"):
      if "stablehlo.multiply" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "stablehlo.add" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)

  def test_while(self):
    @jax.jit
    def f(a):
      with set_xla_metadata(a="b"):
        return jax.lax.while_loop(lambda x: x < 10, lambda x: x + 1, a)

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(1.0).as_text()
    )

  def test_while_condition_body(self):
    @jax.jit
    def f_condition(x):
      with set_xla_metadata(a="b"):
        return x < 10

    @jax.jit
    def f_body(x):
      with set_xla_metadata(a="c"):
        return x + 1

    @jax.jit
    def while_fn(a):
      return jax.lax.while_loop(f_condition, f_body, a)

    for line in while_fn.lower(1.0).as_text().split("\n"):
      if "stablehlo.compare" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "stablehlo.add" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "c"}', line)

  def test_cond_annotates_branches(self):
    sin = jnp.sin
    cos = jnp.cos

    @jax.jit
    def f(x):
      with set_xla_metadata(a="b"):
        return jax.lax.cond(x < 0., sin, cos, x)

    hlo_lines = f.lower(1.).as_text().split("\n")
    sin_hlo, = (line for line in hlo_lines if "stablehlo.sine"   in line)
    cos_hlo, = (line for line in hlo_lines if "stablehlo.cosine" in line)
    self.assertIn('mhlo.frontend_attributes = {a = "b"}', sin_hlo)
    self.assertIn('mhlo.frontend_attributes = {a = "b"}', cos_hlo)

  def test_cond_annotates_branches_and_none_unsets(self):
    sin = jnp.sin

    def cos(x):
      with set_xla_metadata(a=None):
        return jnp.cos(x)

    @jax.jit
    def f(x):
      with set_xla_metadata(a="b"):
        return jax.lax.cond(x < 0., sin, cos, x)

    hlo_lines = f.lower(1.).as_text().split("\n")
    sin_hlo, = (line for line in hlo_lines if "stablehlo.sine"   in line)
    cos_hlo, = (line for line in hlo_lines if "stablehlo.cosine" in line)
    self.assertIn(   'mhlo.frontend_attributes = {a = "b"}', sin_hlo)
    self.assertNotIn('mhlo.frontend_attributes = {a = "b"}', cos_hlo)

  def test_nested_jit(self):
    @jax.jit
    def f(x, y):
      with set_xla_metadata(a="b"):
        z = x * y

        @jax.jit
        def g(z):
          with set_xla_metadata(c="d"):
            return z**2 + 1

        return g(z)

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b", c = "d"}',
        f.lower(1.0, 2.0).as_text(),
    )

  def test_grad(self):
    @jax.jit
    def f(x, y):
      with set_xla_metadata(a="b"):
        return jax.grad(lambda x: x**3 + y**2 + jnp.sin(x))(x)

    f_jaxpr = jax.make_jaxpr(f)(1.0, 2.0)
    eqns = f_jaxpr.eqns
    for eq in eqns[1:]:
      self.assertDictEqual(eq.ctx.attributes, {"a": "b"})

    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(1.0, 2.).as_text()
    )

  def test_grad_outside_ctx(self):
    @jax.jit
    def f(x):
      with set_xla_metadata(a="b"):
        return x**3 + x**2 + jnp.sin(x)

    grad_fn = jax.jit(jax.grad(f))
    for line in grad_fn.lower(1.0).as_text().split("\n"):
      if "stablehlo.cosine" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "call @integer_pow" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)

  def test_vmap(self):
    dct = {"a": 0.0, "b": jnp.arange(5.0)}

    @jax.jit
    def f(dct, x):
      with set_xla_metadata(a="b"):
        return dct["a"] + dct["b"] + x

    with set_xla_metadata(a="d"):
      f_vmap = jax.vmap(f, in_axes=({"a": None, "b": 0}, None))
      f_jaxpr = jax.make_jaxpr(f_vmap)(dct, 1.0)
      eqns = f_jaxpr.eqns
      for eq in eqns[1:]:
        self.assertDictEqual(eq.ctx.attributes, {"a": "d"})
    @jax.jit
    def f2(x, y):
      with set_xla_metadata(a="b"):
        return (x + y, y * 2.0)

    f2_vmap = jax.vmap(f2, in_axes=(0, None))
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}',
        jax.jit(f2_vmap).lower(jnp.arange(5.0), 1.0).as_text(),
    )

  def test_multiple_instructions(self):
    @jax.jit
    def f(x, a):
      y = jnp.matmul(x, x)
      with set_xla_metadata(a="b"):
        return y + a

    for line in f.lower(jnp.arange(5.0), 1.0).as_text().split("\n"):
      # matmul doesn't have attributes
      if "stablehlo.dot_general" in line:
        self.assertNotIn('mhlo.frontend_attributes = {a = "b"}', line)
      if "stablehlo.add" in line:
        self.assertIn('mhlo.frontend_attributes = {a = "b"}', line)

  def test_softmax(self):
    @jax.jit
    def f(x):
      with set_xla_metadata(a="b"):
        return jax.nn.softmax(x)
    self.assertIn(
        'mhlo.frontend_attributes = {a = "b"}', f.lower(jnp.arange(5.0)).as_text()
    )

  @parameterized.parameters(
      ("x*x", lambda x: x * x, "multiply"),
      ("sin(x)", jnp.sin, "sin"),
      ("tanh(x)", jnp.tanh, "tanh"),
      ("1/x", lambda x: 1 / x, "divide"),
      ("sinc(x)", jnp.sinc, "call"),
  )
  def test_value_tagging(self, name, fn, expected_tagged_op):
    metadata = {"test_value_tagging": name}

    def wrapped_fn(x):
      return set_xla_metadata(fn(x), **metadata)

    x_scalar = jnp.array(0.7)
    text = jax.jit(wrapped_fn).lower(x_scalar).as_text("hlo")
    self._assert_metadata_appears_once_per_op(
        text, [expected_tagged_op], metadata)

  @parameterized.parameters(
      ("x*x", lambda x: x * x, "add"),
      ("sin(x)", jnp.sin, "cosine"),
      ("tanh(x)", jnp.tanh, "add"),
      ("1/x", lambda x: 1 / x, "negate"),
      ("sinc(x)", jnp.sinc, "call"),
  )
  def test_value_grad_tagging(self, name, fn, expected_tagged_op):
    metadata = {"test_value_grad_tagging": name}

    @jax.custom_vjp
    def wrapped_fn(x):
      return fn(x)

    def fwd(*args):
      primal_out, vjp_fn = jax.vjp(fn, *args)
      return primal_out, vjp_fn

    def bwd(vjp_fn, cts_in):
      cts_out = vjp_fn(cts_in)
      cts_out = set_xla_metadata(cts_out, **metadata)
      return cts_out

    wrapped_fn.defvjp(fwd, bwd)

    x_scalar = jnp.array(0.7)
    text = jax.jit(jax.grad(wrapped_fn)).lower(x_scalar).as_text("hlo")
    self._assert_metadata_appears_once_per_op(
        text, [expected_tagged_op], metadata)

  def test_vmap_multi_input_output_value_tagging(self):
    metadata = {"test_vmap_multi_input_output_value_tagging": "value"}

    fn = lambda x, y, z: (x @ y + z, z - x @ y)

    @jax.vmap
    def vmapped_fn(x_item, y_item, z_item):
      return set_xla_metadata(fn(x_item, y_item, z_item), **metadata)

    batch_size, num_rows, num_cols = 4, 5, 6
    rng = np.random.default_rng(0)
    x_batch = rng.random((batch_size, num_rows, num_cols)).astype(np.float32)
    y_batch = rng.random((batch_size, num_cols, num_rows)).astype(np.float32)
    z_batch = rng.random((batch_size, num_rows, num_rows)).astype(np.float32)
    inputs = (x_batch, y_batch, z_batch)

    text = jax.jit(vmapped_fn).lower(*inputs).as_text("hlo")
    self._assert_metadata_appears_once_per_op(
        text, ["add", "subtract"], metadata)

  def test_sharding_support_value_tagging(self):
    mesh = jtu.create_mesh((1,), "data")
    np_inp = np.arange(8, dtype=np.float32)
    arr = jax.device_put(np_inp, jax.NamedSharding(mesh, jax.P("data")))

    metadata = {"test_sharding_support_value_tagging": "value"}

    @jax.jit
    def wrapped_fn(x):
      return set_xla_metadata(x * 2.0, **metadata)

    text = jax.jit(wrapped_fn).lower(arr).as_text("hlo")
    self._assert_metadata_appears_once_per_op(text, ["multiply"], metadata)

  def test_scan_support_value_tagging(self):
    metadata = {"test_scan_support_value_tagging": "value"}

    fn = lambda carry, x: (carry + x) * 2.0

    def scan_body_val_with_metadata(carry, x):
      tagged_result = set_xla_metadata(fn(carry, x), **metadata)
      return tagged_result, tagged_result

    def scan_fn(init_carry, inputs_arr):
      return jax.lax.scan(scan_body_val_with_metadata, init_carry, inputs_arr)

    inputs = (jnp.array(0.0), jnp.arange(1, 4, dtype=jnp.float32))
    text = jax.jit(scan_fn).lower(*inputs).as_text("hlo")
    self._assert_metadata_appears_once_per_op(text, ["multiply"], metadata)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
