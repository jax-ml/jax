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

"""XLA Composite Test."""

from absl.testing import absltest
import jax
from jax import export as jax_export
from jax._src import test_util as jtu
from jax._src.lax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo as stablehlo
from jax.interpreters import mlir
import jax.numpy as jnp
import jaxtyping

ArrayLike = jaxtyping.ArrayLike
Array = jaxtyping.Array


# Step 1: Define a jax primitive.
my_acos_p = lax.standard_unop(lax._float | lax._complex, "my_acos")


# Step 2: Define a jax api.
def my_acos(x: ArrayLike) -> Array:
  return my_acos_p.bind(x)


# Step 3: Define auto diff rule.
lax.ad.defjvp(
    my_acos_p,
    lambda g, x: lax.mul(g, -lax.rsqrt(lax._const(x, 1) - lax.square(x))),
)


# Step 4: Define lowering to stablehlo.composite.
def _composite_acos_lowering(
    ctx: mlir.LoweringRuleContext, arg: mlir.ir.BlockArgument
) -> mlir.ir.OpResultList:

  @jax.jit
  def my_acos_impl(x: ArrayLike) -> Array:
    return jnp.acos(x)

  # TODO(gunhyun): this implementation leaks a CallOp.
  lowered_fun = mlir.lower_fun(my_acos_impl, multiple_results=False)
  call_op = lowered_fun(ctx, arg)[0].owner

  composite = stablehlo.CompositeOp(
      [result.type for result in call_op.results],
      call_op.operands,
      name=ir.StringAttr.get("chlo.acos"),
      composite_attributes=ir.DictAttr.get({}),
      decomposition=call_op.attributes["callee"],
  )
  return composite.results


# Step 5: Register your custom composite lowering to stablehlo.composite.
mlir.register_lowering(my_acos_p, _composite_acos_lowering)


class XlaCompositeTest(jtu.JaxTestCase):

  def test_acos_composite(self):

    @jax.jit
    def f(x: ArrayLike) -> Array:
      return my_acos(x)

    x = jnp.array(1.0, dtype=jnp.float32)
    self.assertAllClose(jnp.acos(x), f(x))

    mlir_module = jax_export.export(f)(x).mlir_module()
    self.assertIn('stablehlo.composite "chlo.acos"', mlir_module)

if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
