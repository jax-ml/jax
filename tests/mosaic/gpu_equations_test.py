# Copyright 2025 The JAX Authors. All Rights Reserved.
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
"""Tests for Mosaic GPU's `equations` module."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations

config.parse_flags_with_absl()

C = equations.ConstantExpression
Eq = equations.Equation


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  return context


class EquationSystemTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()
    with ir.InsertionPoint(self.module.body):
      bf16 = ir.BF16Type.get()
      op = arith.ConstantOp(bf16, ir.FloatAttr.get(bf16, 1.0))
      self.variable = lambda i: equations.Variable(op, equations.VariableType.OPERAND, i)

  def test_equation_system_is_unsatisfiable_if_assignments_are_incompatible(self):
    v0 = self.variable(0)
    layout0, layout1 = [C(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = equations.EquationSystem(
        unknowns=set([v0]),
        equations=set([Eq(v0, layout0), Eq(v0, layout1)]),
    )
    self.assertIsInstance(equations.simplify(system), equations.Unsatisfiable)

  def test_simplify_equation_system_removes_tautological_equations(self):
    v0, v1 = self.variable(0), self.variable(1)
    system = equations.EquationSystem(
        unknowns=set([v0, v1]),
        equations=set([Eq(v0, v1), Eq(v0, v0)]),
    )
    self.assertLen(equations.simplify(system).equations, 1)

  def test_simplify_equation_system_of_simplified_system_is_noop(self):
    v0, v1 = self.variable(0), self.variable(1)
    system = equations.EquationSystem(
        unknowns=set([v0, v1]),
        equations=set([Eq(v0, v1)]),
    )
    self.assertEqual(equations.simplify(system), system)

  def test_simplify_equation_system_assigns_variables_with_known_equations(self):
    v0, v1 = self.variable(0), self.variable(1)
    layout = C(mgpu.WGSplatFragLayout((1, 1)))

    with self.subTest("left-to-right-assignment"):
      system = equations.EquationSystem(
          unknowns=set([v0, v1]),
          equations=set([Eq(v0, layout), Eq(v0, v1)]),
      )
      self.assertEqual(
          equations.simplify(system),
          equations.EquationSystem(solution={v0: layout, v1: layout})
      )

    with self.subTest("right-to-left-assignment"):
      system = equations.EquationSystem(
          unknowns=set([v0, v1]),
          equations=set([Eq(v1, layout), Eq(v0, v1)]),
      )
      self.assertEqual(
          equations.simplify(system),
          equations.EquationSystem(solution={v0: layout, v1: layout})
      )


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
