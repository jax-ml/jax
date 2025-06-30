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
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import equations

config.parse_flags_with_absl()

C = equations.ConstantExpression
Eq = equations.Equation
V = equations.Variable


class EquationSystemTest(parameterized.TestCase):

  def test_equation_system_is_unsatisfiable_if_assignments_are_incompatible(self):
    v0 = V(0)
    layout0, layout1 = [C(mgpu.WGSplatFragLayout((1, i))) for i in (1, 2)]
    system = equations.EquationSystem(
        equations=[Eq(v0, layout0), Eq(v0, layout1)],
    )
    self.assertIsInstance(equations.simplify(system), equations.Unsatisfiable)

  def test_simplify_equation_system_removes_tautological_equations(self):
    v0, v1 = V(0), V(1)
    system = equations.EquationSystem(
        equations=[Eq(v0, v1), Eq(v0, v0)],
    )
    self.assertLen(equations.simplify(system).equations, 1)

  def test_simplify_equation_system_of_simplified_system_is_noop(self):
    v0, v1 = V(0), V(1)
    system = equations.EquationSystem(
        equations=[Eq(v0, v1)],
    )
    self.assertEqual(equations.simplify(system), system)

  def test_simplify_equation_system_assigns_variables_with_known_equations(self):
    v0, v1 = V(0), V(1)
    layout = C(mgpu.WGSplatFragLayout((1, 1)))

    with self.subTest("left-to-right-assignment"):
      system = equations.EquationSystem(
          equations=[Eq(v0, layout), Eq(v0, v1)],
      )
      self.assertEqual(
          equations.simplify(system),
          equations.EquationSystem(assignments={v0: layout, v1: layout})
      )

    with self.subTest("right-to-left-assignment"):
      system = equations.EquationSystem(
          equations=[Eq(v1, layout), Eq(v0, v1)],
      )
      self.assertEqual(
          equations.simplify(system),
          equations.EquationSystem(assignments={v0: layout, v1: layout})
      )

  def test_equation_system_unknowns_are_all_the_variables_without_assignment(self):
    v0, v1, v2 = V(0), V(1), V(2)
    layout = C(mgpu.WGSplatFragLayout((1, 1)))
    system = equations.EquationSystem(assignments={v0: layout},
                                      equations=[Eq(v1, v2)])
    self.assertSequenceEqual(system.unknowns(), [v1, v2])

if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
