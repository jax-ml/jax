# Copyright 2023 The JAX Authors.
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

import inspect
import re
import unittest

from absl.testing import absltest
import jax
from jax import lax
from jax._src import source_info_util
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version
from jax._src.lib.mlir import ir

jax.config.parse_flags_with_absl()


def find_operations(root, selector: str):
  """Provides a CSS-like selector for MLIR operations.

  Supported syntax: "op_name", "op_name[attr_name='attr_value']", and
  space-separated descendant queries. Example: "func.func[sym_name='outer']
  stablehlo.add"
  """
  parts = selector.split()

  def _parse_part(part):
    if part == "*":
      return (None, None, None)
    m = re.match(r'^([\w\.]+)?(?:\[([\w\.]+)=[\'"](.*?)[\'"]\])?$', part)
    if not m:
      raise ValueError(f"Invalid selector part: {part}")
    return m.groups()

  def _match_op(op, op_name, attr_name, attr_value):
    if op_name and op.OPERATION_NAME != op_name:
      return False
    if attr_name:
      if attr_name not in op.attributes:
        return False
      attr = op.attributes[attr_name]
      try:
        val = attr.value
      except AttributeError:
        val = str(attr)
        if val.startswith('"') and val.endswith('"'):
          val = val[1:-1]
      if val != attr_value:
        return False
    return True

  def _get_descendants(node):
    if isinstance(node, ir.Module):
      node = node.operation
    for region in node.regions:
      for block in region.blocks:
        for child_op in block.operations:
          yield child_op
          yield from _get_descendants(child_op)

  def _find(node, part_idx):
    if part_idx == len(parts):
      yield node
      return
    parsed = _parse_part(parts[part_idx])
    for desc in _get_descendants(node):
      if _match_op(desc, *parsed):
        if part_idx == len(parts) - 1:
          yield desc
        else:
          yield from _find(desc, part_idx + 1)

  return list(_find(root, 0))


def find_operation(root, selector: str):
  ops = find_operations(root, selector)
  if len(ops) != 1:
    raise ValueError(
        f"Expected exactly one operation matching '{selector}', found {len(ops)}"
    )
  return ops[0]


class SourceInfoTest(jtu.JaxTestCase):

  def test_inline_jit_location_uses_callee_location(self):

    # 'f' should be inlined into both 'g' and 'h', using the source line
    # information of the call site. In particular, the source line information
    # of 'h' should not refer to the source information of 'g'.
    @jax.jit(inline=True)
    def f(x):
      return lax.add(x, 3)

    def g(x):
      return lax.add(f(x), 4)

    def h(x):
      return lax.add(f(x), 5)

    for fn in (g, h):
      lines, fn_startline = inspect.getsourcelines(fn)
      fn_endline = fn_startline + len(lines)
      jaxpr = jax.make_jaxpr(fn)(2)
      for eqn in jaxpr.eqns:
        frame = source_info_util.user_frame(eqn.source_info.traceback)
        assert frame is not None, eqn
        self.assertLessEqual(fn_startline, frame.start_line)
        self.assertLessEqual(frame.end_line, fn_endline)

  @unittest.skipIf(jaxlib_extension_version < 409, reason="Needs newer jaxlib")
  def test_jit_traceback_boundaries_in_mlir(self):

    @jax.jit
    def inner(x):
      return lax.add(x, 1)

    @jax.jit
    def outer(x):
      y = lax.mul(x, 2)
      return inner(y)

    mlir_module = outer.lower(0).compiler_ir()

    add_op = find_operation(
        mlir_module, "func.func[sym_name='inner'] stablehlo.add"
    )
    add_loc_str = str(add_op.location)

    self.assertIn(
        '.inner"', add_loc_str, f"Expected 'inner' in location: {add_loc_str}"
    )
    self.assertNotIn(
        '.outer"', add_loc_str, f"Unexpected 'outer' in location: {add_loc_str}"
    )

    mul_op = find_operation(
        mlir_module, "func.func[sym_name='main'] stablehlo.multiply"
    )
    call_op = find_operation(
        mlir_module, "func.func[sym_name='main'] func.call"
    )

    for op in [mul_op, call_op]:
      loc_str = str(op.location)
      self.assertIn(
          '.outer"', loc_str, f"Expected 'outer' in location: {loc_str}"
      )
      self.assertNotIn(
          '.inner"', loc_str, f"Unexpected 'inner' in location: {loc_str}"
      )

  def test_while_loop_traceback_boundaries_in_mlir(self):

    @jax.jit
    def while_test(x):
      def cond(val):
        return lax.lt(val, 10)

      def body(val):
        return lax.add(val, 1)

      return lax.while_loop(cond, body, x)

    mlir_module = while_test.lower(0).compiler_ir()

    while_op = find_operation(
        mlir_module, "func.func[sym_name='main'] stablehlo.while"
    )
    while_loc_str = str(while_op.location)
    self.assertIn(
        '.while_test"',
        while_loc_str,
        f"Expected 'while_test' in location: {while_loc_str}",
    )
    add_op = find_operation(
        mlir_module,
        "func.func[sym_name='main'] stablehlo.while stablehlo.add",
    )
    cmp_op = find_operation(
        mlir_module,
        "func.func[sym_name='main'] stablehlo.while stablehlo.compare",
    )

    for op in [add_op, cmp_op]:
      loc_str = str(op.location)
      self.assertIn(
          '.while_test"',
          loc_str,
          f"Expected 'while_test' in descending location: {loc_str}",
      )

  def test_cond_traceback_boundaries_in_mlir(self):

    @jax.jit
    def cond_test(x):
      def true_fn(val):
        return lax.add(val, 1)

      def false_fn(val):
        return lax.mul(val, 2)

      return lax.cond(lax.lt(x, 10), true_fn, false_fn, x)

    mlir_module = cond_test.lower(0).compiler_ir()

    if_op = find_operation(
        mlir_module, "func.func[sym_name='main'] stablehlo.case"
    )
    if_loc_str = str(if_op.location)
    self.assertIn(
        '.cond_test"',
        if_loc_str,
        f"Expected 'cond_test' in location: {if_loc_str}",
    )

    # Check that specific ops inside the branches carry the same traceback boundary
    add_op = find_operation(
        mlir_module,
        "func.func[sym_name='main'] stablehlo.case stablehlo.add",
    )
    mul_op = find_operation(
        mlir_module,
        "func.func[sym_name='main'] stablehlo.case stablehlo.multiply",
    )

    for op in [add_op, mul_op]:
      loc_str = str(op.location)
      self.assertIn(
          '.cond_test"',
          loc_str,
          f"Expected 'cond_test' in descending location: {loc_str}",
      )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
