# Copyright 2025 The JAX Authors.
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
import sys

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.experimental import source_mapper
from jax.experimental.source_mapper import hlo
from jax.experimental.source_mapper import mlir


HLO_EXAMPLE = r"""HloModule m, entry_computation_layout={()->pred[]}

FileNames
1 "<embedded module>"
2 "experimental/module.py"
3 "yet/another/test.py"

FunctionNames
1 "main"
2 "method"

FileLocations
1 {file_name_id=1 function_name_id=1 line=153 end_line=153 column=2 end_column=31}
2 {file_name_id=3 function_name_id=2 line=35 end_line=35 column=2 end_column=24}
3 {file_name_id=2 function_name_id=2 line=83 end_line=83 column=2 end_column=15}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}


ENTRY %constant_pred () -> pred[] {
  ROOT %constant = pred[] constant(true), metadata={op_type="const" op_name="opname" stack_frame_id=1}
}"""


class SourceMapperTest(jtu.JaxTestCase):

  def setUp(self):
    if sys.platform == "win32":
      self.skipTest("Only works on non-Windows platforms")

  def test_jaxpr_pass(self):
    def jax_fn(x, y):
      return x + y
    test_x = jnp.array([1, 2, 3])
    test_y = jnp.array([4, 5, 6])
    source_maps = source_mapper.generate_sourcemaps(
        jax_fn,
        passes=source_mapper.filter_passes("jaxpr"))(test_x, test_y)
    self.assertLen(source_maps, 1)
    dump = source_maps[0]
    self.assertEqual(dump.pass_name, "jaxpr")
    self.assertIn("add a b", dump.generated_code)
    source_map = dump.source_map
    self.assertLen(source_map.sources, 1)
    self.assertEqual(source_map.sources[0],
                     source_mapper.canonicalize_filename(__file__))
    mappings = source_map.mappings
    self.assertLen(mappings, len(dump.generated_code.split("\n")) + 1)
    gen_col, file_idx, src_line, _ = mappings[0][0]
    # It's hard to guarantee at what column the add instruction will be
    # generated in the dump. We just sanity-check that it's greater than 0.
    self.assertGreater(gen_col, 0)
    # There is only one file, so we should map to that
    self.assertEqual(file_idx, 0)
    # These should line up with the function definition of jax_fn above.
    self.assertEqual(src_line, jax_fn.__code__.co_firstlineno)
    # TODO(justinfu): This fails on external but not internal builds.
    # self.assertEqual(src_col, 13)

  @parameterized.parameters(
      ("hlo:stable-hlo", "stablehlo.add", 13),
      ("hlo:original", "add", 0),
      # TODO(justinfu): Make the hlo:optimized test less strict.
      # ("hlo:optimized", "add", 0),
  )
  def test_hlo_passes(self, pass_name, expected_hlo_op, expected_col):
    del expected_col
    def jax_fn(x, y):
      return x + y
    test_x = jnp.array([1, 2, 3])
    test_y = jnp.array([4, 5, 6])
    source_maps = source_mapper.generate_sourcemaps(
        jax_fn,
        passes=source_mapper.filter_passes(pass_name))(test_x, test_y)
    self.assertLen(source_maps, 1)
    dump = source_maps[0]
    self.assertEqual(dump.pass_name, pass_name)
    self.assertIn(expected_hlo_op, dump.generated_code)
    source_map = dump.source_map
    self.assertLen(source_map.sources, 1)
    self.assertEqual(source_map.sources[0],
                     source_mapper.canonicalize_filename(__file__))
    mappings = source_map.mappings
    self.assertLen(mappings, len(dump.generated_code.split("\n")) + 1)
    nonempty_mappings = [m for m in mappings if m]
    self.assertGreaterEqual(len(nonempty_mappings), 1)
    gen_col, file_idx, src_line, _ = nonempty_mappings[0][0]
    self.assertGreater(gen_col, 0)
    # There is only one file, so we should map to that
    self.assertEqual(file_idx, 0)
    # These should line up with the function definition of jax_fn above.
    self.assertEqual(src_line, jax_fn.__code__.co_firstlineno)
    # TODO(justinfu): This fails on external but not internal builds.
    # self.assertEqual(src_col, expected_col)


class HLOParserTest(jtu.JaxTestCase):

  def test_hlo_parser(self):
    source_map = hlo._parse_hlo_new_format(HLO_EXAMPLE.split("\n"))
    print(source_map)
    self.assertLen(source_map.sources, 1)
    self.assertEqual(source_map.sources[0], "<embedded module>")
    mappings = source_map.mappings
    constant_line_idx = -1
    for i, line in enumerate(HLO_EXAMPLE.split("\n")):
      if r"ROOT %constant" in line:
        constant_line_idx = i
        break
    line_mappings = mappings[constant_line_idx]
    gen_col, file_idx, src_line, _ = line_mappings[0]

    self.assertEqual(
        file_idx, 0
    )  # "<embedded module>" is the first and only used source
    self.assertEqual(src_line, 152)  # 153 - 1
    self.assertEqual(gen_col, 2)


class MLIRParserTest(jtu.JaxTestCase):

  def test_callsite_fallback_and_chained_redirects(self):
    mlir_dump = """module {
  func.func @kernel(%arg0: f32) -> f32 {
    %0 = arith.addf %arg0, %arg0 : f32 loc(#loc4)
    %1 = arith.mulf %0, %arg0 : f32 loc(#loc7)
    %2 = arith.subf %1, %arg0 : f32 loc(#loc9)
    %3 = arith.divf %2, %arg0 : f32 loc(#loc3)
    %4 = arith.maximumf %3, %arg0 : f32 loc(#loc10)
    return %4 : f32 loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc1 = loc("user_kernel.py":42:6 to :31)
#loc2 = loc("helper.py":15:4)
#loc3 = loc("add")
#loc4 = loc(callsite(#loc2 at #loc1))
#loc5 = loc("add:"(#loc3))
#loc6 = loc("kernel_scope"(#loc1))
#loc7 = loc(callsite(#loc5 at #loc6))
#loc8 = loc(callsite(#loc3 at #loc7))
#loc9 = loc(callsite(#loc3 at #loc8))
#loc10 = loc(callsite(#loc11 at #loc10))
#loc11 = loc(callsite(#loc10 at #loc1))"""
    sm = mlir.create_mlir_sourcemap(mlir_dump)
    self.assertEqual(sm.sources, ["helper.py", "user_kernel.py"])

    # Decode cumulative VLQ state per line: (gen_col, file_idx, src_line, src_col)
    abs_file = 0
    abs_line = 0
    abs_col = 0
    decoded_by_line = {}
    for line_idx, group in enumerate(sm.mappings[:-1]):
      if not group:
        continue
      gen_col, d_file, d_line, d_col = group[0]
      abs_file += d_file
      abs_line += d_line
      abs_col += d_col
      decoded_by_line[line_idx] = (gen_col, abs_file, abs_line, abs_col)

    # Line 2 (%0): callsite(#loc2 at #loc1) prefers callee #loc2 (helper.py:15:4).
    self.assertEqual(decoded_by_line[2], (4, 0, 14, 4))
    # Line 3 (%1): callsite(#loc5 at #loc6) falls back from scoped NameLoc #loc5
    # to caller #loc6 -> #loc1 (user_kernel.py:42:6).
    self.assertEqual(decoded_by_line[3], (4, 1, 41, 6))
    # Line 4 (%2): chained callsite(#loc3 at callsite(#loc3 at #loc7)) resolves
    # through multiple bare NameLoc frames to user_kernel.py:42:6.
    self.assertEqual(decoded_by_line[4], (4, 1, 41, 6))
    # Line 5 (%3): bare NameLoc (#loc3) has no FileLineColLoc and is skipped cleanly.
    self.assertNotIn(5, decoded_by_line)
    # Line 6 (%4): cyclic callsite graph terminates without infinite recursion
    # and resolves via fallback branch to user_kernel.py:42:6.
    self.assertEqual(decoded_by_line[6], (4, 1, 41, 6))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
