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
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax.experimental import source_mapper


class SourceMapperTest(jtu.JaxTestCase):

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
    self.assertLen(nonempty_mappings, 1)
    gen_col, file_idx, src_line, _ = nonempty_mappings[0][0]
    self.assertGreater(gen_col, 0)
    # There is only one file, so we should map to that
    self.assertEqual(file_idx, 0)
    # These should line up with the function definition of jax_fn above.
    self.assertEqual(src_line, jax_fn.__code__.co_firstlineno)
    # TODO(justinfu): This fails on external but not internal builds.
    # self.assertEqual(src_col, expected_col)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
