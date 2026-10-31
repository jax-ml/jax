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

"""Tests for pgo_nsys_converter."""

from absl.testing import absltest
from jax.tools import pgo_nsys_converter


class PgoNsysConverterTest(absltest.TestCase):

  def test_extract_hlo_op_name_tsl_thunk_format(self):
    self.assertEqual(
        pgo_nsys_converter.extract_hlo_op_name(
            "TSL:Thunk:#name=jit(train_step)/dot_general,"
            "hlo_op=custom-call.12,unique_hlo_op_id=412#"
        ),
        "custom-call.12",
    )
    self.assertEqual(
        pgo_nsys_converter.extract_hlo_op_name(
            ":Thunk:#hlo_op=loop_add_fusion.5,unique_hlo_op_id=88#"
        ),
        "loop_add_fusion.5",
    )
    self.assertEqual(
        pgo_nsys_converter.extract_hlo_op_name("Thunk:#hlo_op=fusion.1#"),
        "fusion.1",
    )

  def test_extract_hlo_op_name_xla_domain_format(self):
    self.assertEqual(
        pgo_nsys_converter.extract_hlo_op_name("XLA:custom-call.12"),
        "custom-call.12",
    )
    self.assertEqual(
        pgo_nsys_converter.extract_hlo_op_name(":loop_add_fusion.5"),
        "loop_add_fusion.5",
    )

  def test_extract_hlo_op_name_excludes_command_buffer_and_module(self):
    self.assertIsNone(
        pgo_nsys_converter.extract_hlo_op_name(
            "TSL:Thunk:#hlo_op=command_buffer#"
        )
    )
    self.assertIsNone(
        pgo_nsys_converter.extract_hlo_op_name("XLA:command_buffer")
    )
    self.assertIsNone(
        pgo_nsys_converter.extract_hlo_op_name(
            "TSL:XlaModule:#hlo_module=pjit_train_step,program_id=1#"
        )
    )
    self.assertIsNone(pgo_nsys_converter.extract_hlo_op_name("XLA:XlaModule"))


if __name__ == "__main__":
  absltest.main()
