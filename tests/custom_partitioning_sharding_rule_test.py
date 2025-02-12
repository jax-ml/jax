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

from absl.testing import absltest
from jax._src import test_util as jtu
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import sdy
from jax._src.custom_partitioning_sharding_rule import ArrayMapping, BATCHING, CompoundFactor, sdy_sharding_rule_to_mlir, str_to_sdy_sharding_rule, SdyShardingRule
from jax._src.lib.mlir.dialects import hlo as stablehlo


class SdyShardingRuleTest(jtu.JaxTestCase):
  def test_compound_factor_not_enough_factors(self):
    with self.assertRaisesRegex(ValueError, "A compound factor should contain at least two factors"):
      CompoundFactor("i")

  def test_compound_factor_batching_now_allowed(self):
    with self.assertRaisesRegex(ValueError, "Ellipsis can't be used in a compound factor"):
      CompoundFactor(BATCHING, "i")

  def test_compound_factor_element_not_a_str(self):
    with self.assertRaisesRegex(ValueError, "Each element of CompoundFactor must be a str"):
      CompoundFactor("i", 2)

  def test_compound_factor_str(self):
    c = CompoundFactor("i", "j", "k")
    self.assertEqual(str(c), "('i', 'j', 'k')")

  def test_value_mapping_element_not_a_str_or_compound_factor(self):
    with self.assertRaisesRegex(ValueError, "Each element of ArrayMapping must be a str or CompoundFactor"):
      ArrayMapping(CompoundFactor("i", "j"), 3)

  def test_value_mapping_factor_name_not_start_with_letter(self):
    with self.assertRaisesRegex(ValueError, "Factor names have to start with a letter"):
      ArrayMapping("3i", "j")

  def test_value_mapping_ellipsis_not_first(self):
    with self.assertRaisesRegex(ValueError, "Ellipsis can only be used at the beginning of a dimension"):
      ArrayMapping("i_j", BATCHING)

  def test_value_mapping_str(self):
    v = ArrayMapping(f"{BATCHING}2", "m", CompoundFactor("i", "j"), "k")
    self.assertEqual(str(v), f"('{BATCHING}2', 'm', ('i', 'j'), 'k')")

  def test_sdy_sharding_rule_factor_size_not_used(self):
    with self.assertRaisesRegex(ValueError, "Factor k is not used"):
      SdyShardingRule(("i",), ("j",), k=10)

  def test_sdy_sharding_rule_factor_sizes_missing(self):
    with self.assertRaisesRegex(
        ValueError,
        "Factor k is only used in compound factors; must specify its size"):
      SdyShardingRule((ArrayMapping("i"), ArrayMapping("j")),
                      (ArrayMapping(CompoundFactor("j", "k")),))

  def test_sdy_sharding_rule_factor_size_not_necessary(self):
    with self.assertRaisesRegex(
        ValueError,
        "Factor i represents a whole dimension; do not specify its size"):
      SdyShardingRule((ArrayMapping("i"),), (ArrayMapping("j"),), i=10)

  def test_sdy_sharding_rule_compound_factor_size_not_necessary(self):
    with self.assertRaisesRegex(
        ValueError,
        "Factor i represents a whole dimension; do not specify its size"):
      SdyShardingRule((ArrayMapping(CompoundFactor("i", "j")),),
                      (ArrayMapping("i"),), i=10, j=20)

  def test_sdy_sharding_rule_str(self):
    r = SdyShardingRule((ArrayMapping("i"), ArrayMapping("j")),
                        (ArrayMapping(CompoundFactor("j", "k")),), k=10)
    self.assertEqual(str(r), "SdyShardingRule((('i',), ('j',)), ((('j', 'k'),),), {'k': 10})")


class StrToSdyShardingRuleTest(jtu.JaxTestCase):

  def test_rule_is_not_a_str(self):
    with self.assertRaisesRegex(TypeError, "rule must be a str"):
      str_to_sdy_sharding_rule(1)

  def test_factor_sizes_is_not_a_proper_dict(self):
    with self.assertRaisesRegex(
        TypeError, "factor_sizes must be a dict of str to int"):
      str_to_sdy_sharding_rule("i->j", i="j")

  def test_sharding_rule_ellipsis_not_complete(self):
    with self.assertRaisesRegex(
        ValueError, "Character '.' must be used inside ellipsis '...'"):
      str_to_sdy_sharding_rule(".i -> j")

  def test_sharding_rule_invalid_factor_name(self):
    with self.assertRaisesRegex(ValueError, "Factor names have to start with a letter"):
      str_to_sdy_sharding_rule("2i -> j")

  def test_sharding_rule_missing_results(self):
    with self.assertRaisesRegex(ValueError, "There is no -> in rule"):
      str_to_sdy_sharding_rule("i")

  def test_sharding_rule_inbalenced_brackets(self):
    with self.assertRaisesRegex(ValueError, "Brackets are not balanced"):
      str_to_sdy_sharding_rule("i j, k)->j")

  def test_sharding_rule_inbalenced_brackets2(self):
    with self.assertRaisesRegex(ValueError, "Brackets are not balanced"):
      str_to_sdy_sharding_rule("i (j k->j")

  def test_sharding_rule_empty_compound_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Brackets should contain at least two factors"):
      str_to_sdy_sharding_rule("i ( ) j k->j")

  def test_sharding_rule_one_factorcompound_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Brackets should contain at least two factors"):
      str_to_sdy_sharding_rule("i (j ) k->j")

  def test_sharding_rule_nested_brackets(self):
    with self.assertRaisesRegex(
        ValueError, "Compound factors should be one level"):
      str_to_sdy_sharding_rule("i (j (k))->j")

  def test_sharding_rule_unknown_char(self):
    with self.assertRaisesRegex(ValueError, "Unknown character"):
      str_to_sdy_sharding_rule("i; j->j")

  def test_sharding_rule_unknown_single_char_ellipse(self):
    with self.assertRaisesRegex(ValueError, "Unknown character"):
      str_to_sdy_sharding_rule("…j->…j")

  def test_sharding_rule_ellipsis_not_leading_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Ellipsis can only be used at the beginning of a dimension"):
      str_to_sdy_sharding_rule("i ... -> j")

  def test_sharding_rule_ellipsis_inside_compound_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Ellipsis can only be used at the beginning of a dimension"):
      str_to_sdy_sharding_rule("i, (..., j) -> j")

  def test_sharding_rule_scalar_operand_scalar_result(self):
    rule = str_to_sdy_sharding_rule("->")
    self.assertEqual(str(rule), "SdyShardingRule(((),), ((),), {})")

  def test_sharding_rule_one_scalar_operand(self):
    rule = str_to_sdy_sharding_rule("i j, , k->j")
    self.assertEqual(
        str(rule), "SdyShardingRule((('i', 'j'), (), ('k',)), (('j',),), {})")

  def test_sharding_rule_factor_elementwise_add(self):
    # An ellipsis without a number ... is treated as the same as ...0.
    rule = str_to_sdy_sharding_rule("...0 i j, ...1 i j -> ...i j")
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('…0', 'i', 'j'), ('…1', 'i', 'j')), (('…0', 'i',"
        " 'j'),), {})")

  def test_sharding_rule_factor_vector_scalar_add(self):
    rule = str_to_sdy_sharding_rule("...87 i,  -> ...87 i")
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('…87', 'i'), ()), (('…87', 'i'),), {})")

  def test_sharding_rule_factor_reshape_combining(self):
    rule = str_to_sdy_sharding_rule("i j -> (i j)")
    self.assertEqual(
        str(rule), "SdyShardingRule((('i', 'j'),), ((('i', 'j'),),), {})")

  def test_sharding_rule_factor_reshape_reordering(self):
    rule = str_to_sdy_sharding_rule("(j i) -> (i j)", i=10, j=20)
    self.assertEqual(
        str(rule),
        "SdyShardingRule(((('j', 'i'),),), ((('i', 'j'),),), {'i': 10, 'j':"
        " 20})")

  def test_sharding_rule_factor_compound_then_individual(self):
    rule = str_to_sdy_sharding_rule("(i j) (j k) i -> j k")
    self.assertEqual(
        str(rule),
        "SdyShardingRule(((('i', 'j'), ('j', 'k'), 'i'),), (('j', 'k'),), {})")

  def test_sharding_rule_factor_individual_then_compound(self):
    rule = str_to_sdy_sharding_rule("i j k -> (i j) (j k)")
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('i', 'j', 'k'),), ((('i', 'j'), ('j', 'k')),), {})")

  def test_sharding_rule_factor_infer_k(self):
    rule = str_to_sdy_sharding_rule("i_ (j k)-> j foo (m bar_24)", k=10, m=10, bar_24=20)
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('i_', ('j', 'k')),), (('j', 'foo', ('m', 'bar_24'))"
        ",), {'k': 10, 'm': 10, 'bar_24': 20})")


class SdyShardingRuleConversionTest(jtu.JaxTestCase):

  def run(self, result=None):
    with ir.Context() as ctx, ir.Location.unknown(ctx):
      sdy.register_dialect(ctx)
      stablehlo.register_dialect(ctx)
      module = ir.Module.create()
      with ir.InsertionPoint(module.body):
        super().run(result)

  def get_tensor_type(self, shape):
    return ir.RankedTensorType.get(shape, ir.F32Type.get())

  def create_tensor_value(self, shape):
    return ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type(shape)],
        attributes=dict(call_target_name=ir.StringAttr.get("dummy_target"))
        ).result

  def test_conversion_rule_op_mismatch_in_operands_num(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("i j-> i j")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule has 1 operands, but the operation has 2 operands"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,])

  def test_conversion_rule_op_mismatch_in_operands_rank(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("i j, i j k-> i j")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule 1th operand has rank 3, but the operation 1th "
        "operand has rank 2"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,])

  def test_conversion_rule_op_mismatch_in_results_num(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0,
            opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("i j, i j -> i j, i j")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule has 2 results, but the operation has 1 results"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,])

  def test_conversion_rule_op_mismatch_in_results_dim(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("i j, i j -> i j k")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule 0th result has rank 3, but the operation 0th "
        "result has rank 2"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,])

  def test_conversion_factor_has_two_sizes(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 64))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("i j, i j -> i j")
    with self.assertRaisesRegex(
        ValueError,
        "Factor j corresponds to two sizes: 32 and 64"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,])

  def test_conversion_batching_dim_has_two_sizes(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 64))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("..., ... -> ...")
    with self.assertRaisesRegex(
        ValueError,
        "Batching dimension 0_1 corresponds to two sizes: 32 and 64"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,],)

  def test_conversion_invalid_batching_dim(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("... i j k, ... i j k -> ... i j k")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule 0th operand has rank 3, but the operation 0th operand has rank 2"):
      sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type, result.operands[1].type],
        [result.result.type,])

  def test_conversion_compound_dimension_size_mismatch(self):
    opnd = self.create_tensor_value((2, 4))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((9,))],
        operands=[opnd,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("i j -> (i j)")
    with self.assertRaisesRegex(
        ValueError,
        "0th result actual size 9 doesn't match the size 8 derived from the"
        " compound factors"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type],
          [result.result.type,])

  def test_conversion_elementwise_rule_mismatching_ellipsis_rank(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16,))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("..., ... -> ...")
    with self.assertRaisesRegex(
        ValueError,
        "Ellipsis represents different number of leading dimensions 2 and 1"):
      sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type, result.operands[1].type],
          [result.result.type,])

  def test_conversion_compound_then_individual(self):
    opnd = self.create_tensor_value((8,))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((2,4))],
        operands=[opnd,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("(i j) -> i j")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
          [result.operands[0].type],
          [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([ij])->([i, j]) {i=2, j=4}>")

  def test_conversion_elementwise_rule_scalar_instance(self):
    opnd0 = self.create_tensor_value(())
    opnd1 = self.create_tensor_value(())
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type(())],
        operands=[opnd0, opnd1],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("..., ... -> ...")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type, result.operands[1].type],
        [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([], [])->([])>")

  def test_conversion_elementwise_rule_2D_instance(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("..., ... -> ...")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type, result.operands[1].type],
        [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=16, j=32}>")

  def test_conversion_vector_scalar_add_2D_instance(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value(())
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),)
    rule = str_to_sdy_sharding_rule("...,  -> ...")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type, result.operands[1].type],
        [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j], [])->([i, j]) {i=16, j=32}>")

  def test_conversion_reshape_rule(self):
    opnd0 = self.create_tensor_value((2, 4))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((8,))],
        operands=[opnd0,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("i j -> (i j)")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type],
        [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>")

  def test_conversion_contracting_dim_matmul(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((32, 8))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 8))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("... contracting_dim, contracting_dim k -> ... k")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type, result.operands[1].type],
        [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=16, j=32, k=8}>")


  def test_conversion_multiple_batching_groups(self):
    opnd0 = self.create_tensor_value((4, 5, 16, 32))
    opnd1 = self.create_tensor_value((6, 7, 8, 32, 16))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((4, 5, 32, 16))],
        operands=[opnd0, opnd1,],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")))
    rule = str_to_sdy_sharding_rule("... j i, ...1 i j -> ...i j")
    mlir_rule = sdy_sharding_rule_to_mlir(rule,
        [result.operands[0].type, result.operands[1].type],
        [result.result.type,])
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j, k, l], [m, n, o, l, k])->([i, j, l, k]) {i=4, j=5, k=16, l=32, m=6, n=7, o=8}>")


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
