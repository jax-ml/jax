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
from jax.experimental.custom_partitioning import SdyShardingRule
from mlir.dialects import stablehlo


class SdyShardingRuleTest(jtu.JaxTestCase):

  def test_rule_is_not_a_str(self):
    with self.assertRaisesRegex(TypeError, "rule must be a str"):
      SdyShardingRule(1)

  def test_factor_sizes_is_not_a_proper_dict(self):
    with self.assertRaisesRegex(
        TypeError, "factor_sizes must be a dict of str to int"
    ):
      SdyShardingRule("i->j", i="j")

  def test_sharding_rule_ellipsis_not_complete(self):
    with self.assertRaisesRegex(
        ValueError, "Character '.' must be used inside ellipsis '...'"
    ):
      SdyShardingRule(".i -> j")

  def test_sharding_rule_missing_results(self):
    with self.assertRaisesRegex(ValueError, "There is no -> in rule"):
      SdyShardingRule("i")

  def test_sharding_rule_inbalenced_brackets(self):
    with self.assertRaisesRegex(ValueError, "Brackets are not balanced"):
      SdyShardingRule("i j, k)->j")

  def test_sharding_rule_inbalenced_brackets2(self):
    with self.assertRaisesRegex(ValueError, "Brackets are not balanced"):
      SdyShardingRule("i (j k->j")

  def test_sharding_rule_empty_compound_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Brackets should contain at least two factors"
    ):
      SdyShardingRule("i ( ) j k->j")

  def test_sharding_rule_one_factorcompound_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Brackets should contain at least two factors"
    ):
      SdyShardingRule("i (j ) k->j")

  def test_sharding_rule_nested_brackets(self):
    with self.assertRaisesRegex(
        ValueError, "Compound factors should be one level"
    ):
      SdyShardingRule("i (j (k))->j")

  def test_sharding_rule_unknown_char(self):
    with self.assertRaisesRegex(ValueError, "Unknown character"):
      SdyShardingRule("i; j->j")

  def test_sharding_rule_ellipsis_not_leading_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Ellipsis can only be used at the beginning of a dimension"
    ):
      SdyShardingRule("i ... -> j")

  def test_sharding_rule_ellipsis_inside_compound_dim(self):
    with self.assertRaisesRegex(
        ValueError, "Ellipsis can only be used at the beginning of a dimension"
    ):
      SdyShardingRule("i, (..., j) -> j")

  def test_sharding_rule_scalar_operand_scalar_result(self):
    rule = SdyShardingRule("->")
    self.assertEqual(str(rule), "SdyShardingRule(((),), ((),), {})")

  def test_sharding_rule_one_scalar_operand(self):
    rule = SdyShardingRule("i j, , k->j")
    self.assertEqual(
        str(rule), "SdyShardingRule((('i', 'j'), (), ('k',)), (('j',),), {})"
    )

  def test_sharding_rule_factor_size_not_used(self):
    with self.assertRaisesRegex(ValueError, "Factor k is not used"):
      SdyShardingRule("i->j", k=10)

  def test_sharding_rule_factor_size_not_necessary(self):
    with self.assertRaisesRegex(
        ValueError,
        "Factor i represents a whole dimension; do not specify its size",
    ):
      SdyShardingRule("i->j", i=10)

  def test_sharding_rule_compound_factor_size_not_necessary(self):
    with self.assertRaisesRegex(
        ValueError,
        "Factor i represents a whole dimension; do not specify its size",
    ):
      SdyShardingRule("(i j) -> i", i=10, j=20)

  def test_sharding_rule_factor_sizes_missing(self):
    with self.assertRaisesRegex(
        ValueError,
        "Factor k is only used in compound factors; must specify its size",
    ):
      SdyShardingRule("i j -> (j k)")

  def test_sharding_rule_factor_elementwise_add(self):
    rule = SdyShardingRule("... i j, ...i j -> ...i j")
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('…', 'i', 'j'), ('…', 'i', 'j')), (('…', 'i',"
        " 'j'),), {})",
    )

  def test_sharding_rule_factor_vector_scalar_add(self):
    rule = SdyShardingRule("...i,  -> ...i")
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('…', 'i'), ()), (('…', 'i'),), {})",
    )

  def test_sharding_rule_factor_reshape_combining(self):
    rule = SdyShardingRule("i j -> (i j)")
    self.assertEqual(
        str(rule), "SdyShardingRule((('i', 'j'),), ((('i', 'j'),),), {})"
    )

  def test_sharding_rule_factor_reshape_reordering(self):
    rule = SdyShardingRule("(j i) -> (i j)", i=10, j=20)
    self.assertEqual(
        str(rule),
        "SdyShardingRule(((('j', 'i'),),), ((('i', 'j'),),), {'i': 10, 'j':"
        " 20})",
    )

  def test_sharding_rule_factor_compound_then_individual(self):
    rule = SdyShardingRule("(i j) (j k) i -> j k")
    self.assertEqual(
        str(rule),
        "SdyShardingRule(((('i', 'j'), ('j', 'k'), 'i'),), (('j', 'k'),), {})",
    )

  def test_sharding_rule_factor_individual_then_compound(self):
    rule = SdyShardingRule("i j k -> (i j) (j k)")
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('i', 'j', 'k'),), ((('i', 'j'), ('j', 'k')),), {})",
    )

  def test_sharding_rule_factor_infer_k(self):
    rule = SdyShardingRule("i (j k)-> j foo (m n)", k=10, m=10, n=20)
    self.assertEqual(
        str(rule),
        "SdyShardingRule((('i', ('j', 'k')),), (('j', 'foo', ('m', 'n')),),"
        " {'k': 10, 'm': 10, 'n': 20})",
    )


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
        attributes=dict(call_target_name=ir.StringAttr.get("dummy_target")),
    ).result

  def test_conversion_rule_op_mismatch_in_operands_num(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j-> i j")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule has 1 operands, but the operation has 2 operands",
    ):
      rule.build(
          [result.operands[0].type, result.operands[1].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_rule_op_mismatch_in_operands_rank(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j, i j k-> i j")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule 1th operand has rank 3, but the operation 1th "
        "operand has rank 2",
    ):
      rule.build(
          [result.operands[0].type, result.operands[1].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_rule_op_mismatch_in_results_num(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j, i j -> i j, i j")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule has 2 results, but the operation has 1 results",
    ):
      rule.build(
          [result.operands[0].type, result.operands[1].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_rule_op_mismatch_in_results_dim(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j, i j -> i j k")
    with self.assertRaisesRegex(
        ValueError,
        "Sharding rule 0th result has rank 3, but the operation 0th "
        "result has rank 2",
    ):
      rule.build(
          [result.operands[0].type, result.operands[1].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_factor_has_two_sizes(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 64))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j, i j -> i j")
    with self.assertRaisesRegex(
        ValueError,
        "Factor j corresponds to two sizes: 32 and 64",
    ):
      rule.build(
          [result.operands[0].type, result.operands[1].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_compound_dimension_size_mismatch(self):
    opnd = self.create_tensor_value((2, 4))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((9,))],
        operands=[
            opnd,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j -> (i j)")
    with self.assertRaisesRegex(
        ValueError,
        "0th result actual size 9 doesn't match the size 8 derived from the"
        " compound factors",
    ):
      rule.build(
          [result.operands[0].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_elementwise_rule_mismatching_ellipsis_rank(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16,))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("..., ... -> ...")
    with self.assertRaisesRegex(
        ValueError,
        "Ellipsis represents different number of leading dimensions 2 and 1",
    ):
      rule.build(
          [result.operands[0].type, result.operands[1].type],
          [
              result.result.type,
          ],
      )

  def test_conversion_elementwise_rule_scalar_instance(self):
    opnd0 = self.create_tensor_value(())
    opnd1 = self.create_tensor_value(())
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type(())],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("..., ... -> ...")
    mlir_rule = rule.build(
        [result.operands[0].type, result.operands[1].type],
        [
            result.result.type,
        ],
    )
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([], [])->([])>",
    )

  def test_conversion_elementwise_rule_2D_instance(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value((16, 32))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("..., ... -> ...")
    mlir_rule = rule.build(
        [result.operands[0].type, result.operands[1].type],
        [
            result.result.type,
        ],
    )
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=16, j=32}>",
    )

  def test_conversion_vector_scalar_add_2D_instance(self):
    opnd0 = self.create_tensor_value((16, 32))
    opnd1 = self.create_tensor_value(())
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((16, 32))],
        operands=[
            opnd0,
            opnd1,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("...,  -> ...")
    mlir_rule = rule.build(
        [result.operands[0].type, result.operands[1].type],
        [
            result.result.type,
        ],
    )
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j], [])->([i, j]) {i=16, j=32}>",
    )

  def test_conversion_reshape_rule(self):
    opnd0 = self.create_tensor_value((2, 4))
    result = ir.Operation.create(
        "stablehlo.custom_call",
        results=[self.get_tensor_type((8,))],
        operands=[
            opnd0,
        ],
        attributes=dict(call_target_name=ir.StringAttr.get("foo")),
    )
    rule = SdyShardingRule("i j -> (i j)")
    mlir_rule = rule.build(
        [result.operands[0].type],
        [
            result.result.type,
        ],
    )
    self.assertEqual(
        str(mlir_rule),
        "#sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>",
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
