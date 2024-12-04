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

"""Implements SdyShardingRule."""

from collections import OrderedDict

from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import sdy


_CompoundFactor = tuple[str, ...]
_DimMapping = tuple[str | _CompoundFactor, ...]

# A single character replacement for ... to simplify parsing.
_ELLIPSIS: str = "…"

_BATCHING_DIM_FACTOR_PREFIX = "?"
def _get_batching_dim_factor_name(batch_dim_order):
  """ Constructs a factor name for a batching dimension.

  We expand the leading ... into factors representing the batching dimensions
  to support building the MLIR representation for the sharding rule. For this
  reason, we construct a factor name that won't be used by users for the
  batching dimensions.
  """
  return f"?{batch_dim_order}"

def _parse_operands(
    rule: str,
) -> tuple[_DimMapping, ...]:
  """Parses the LHS or RHS of an Einsum notation like string.

  Converts each operand or result in the Einsum notation like string to a tuple
  of _DimMapping.

  Args:
    rule: The Einsum notation for the operands or results of an operation.

  Returns:
    The tuple of operands.

  Raises:
    ValueError: If the rule is not balanced or contains unknown characters.
  """

  # Remove unnecessary spaces in the rule to simplify the parsing process.
  words = rule.split()
  rule = " ".join(words)

  # Similar to einops rules, an empty LHS/RHS has a single scalar
  # operand/result.
  if not rule:
    return ((),)

  all_operands = []
  # Represent all dimensions of an operand. When an operand[0]==_ELLIPSIS, the
  # operand may have 0 or more leading dimensions.
  operand = []
  current_factor = None
  # A value of None indicates the current dimension is not a compound dimension,
  # while a value of [] indicates that we have just started parsing a compound
  # dimension.
  current_compound_dim = None

  def add_factor(x):
    if current_compound_dim is None:
      operand.append(x)
    else:
      current_compound_dim.append(x)

  for char in rule:
    if char == _ELLIPSIS:
      if (
          current_factor is not None
          or current_compound_dim is not None
          or operand
      ):
        raise ValueError(
            "Ellipsis can only be used at the beginning of a dimension"
        )
      add_factor(_ELLIPSIS)
      continue
    if char in "(), ":
      if current_factor is not None:
        add_factor(current_factor)
        current_factor = None
      if char == "(":
        if current_compound_dim is not None:
          raise ValueError(
              "Compound factors should be one level, nested brackets are not"
              " allowed"
          )
        current_compound_dim = []
      elif char == ")":
        if current_compound_dim is None:
          raise ValueError("Brackets are not balanced")
        if len(current_compound_dim) <= 1:
          raise ValueError("Brackets should contain at least two factors")
        operand.append(tuple(current_compound_dim))
        current_compound_dim = None
      elif char == ",":
        all_operands.append(tuple(operand))
        operand = []
    elif str.isalnum(char) or char in ["_"]:
      if current_factor is None:
        current_factor = char
      else:
        current_factor += char
    else:
      raise ValueError(f"Unknown character '{char}'")

  if current_compound_dim is not None:
    raise ValueError(f"Brackets are not balanced in rule: '{rule}'")
  if current_factor is not None:
    add_factor(current_factor)
  all_operands.append(tuple(operand))

  return tuple(all_operands)


class SdyShardingRule:
  """A representation for Shardy sharding rule.

  A SdyShardingRule includes an Enisum notation like string and an optional
  list of factor sizes. A factor is a name in the Einsum notation. If a factor
  is only used in compound factors, its size must be specified.

  SdyShardingRule examples:

  * Contracting dim matmul AB@BC->AC: SdyShardingRule('i j, j k-> i k')
  * A reshape (8,) -> (4, 2): SdyShardingRule('(i j) -> i j')
  * Another reshape (4, 2) -> (2, 4): SdyShardingRule('(i j)->(j i)`, i=4, j=5)
  * An elementwise add of any dimensions x + y -> z: SdyShardingRule('..., ...-> ...')
  """

  def __init__(self, rule: str, **factor_sizes):
    """Constructs a SdyShardingRule object from the Einsum notation like string.

    This is done by verifying that the input Einsum notation like string and
    with optional factor sizes represents a valid sharding rule and converting
    it to an internal representation.

    Args:
      rule: The Einsum notation like string for an operation.
      **factor_sizes: The optional factor sizes.

    Raises:
      ValueError: If there is any problem with the rule or factor_sizes.
    """
    if not isinstance(rule, str):
      raise TypeError(f"rule must be a str, but got {type(rule)}")
    if not all(isinstance(size, int) for size in factor_sizes.values()):
      raise TypeError(
          f"factor_sizes must be a dict of str to int, but got {factor_sizes}"
      )

    # Replace ... with a single char to simplify parsing.
    if _ELLIPSIS in rule:
      raise ValueError(f"Unknown character '{_ELLIPSIS}'")
    if "." in rule:
      rule = rule.replace("...", _ELLIPSIS)
      if "." in rule:
        raise ValueError(f"Character '.' must be used inside ellipsis '...'")

    try:
      operands, results = rule.split("->")
    except ValueError as e:
      raise ValueError(f"There is no -> in rule: '{rule}'") from e

    self.operands = _parse_operands(operands)
    self.results = _parse_operands(results)

    # Find all factors and mark whether their size can be inferred.
    factors_inferrable = dict()
    for operand in self.operands + self.results:
      for dim in operand:
        if dim == _ELLIPSIS:
          continue
        if isinstance(dim, str):
          factors_inferrable[dim] = True
        else:
          assert isinstance(dim, tuple)
          for factor in dim:
            assert isinstance(factor, str)
            if factor not in factors_inferrable.keys():
              factors_inferrable[factor] = False

    # Check that factors in factor_sizes are used in the rule.
    for factor in factor_sizes:
      if factor not in factors_inferrable:
        raise ValueError(
            f"Factor {factor} is not used in the rule, but size is provided"
        )

    # Check that factors that are used for a whole dimension aren't in
    # factor_sizes and factors that are never used for a whole dimension are
    # in factor_sizes.
    for factor, inferrable in factors_inferrable.items():
      if factor not in factor_sizes and not inferrable:
        raise ValueError(
            f"Factor {factor} is only used in compound factors; must specify"
            " its size"
        )
      if factor in factor_sizes and inferrable:
        raise ValueError(
            f"Factor {factor} represents a whole dimension; do not specify its"
            " size"
        )

    self.factor_sizes = factor_sizes

  def __str__(self):
    return (
        f"SdyShardingRule({self.operands}, {self.results}, {self.factor_sizes})"
    )

  def build(
      self,
      operand_types: list[ir.Type],
      result_types: list[ir.Type],
  ) -> ir.Attribute:
    """Builds the MLIR representation for the sharding rule.

    This is done by verifying that the rule is consistent with the types of
    the operation and converting the Einsum notation like string to
    OpShardingRuleAttr.
    """
    if len(self.operands) != len(operand_types):
      raise ValueError(
          f"Sharding rule has {len(self.operands)} operands, but the operation"
          f" has {len(operand_types)} operands"
      )
    if len(self.results) != len(result_types):
      raise ValueError(
          f"Sharding rule has {len(self.results)} results, but the operation"
          f" has {len(result_types)} results"
      )

    factors_to_indices_sizes = OrderedDict()
    types = operand_types + result_types
    UNKNOWN = -1  # Representation for unknown factor size or factor index.

    def get_message_for_operand(i):
      if i >= len(operand_types):
        return f"{i - len(operand_types)}th result"
      else:
        return f"{i}th operand"

    def get_rank_for_operand(i):
      return ir.ShapedType(types[i]).rank

    def get_size_for_operand_dim(i, j):
      return ir.ShapedType(types[i]).shape[j]

    def add_factor(factor, size):
      """Adds a factor to factors_to_indices_sizes.

      'size' may be a dimensions size, a user specified factor size, or UNKNOWN
      if a factor is first used as in a compound factor and then used for a
      whole dimension.
      """
      item = factors_to_indices_sizes.get(factor, [UNKNOWN, UNKNOWN])
      factor_index = item[0]
      factor_size = item[1]
      if factor_index != UNKNOWN:
        # Not the first time seeing the factor.
        if size != UNKNOWN and factor_size != UNKNOWN and factor_size != size:
          factor_or_batching_dim = (
              f"Factor {factor}"
              if _BATCHING_DIM_FACTOR_PREFIX not in factor
              else f"Batching dimension {factor[1:]}"
          )
          raise ValueError(
              f"{factor_or_batching_dim} corresponds to two sizes:"
              f" {factor_size} and {size}"
          )
        if size != UNKNOWN and factor_size == UNKNOWN:
          factors_to_indices_sizes[factor] = [factor_index, size]
      else:
        # First time seeing the factor.
        factor_index = len(factors_to_indices_sizes)
        factors_to_indices_sizes[factor] = [factor_index, size]

    def add_batching_dim_factor(batch_dim_order, factor_size):
      ellipsis_batch_dim_name = f"?{batch_dim_order}"
      add_factor(ellipsis_batch_dim_name, factor_size)

    def build_dim_mapping_for_compound_factors(i, j, factors):
      accumulated_size = 1
      all_indices = []
      for factor in factors:
        item = factors_to_indices_sizes[factor]
        factor_index = item[0]
        factor_size = item[1]
        accumulated_size *= factor_size
        all_indices.append(factor_index)

      dim_size = get_size_for_operand_dim(i, j)
      if accumulated_size != dim_size:
        raise ValueError(
            f"{get_message_for_operand(i)} actual size {dim_size} doesn't match"
            f" the size {accumulated_size} derived from the compound factors"
            f" {factors}"
        )

      return sdy.DimMappingAttr.get(factor_indices=all_indices)

    # Add factors and their sizes in the order they appear in the rule,
    # including the batching dimensions represented by ellipsis.
    ellipsis_rank = None
    for i, operand in enumerate(self.operands + self.results):
      if operand and operand[0] == _ELLIPSIS:
        has_ellipsis = True
        operand = operand[1:]
      else:
        has_ellipsis = False
      rule_rank = len(operand)
      op_rank = get_rank_for_operand(i)
      # The number of dimensions represented by ellipsis.
      current_ellipsis_rank = 0
      if has_ellipsis and op_rank > rule_rank:
        current_ellipsis_rank = op_rank - rule_rank
      if has_ellipsis:
        if ellipsis_rank is None:
          ellipsis_rank = current_ellipsis_rank
        elif ellipsis_rank != current_ellipsis_rank:
          raise ValueError(
              "Ellipsis represents different number of leading dimensions"
              f" {ellipsis_rank} and {current_ellipsis_rank}"
          )
      rule_rank += current_ellipsis_rank
      if rule_rank != op_rank:
        msg = get_message_for_operand(i)
        raise ValueError(
            f"Sharding rule {msg} has rank {rule_rank}, but the operation"
            f" {msg} has rank {op_rank}"
        )

      for j in range(current_ellipsis_rank):
        add_batching_dim_factor(j, get_size_for_operand_dim(i, j))

      for j, dim in enumerate(operand):
        if isinstance(dim, str):
          add_factor(
              dim, get_size_for_operand_dim(i, j + current_ellipsis_rank)
          )
        else:
          assert isinstance(dim, tuple)
          for factor in dim:
            add_factor(factor, self.factor_sizes.get(factor, UNKNOWN))

    # Build the tensor mappings for each operand and result.
    tensor_mappings = []
    for i, operand in enumerate(self.operands + self.results):
      dim_mappings = []

      if operand and operand[0] == _ELLIPSIS:
        operand = operand[1:]
        current_ellipsis_rank = ellipsis_rank
      else:
        current_ellipsis_rank = 0

      for j in range(current_ellipsis_rank):
        dim_mappings.append(
            sdy.DimMappingAttr.get(
                factor_indices=[
                    factors_to_indices_sizes[_get_batching_dim_factor_name(j)][0]
                ]
            )
        )

      for j, dim in enumerate(operand):
        if isinstance(dim, str):
          dim_mappings.append(
              sdy.DimMappingAttr.get(
                  factor_indices=[factors_to_indices_sizes[dim][0]]
              )
          )
        else:
          assert isinstance(dim, tuple)
          dim_mappings.append(
              build_dim_mapping_for_compound_factors(
                  i, j + current_ellipsis_rank, dim
              )
          )

      tensor_mappings.append(
          sdy.TensorMappingAttr.get(dim_mappings=dim_mappings)
      )

    op_sharding_rule = sdy.OpShardingRuleAttr.get(
        factor_sizes=[item[1] for item in factors_to_indices_sizes.values()],
        operand_mappings=tensor_mappings[0 : len(operand_types)],
        result_mappings=tensor_mappings[len(operand_types) :],
    )
    return op_sharding_rule
