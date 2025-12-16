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
"""Test utilities for the custom pipeline scheduling API."""
import dataclasses
from typing import Any, Sequence

from jax._src import debugging
from jax._src.pallas.pipelining import schedulers
from jax._src.pallas.pipelining import internal


def print_stage(
    ctx: schedulers.PipelineContext, stage: internal.PipelineStage, *args
):
  """Evaluation function that prints the stage name and iteration number."""
  del args
  debugging.debug_print(
      "[itr={}] %s" % stage, ctx.linearized_index, ordered=True)


@dataclasses.dataclass(frozen=True)
class AnyOrder:
  """A helper class to mark the order of elements as unimportant."""
  elements: Sequence[Any]


def compare_lists(result, expected):
  """Returns if two lists are equal while respecting ``AnyOrder`` elements."""
  result_ptr = 0
  expected_ptr = 0
  any_order_set = None
  while result_ptr < len(result) and expected_ptr < len(expected):
    cur_result = result[result_ptr]
    cur_expected = expected[expected_ptr]
    if isinstance(cur_expected, AnyOrder):
      if any_order_set is None:
        any_order_set = set(cur_expected.elements)

      if cur_result in any_order_set:
        result_ptr += 1
        any_order_set.remove(cur_result)
      else:
        return False
      if not any_order_set:
        any_order_set = None
        expected_ptr += 1
    else:
      if cur_result == cur_expected:
        result_ptr += 1
        expected_ptr += 1
      else:
        return False
  return True
