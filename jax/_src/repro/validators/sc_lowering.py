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

import contextlib
from unittest import mock

from jax._src.pallas.mosaic import tpu_info
from jax._src import pjit
from jax._src.pjit import _resolve_and_lower as original_resolve_and_lower

def _resolve_and_lower_for_tpu(
    args, jaxpr, in_shardings, out_shardings, in_layouts,
    out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
    lowering_platforms, lowering_parameters, pgle_profiler,
    compiler_options_kvs):
  new_lowering_platforms = ("tpu",)
  return original_resolve_and_lower(
      args, jaxpr, in_shardings, out_shardings, in_layouts,
      out_layouts, donated_invars, ctx_mesh, name, keep_unused, inline,
      new_lowering_platforms, lowering_parameters, pgle_profiler,
      compiler_options_kvs)

def mock_tpu_context(
    chip_version: tpu_info.ChipVersion = tpu_info.ChipVersion.TPU_V6E,
    num_cores: int = 1,
) -> contextlib.AbstractContextManager:
  fake_info = tpu_info._get_tpu_info_impl(chip_version, num_cores)
  mock_stack = contextlib.ExitStack()
  mock_stack.enter_context(mock.patch(
      "jax._src.pallas.mosaic.tpu_info.get_tpu_info",
      return_value=fake_info))
  mock_stack.enter_context(mock.patch(
      "jax._src.pallas.mosaic.tpu_info.is_tpu_device",
      return_value=True))
  return mock_stack

def validate_repro(repro_fun) -> str:
  with mock_tpu_context(tpu_info.ChipVersion.TPU_V5P):
    with mock.patch.object(pjit, '_resolve_and_lower',
                           _resolve_and_lower_for_tpu):
      try:
        repro_fun()
      except Exception as e:
        if "Swap only supports bfloat16 arrays of shapes" in str(e):
          return ""
        return str(e)
    return "No error"
