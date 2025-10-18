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
"""Generates source maps for JAX functions."""
import os
import tempfile
from typing import Protocol
from collections.abc import Sequence

from jax.experimental.source_mapper import common


class SourceMapGeneratorFn(Protocol):
  def __call__(self, *args, **kwargs) -> Sequence[common.SourceMapDump]:
    ...


def generate_sourcemaps(
    f,
    passes: Sequence[common.Pass],
    **pass_kwargs
) -> SourceMapGeneratorFn:
  """Generates a SourceMapBundle for the specified compiler passes.

  Args:
    f: The function to compile.
    passes: Which compiler passes to generate sourcemaps for.
    **pass_kwargs: Keyword arguments for individual passes.
  """
  def wrapper(*args, **kwargs) -> Sequence[common.SourceMapDump]:
    pass_results: list[common.SourceMapDump] = []
    compile_cache = {}
    with tempfile.TemporaryDirectory() as work_dir:
      for pass_to_eval in passes:
        if pass_to_eval.compile_fn not in compile_cache:
          dirname = pass_to_eval.name.replace(":", "__")
          pass_work_dir = os.path.join(work_dir, dirname)
          os.makedirs(pass_work_dir, exist_ok=False)
          compile_result = pass_to_eval.compile_fn(
              pass_work_dir, f, args, kwargs, **pass_kwargs
          )
          compile_cache[pass_to_eval.compile_fn] = compile_result
        compile_result = compile_cache[pass_to_eval.compile_fn]
        pass_results.append(pass_to_eval.generate_dump(compile_result,
                                                       **pass_kwargs))
    return pass_results
  return wrapper
