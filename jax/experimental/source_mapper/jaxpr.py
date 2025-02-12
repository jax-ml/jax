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
"""Source mapping generator for Jaxprs."""
import re
from typing import Any

import jax
from jax._src import config
from jax._src import core
from jax._src import source_info_util
from jax._src import sourcemap
from jax.experimental.source_mapper import common

source_info_util.register_exclusion(__file__)


def compile_jaxpr(work_dir, f, f_args, f_kwargs):
  del work_dir
  return jax.make_jaxpr(f)(*f_args, **f_kwargs)


def canonicalize_filename(file_name: str):
  pattern = config.hlo_source_file_canonicalization_regex.value
  if pattern:
    file_name = re.sub(pattern, '', file_name)
  return file_name


def make_jaxpr_dump(jaxpr: core.Jaxpr, **_) -> common.SourceMapDump:
  pprint_mappings: list[list[tuple[int, int, Any]]] = []
  pprint_str = jaxpr.pretty_print(source_map=pprint_mappings)
  used_source_files = []
  mappings = sourcemap.MappingsGenerator()
  for pprint_map_line in pprint_mappings:
    mappings.new_group()
    for pprint_segment in pprint_map_line:
      start_col, end_col, frame = pprint_segment
      del end_col
      file_name = canonicalize_filename(frame.file_name)
      if file_name not in used_source_files:
        used_source_files.append(file_name)
      file_idx = used_source_files.index(file_name)
      src_line = frame.start_line - 1  # Zero-indexed
      src_col = frame.start_column
      # A segment is a tuple of the form:
      # (generated_col, src_file_idx, src_line, src_col)
      mappings.new_segment(start_col, file_idx, src_line, src_col)
  mappings.new_group()
  source_map = sourcemap.SourceMap(
      version=3,
      sources=used_source_files,
      sources_content=[],
      mappings=mappings.mappings(),
      names=[],
  )
  return common.SourceMapDump(
      source_map=source_map,
      generated_code=pprint_str,
      pass_name='jaxpr',
  )


common.register_pass(
    common.Pass(
        name='jaxpr',
        compile_fn=compile_jaxpr,
        generate_dump=make_jaxpr_dump,  # type: ignore[arg-type]
    )
)
