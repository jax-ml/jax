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
"""Source mapping generator for HLO dialects."""
import enum
import re
from typing import Any

import jax
from jax._src import sourcemap

from jax.experimental.source_mapper import common
from jax.experimental.source_mapper import mlir


class HloPass(enum.Enum):
  STABLE_HLO = "hlo:stable-hlo"
  ORIGINAL = "hlo:original"
  OPTIMIZED = "hlo:optimized"


METADATA_REGEX = re.compile(
    r"metadata={.*op_name=\"(?P<scope>.*)\""
    r" source_file=\"(?P<src_file>.*)\""
    r" source_line=(?P<src_line>[0-9]+).*?}"
)


def parse_hlo_dump(text: str) -> sourcemap.SourceMap:
  mappings = sourcemap.MappingsGenerator()
  used_source_files = []
  for line in text.split("\n"):
    mappings.new_group()
    match = METADATA_REGEX.search(line)
    if match:
      match_dict = match.groupdict()
      _ = match_dict["scope"]  # Unused
      src_file = match_dict["src_file"]
      src_line = int(match_dict["src_line"])
      if src_file not in used_source_files:
        used_source_files.append(src_file)
      src_file_idx = used_source_files.index(src_file)
      src_line -= 1  # Segments are zero-indexed
      first_col = line.index(line.strip()[0])
      mappings.new_segment(first_col, src_file_idx, src_line, 0)
  mappings.new_group()

  return sourcemap.SourceMap(
      version=3,
      sources=used_source_files,
      sources_content=[],
      mappings=mappings.mappings(),
      names=[],
  )


def trace_and_lower(work_dir, f, f_args, f_kwargs, **_):
  lowered = jax.jit(lambda *args: f(*args, **f_kwargs)).lower(*f_args)
  return (lowered, work_dir)


def stable_hlo_generate_dump(args: tuple[Any, str],
                             **_) -> common.SourceMapDump:
  lowered, work_dir = args
  del work_dir
  hlo_text = lowered.as_text(debug_info=True)
  source_map = mlir.create_mlir_sourcemap(hlo_text)
  return common.SourceMapDump(
      source_map=source_map,
      generated_code=hlo_text,
      pass_name=HloPass.STABLE_HLO.value,
  )


common.register_pass(
    common.Pass(
        name=HloPass.STABLE_HLO.value,
        compile_fn=trace_and_lower,  # type: ignore[arg-type]
        generate_dump=stable_hlo_generate_dump,  # type: ignore[arg-type]
    )
)


def original_hlo_generate_dump(args: tuple[Any, str],
                               **_) -> common.SourceMapDump:
  lowered, work_dir = args
  del work_dir
  hlo_text = lowered.as_text(dialect="hlo", debug_info=True)
  source_map = parse_hlo_dump(hlo_text)
  return common.SourceMapDump(
      source_map=source_map,
      generated_code=hlo_text,
      pass_name=HloPass.ORIGINAL.value,
  )


common.register_pass(
    common.Pass(
        name=HloPass.ORIGINAL.value,
        compile_fn=trace_and_lower,  # type: ignore[arg-type]
        generate_dump=original_hlo_generate_dump,  # type: ignore[arg-type]
    )
)


def optimized_generate_dump(args: tuple[Any, str],
                            xla_compiler_flags: dict[str, Any] | None = None,
                            **_) -> common.SourceMapDump:
  lowered, work_dir = args
  compilation_args = {"xla_dump_to": work_dir, **(xla_compiler_flags or {})}
  hlo_text = lowered.compile(compilation_args).as_text()
  source_map = parse_hlo_dump(hlo_text)
  return common.SourceMapDump(
      source_map=source_map,
      generated_code=hlo_text,
      pass_name=HloPass.OPTIMIZED.value,
  )


common.register_pass(
    common.Pass(
        name=HloPass.OPTIMIZED.value,
        compile_fn=trace_and_lower,  # type: ignore[arg-type]
        generate_dump=optimized_generate_dump,  # type: ignore[arg-type]
    )
)
