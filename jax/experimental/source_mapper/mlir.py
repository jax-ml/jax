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
"""Utilities for generating source mappings for MLIR dialects."""
import collections
import re
from typing import cast

from jax._src import sourcemap


# TODO(justinfu): Make a proper parser for MLIR dumps.
LOC_REGEX = re.compile(r"loc\(#loc(?P<id>[0-9]+)\)")

SRC_REGEX = re.compile(
    r"#loc(?P<id>[0-9]+) ="
    r" loc\(\"(?P<file>.*)\":(?P<line>[0-9]+):(?P<col>[0-9]+)"
    r"( to (?P<endlineno>[0-9]+)?:(?P<endcolno>[0-9]+))?\)"
)

SCOPED_REGEX = re.compile(
    r"#loc(?P<id>[0-9]+) = loc\(\"(?P<scope>.*)\"\(#loc(?P<tgt_id>[0-9]+)\)\)"
)

CALLSITE_REGEX = re.compile(
    r"#loc(?P<id>[0-9]+) = loc\(callsite\(#loc(?P<callee>[0-9]+) at"
    r" #loc(?P<caller>[0-9]+)\)\)"
)

Location = collections.namedtuple("Location", ["file", "line", "col"])
Redirect = collections.namedtuple("Redirect", ["tgt_id"])


def create_mlir_sourcemap(mlir_dump: str) -> sourcemap.SourceMap:
  mappings = sourcemap.MappingsGenerator()
  dump_lines: list[str] = mlir_dump.split("\n")

  segment_dict, sources = parse_mlir_locations(dump_lines)
  used_sources = []
  used_sources_filenames = []
  for line in dump_lines:
    mappings.new_group()
    match = LOC_REGEX.search(line)
    if match:
      loc_id = int(match.group("id"))
      if loc_id not in segment_dict:
        # TODO(justinfu): This happens on fusion locations - need to implement.
        continue
      segment = list(segment_dict[loc_id])
      first_col = line.index(line.strip()[0])
      segment[0] = first_col
      # Remap the sourcefile index to only sourcefiles that are used.
      # This is optional but makes the mapping file smaller by pruning
      # unused sourcefiles.
      source_idx = segment[1]
      if source_idx not in used_sources:
        used_sources.append(source_idx)
        used_sources_filenames.append(sources[source_idx])
      segment[1] = used_sources.index(source_idx)
      mappings.new_segment(*segment)
  mappings.new_group()

  return sourcemap.SourceMap(
          version=3,
          sources=used_sources_filenames,
          sources_content=[''] * len(used_sources_filenames),
          mappings=mappings.mappings(),
          names=[],
      )


def parse_mlir_locations(
    mlir_dump: list[str],
) -> tuple[dict[int, sourcemap.Segment], list[str]]:
  locations: dict[int, Location | Redirect] = {}
  source_files = []
  for line in mlir_dump:
    if line.startswith("#loc"):
      src_match = SRC_REGEX.match(line)
      if src_match:
        match_dict = src_match.groupdict()
        filename = match_dict["file"]
        locations[int(match_dict["id"])] = Location(
            file=filename,
            line=int(match_dict["line"]),
            col=int(match_dict["col"]),
        )
        if filename not in source_files:
          source_files.append(filename)
        continue
      scoped_match = SCOPED_REGEX.match(line)
      if scoped_match:
        match_dict = scoped_match.groupdict()
        locations[int(match_dict["id"])] = Redirect(
            tgt_id=int(match_dict["tgt_id"])
        )
        continue
      callsite_match = CALLSITE_REGEX.match(line)
      if callsite_match:
        match_dict = callsite_match.groupdict()
        locations[int(match_dict["id"])] = Redirect(
            tgt_id=int(match_dict["callee"])
        )
        continue
      if "loc(unknown)" in line:
        continue
  # Resolve redirects
  while True:
    new_locations: dict[int, Location | Redirect] = {}
    updated = False
    for loc_id, loc in locations.items():
      if isinstance(loc, Redirect):
        new_locations[loc_id] = locations[loc.tgt_id]
        updated = True
      else:
        new_locations[loc_id] = loc
    locations = new_locations
    if not updated:
      break
  segment_dict: dict[int, sourcemap.Segment] = {}
  for id_, loc in locations.items():
    # A segment is a tuple of the form:
    # (generated_col, src_file_idx, src_line, src_col)
    loc = cast(Location, loc)
    segment_dict[id_] = (
        0,
        source_files.index(loc.file),
        loc.line - 1,  # Zero-indexed, so offset by 1.
        loc.col,
    )
  return segment_dict, source_files
