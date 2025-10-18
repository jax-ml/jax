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

import re
import ast
from pathlib import Path
from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx.util.logging import getLogger

logger = getLogger(__name__)


# (The parse_lines_spec and get_tagged_block functions are unchanged)
def parse_lines_spec(spec: str) -> list[int]:
  items = []
  if not spec:
    return items
  for part in spec.split(","):
    part = part.strip()
    if "-" in part:
      start, end = part.split("-", 1)
      items.extend(range(int(start), int(end) + 1))
    else:
      items.append(int(part))
  return items


def get_tagged_block(filepath, tag, lines_spec=None):
  try:
    full_path = Path(filepath)
    if not full_path.exists():
      raise FileNotFoundError(f"Source file not found at {full_path}")
    content_full = full_path.read_text()
    regex_pattern = rf"# tag: {tag}\n(.*?)\s*# tag: {tag}"
    pattern = re.compile(regex_pattern, re.DOTALL)
    match = pattern.search(content_full)
    if not match:
      raise ValueError(f"Tag '{tag}' not found in '{filepath}'")
    content = match.group(1).strip("\n")
    if lines_spec is None:
      return content
    line_list = content.split("\n")
    if lines_spec.startswith("[") and lines_spec.endswith("]"):
      indexer = ast.literal_eval(lines_spec)
      final_lines = [line_list[i] for i in indexer]
    elif ":" in lines_spec:
      parts_str = (lines_spec.split(":") + ["", "", ""])[:3]
      indexer = slice(*(int(p.strip()) if p.strip() else None for p in parts_str))
      final_lines = line_list[indexer]
    else:
      indexer = int(lines_spec)
      final_lines = [line_list[indexer]]
    if not final_lines:
      return ""
    try:
      indent_level = len(final_lines[0]) - len(final_lines[0].lstrip())
      return "\n".join(line[indent_level:] for line in final_lines)
    except IndexError:
      return ""
  except Exception as e:
    logger.warning(f"Error processing tagged_block: {e}")
    return f"Error processing tagged_block for tag '{tag}' in '{filepath}'."


class TaggedBlockDirective(SphinxDirective):
  has_content = False
  required_arguments = 2
  optional_arguments = 1
  option_spec = {
    "hl_lines": str,
  }

  def run(self):
    source_dir = Path(self.env.srcdir)
    filepath = source_dir / self.arguments[0]
    tag = self.arguments[1]
    lines_spec = self.arguments[2] if len(self.arguments) > 2 else None
    code = get_tagged_block(filepath, tag, lines_spec)
    literal = nodes.literal_block(code, code)
    literal["language"] = "python"
    if "hl_lines" in self.options:
      highlight_lines = parse_lines_spec(self.options["hl_lines"])
      literal["highlight_args"] = {"hl_lines": highlight_lines}
    return [literal]


def setup(app):
  app.add_directive("tagged-block", TaggedBlockDirective)
  # This dictionary fixes the "parallel reading" warning
  return {
    "version": "0.1",
    "parallel_read_safe": True,
    "parallel_write_safe": True,
  }
