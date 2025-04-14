# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from operator import itemgetter
from typing import Any, List

from docutils import nodes
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

# Please add justification for why the option is to be hidden and/or when
# it should be revealed, e.g. when the option is deprecated or when it is
# no longer experimental.
_hidden_config_options = (
  'jax_default_dtype_bits', # an experiment that we never documented, but we can't remove it because Keras depends on its existing broken behavior
  'jax_serialization_version',
  'check_rep', # internal implementation detail of shard_map, DO NOT USE‰
)

def config_option_to_title_case(name: str) -> str:
  """Converts a config option name to title case, with special rules.

  Args:
      name: The configuration option name (e.g., "jax_default_dtype_bits").
      capitalization_rules: An optional function that takes the name as input
          and returns the title-cased name. If None, defaults to a basic
          title-casing.
  """

  # Define capitalization rules as a list of (string, replacement) tuples
  capitalization_rules_list = [
      ("jax", "JAX"), ("xla", "XLA"), ("pgle", "PGLE"), ("cuda", "CUDA"), ("vjp", "VJP"), ("jvp", "JVP"),
      ("pjrt", "PjRT"), ("gpu", "GPU"), ("tpu", "TPU"), ("prng", "PRNG"), ("rocm", "ROCm"), ("spmd", "SPMD"),
      ("bcoo", "BCOO"), ("jit", "JIT"), ("cpu", "CPU"), ("cusparse", "cuSPARSE"), ("ir", "IR"), ("dtype", "DType"),
      ("pprint", "PPrint"), ("x64", "x64")
  ]
  name = name.replace("jax_", "").replace("_", " ").title()
  for find, replace in capitalization_rules_list:
      name = re.sub(rf"\b{find}\b", replace, name, flags=re.IGNORECASE)
  return name

def create_field_item(label, content):
  """Create a field list item with a label and content side by side.

  Args:
    label: The label text for the field name
    content: The content to add (a node or text)

  Returns:
    A field list item with the label and content side by side.
  """
  # Create a field list item
  field = nodes.field()

  # Create the field name (label)
  field_name = nodes.field_name()
  field_name += nodes.Text(label)
  field += field_name

  # Create the field body (content)
  field_body = nodes.field_body()

  if isinstance(content, str):
    para = nodes.paragraph()
    para += nodes.Text(content)
    field_body += para
  elif isinstance(content, nodes.Node):
    field_body += content

  field += field_body
  return field

class ConfigOptionDirective(SphinxDirective):
  required_arguments = 0
  optional_arguments = 0
  has_content = False

  def run(self) -> List[nodes.Node]:
    from jax._src.config import config as jax_config

    config_options = sorted(jax_config.meta.items(), key=itemgetter(0))
    result = []

    for name, (opt_type, meta_args, meta_kwargs) in config_options:
      if name in _hidden_config_options:
        continue

      holder = jax_config._value_holders[name]

      # Create target for linking
      target = nodes.target()
      target['ids'].append(name)
      result.append(target)

      # Create a section for this option
      option_section = nodes.section()
      option_section['ids'].append(name)
      option_section['classes'].append('config-option-section')

      # Create a title with the option name (important for TOC)
      title = nodes.title()
      title['classes'] = ['h4']
      title += nodes.Text(config_option_to_title_case(name))
      option_section += title

      # Create a field list for side-by-side display
      field_list = nodes.field_list()
      field_list['classes'].append('config-field-list')

      # Add type information as a field item
      if opt_type == "enum":
        type_para = nodes.paragraph()
        emphasis_node = nodes.emphasis()
        emphasis_node += nodes.Text("Enum values: ")
        type_para += emphasis_node

        for i, value in enumerate(enum_values := meta_kwargs.get('enum_values', [])):
          type_para += nodes.literal(text=repr(value))
          if i < len(enum_values) - 1:
            type_para += nodes.Text(", ")
      else:
        type_para = nodes.paragraph()
        type_para += nodes.literal(text=opt_type.__name__)

      field_list += create_field_item("Type", type_para)

      # Add default value information
      default_para = nodes.paragraph()
      default_para += nodes.literal(text=repr(holder.value))
      field_list += create_field_item("Default Value", default_para)

      # Add configuration string information
      string_para = nodes.paragraph()
      string_para += nodes.literal(text=repr(name))
      field_list += create_field_item("Configuration String", string_para)

      string_para = nodes.paragraph()
      string_para += nodes.literal(text=name.upper())
      field_list += create_field_item("Environment Variable", string_para)

      # Add the field list to the section
      option_section += field_list

      # Add help text in a description box
      if (help_text := meta_kwargs.get('help')):
        help_para = nodes.paragraph()
        # logger.error(name)
        # logger.warning(help_text)

        # If we get here, help text seems valid - proceed with normal parsing
        # parsed = nodes.Text(help_text)
        help_para += self.parse_text_to_nodes(help_text)

        option_section += help_para

      result.append(option_section)
      # Add an extra paragraph to ensure proper separation
      result.append(nodes.paragraph())
      result.append(nodes.paragraph()) # ensure new line

    return result

  def get_location(self) -> Any:
    return (self.env.docname, self.lineno)

def setup(app):
  app.add_directive("list_config_options", ConfigOptionDirective)

  return {
    "version": "0.1",
    "parallel_read_safe": True,
    "parallel_write_safe": True,
  }
