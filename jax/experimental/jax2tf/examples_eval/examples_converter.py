# Copyright 2020 Google LLC
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
"""Utils for converting JAX examples to TFLite or TFjs and writing them to Markdown."""
import dataclasses
import datetime
import importlib
import os
from typing import Any, Callable, Dict, Sequence, Tuple

from absl import flags
from absl import logging
import jax
from jax import random
from jax.experimental.jax2tf.examples_eval import all_examples
import jax.numpy as jnp

Arg = all_examples.Arg
FLAGS = flags.FLAGS


@dataclasses.dataclass
class ModuleToConvert:
  """Minimal representation of a Module for converting to TFjs or TFLite."""
  input_shape: Tuple[int, ...]
  apply: Callable[..., Any]
  variables: Any
  dtype: Any


@dataclasses.dataclass
class ConvertSuiteResult:
  """Stores the result of converting an example."""
  suite: all_examples.ExampleSuite
  errors: Dict[str, str]  # {example_name -> error_msg}
  num_skipped: int


def _load_import_safe(path, module) -> Callable[..., Any]:
  """Safely imports a module by fixing flag conflicts."""

  # For some examples, the Flax Module resides in the example's binary, which
  # means we import the command-line flags as well, and these sometimes collide.
  # We remove all potential collisions here.
  conflicts = {'learning_rate', 'batch_size'}
  for duplicate in conflicts & FLAGS.__flags.keys():  # pylint: disable=protected-access
    FLAGS.__delattr__(duplicate)

  return getattr(importlib.import_module(path), module)


def make_module(spec: all_examples.ModuleSpec) -> ModuleToConvert:
  """Builds a ModuleToConvert from the given arguments in `spec`."""
  path, module = spec.module_path.rsplit('.', 1)
  module = _load_import_safe(path, module)(*spec.module_args,
                                           **spec.module_kwargs)
  rng = random.PRNGKey(spec.rng_key)

  spec.apply_kwargs.update(
      dict(method=module.__getattribute__(spec.apply_method_fn)))  # pytype: disable=attribute-error

  def make_rng(rng):
    rng, new_rng = random.split(rng)
    return new_rng

  replace_map = {
      Arg.RNG: make_rng(rng),
      Arg.ONES: jnp.ones(spec.input_shape, spec.dtype)
  }

  def replace(args):
    replace_arg = lambda x: replace_map[x] if isinstance(x, Arg) else x
    return jax.tree_map(replace_arg, args)

  variables = module.init(*replace(spec.init_args), **replace(spec.init_kwargs))

  def apply(variables, inputs):
    replace_map.update({Arg.VARS: variables, Arg.INPUT: inputs})
    result = module.apply(*replace(spec.apply_args),
                          **replace(spec.apply_kwargs))
    return result

  return ModuleToConvert(spec.input_shape, apply, variables, spec.dtype)


def log_summary(converter_name: str, results: Dict[str,
                                                   ConvertSuiteResult]) -> None:
  """Logs `results` to console."""
  logging.info('')
  logging.info('======== SUMMARY FOR CONVERTER %s ========', converter_name)
  logging.info('')

  for suite_name, result in results.items():
    suite = result.suite
    logging.info('Suite: %s (%s)', suite_name, suite.name)
    logging.info('Description: %s', suite.description)
    logging.info('URL: %s', suite.url)
    logging.info('Total examples: %d (Skipped: %d)', len(result.suite.examples),
                 result.num_skipped)

    for example_name, error_msg in result.errors.items():
      success = 'FAIL' if error_msg else 'SUCCESS'

      logging.info('=> Example %s: %s', example_name, success)
      if error_msg:
        logging.info('   Error message: %s', error_msg)


def write_markdown(results: Dict[str, Dict[str, ConvertSuiteResult]]) -> None:
  """Writes all results to Markdown file."""

  # We have to write all results of all converters in one go, otherwise we only
  # replace a single template token, and when we write the next converter
  # result, we will remove the previous one (because we start from the template
  # again).
  all_results = {}

  table_header = """
| Example | Result | Error Message |
| --- | --- | --- |"""

  for converter_name, convert_results in results.items():
    converter_results = []
    for convert_result in convert_results.values():
      suite = convert_result.suite

      converter_results.extend([
          f'### {suite.name}',
          f'[URL to examples]({suite.url})',
          '',
          f'Description: {suite.description}',
          ])

      converter_results.append(table_header)

      for example_name, error_msg in convert_result.errors.items():
        success = 'FAIL' if error_msg else 'SUCCESS'
        error_msg = f' {error_msg}' if error_msg else ''
        converter_results.append(f'| {example_name} | {success} |{error_msg}')
    all_results[converter_name] = converter_results

  dirname = os.path.dirname(__file__)
  input_path = os.path.join(dirname, 'converters_results.md.template')
  output_path = os.path.join(dirname, 'converters_results.md')

  if not os.path.isfile(input_path):
    logging.error('Could not find template at %s -- Aborting writing markdown',
                  input_path)

  with open(input_path) as f_in, open(output_path, 'w') as f_out:
    template = f_in.read()
    template = template.replace('{{generation_date}}',
                                str(datetime.date.today()))

    for c in results.keys():
      template = template.replace('{{' + c + '}}', '\n'.join(all_results[c]))

    f_out.write(template)

  logging.info('Written converter results to %s', output_path)


def test_convert(converter_name: str,
                 converter_fn: Callable[[ModuleToConvert], None],
                 suite_names: str,
                 examples: Sequence[str],
                 fail_on_error: bool) -> Dict[str, ConvertSuiteResult]:
  """Converts given examples using `conversion_fn and return results."""
  logging.info('=== Testing converter %s', converter_name)

  keep_example = lambda s, e: not examples or f'{s}/{e}' in examples

  results = {}
  for suite_name in suite_names:
    suite = all_examples.get_suite(suite_name)
    if not suite:
      logging.error('Suite %s not found!', suite_name)
      continue
    logging.info('Trying suite %s', suite_name)
    errors = {}
    num_skipped = 0
    for example_name, spec in suite.examples.items():
      if not keep_example(suite_name, example_name):
        num_skipped += 1
        continue
      logging.info('Trying example %s', example_name)
      error_msg = ''
      if fail_on_error:
        converter_fn(make_module(spec))
        logging.info('OK!')
      else:
        try:
          converter_fn(make_module(spec))
          logging.info('OK!')
        except Exception as e:  # pylint: disable=broad-except
          error_msg = repr(e)
          logging.info('ERROR %s', error_msg)

      errors[example_name] = error_msg

    results[suite_name] = ConvertSuiteResult(
        suite=suite, errors=errors, num_skipped=num_skipped)

  log_summary(converter_name, results)

  return results
