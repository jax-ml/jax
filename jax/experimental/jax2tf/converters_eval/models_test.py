# Copyright 2022 Google LLC
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
r"""Converts JAX examples to TFjs and TFLite using specifiable converters.

Run this as follows:

```
# Run all converters on all suites and regenerate the table.
# The new output table is in README.md.
python3 models_test.py

# Run only CNN and Resnet50 with the TFjs converter and don't write Markdown.
python3 models_test.py \
    --examples=flax/cnn,flax/resnet50 \
    --converters=jax2tf_to_tfjs \
    --write_markdown=False
```
"""
from typing import Sequence
import re

from absl import app
from absl import flags
from absl import logging
from jax.experimental.jax2tf.converters_eval import converters

import datetime
import os
from typing import Dict, List, Tuple

from jax.experimental.jax2tf.converters_eval.models import get_test_cases


CONVERTERS = {
    'jax2tf_xla': converters.jax2tf_xla,
    'jax2tf_to_tfjs': converters.jax2tf_to_tfjs,
    'jax2tf_to_tflite': converters.jax2tf_to_tflite
}


flags.DEFINE_list(
    'converters', list(CONVERTERS.keys()),
    f'Which converters to test. Available converters: {list(CONVERTERS.keys())}'
)

flags.DEFINE_list('examples', [],
                  ('List of examples to test, e.g.: "flax/mnist,flax/seq2seq". '
                   'If empty, will test all examples.'))

flags.DEFINE_bool(
    'write_markdown',
    True,
    'If true, write results as Markdown. Otherwise, only output to stdout.')

flags.DEFINE_bool(
    'fail_on_error',
    False,
    ('If true, exit with an error when a conversion fails. Useful for '
     'debugging because it will show the entire stack trace.')
)

flags.DEFINE_string(
    'output_path', None,
    'Output path to write results table to. If empty will write to '
    'converters_results.md in same folder.')

FLAGS = flags.FLAGS


def write_markdown(results: Dict[str, List[Tuple[str, str,]]]) -> None:
  """Writes all results to Markdown file."""
  converters = FLAGS.converters

  table_lines = []

  table_lines.append('| Example | ' + ' '.join([f'{c} |' for c in converters]))
  table_lines.append('|' + (' --- |' * (len(converters)+1)))

  error_lines = []

  def _header2anchor(header):
    header = header.lower().replace(' ', '-')
    header = '#' + re.sub(r'[^a-zA-Z0-9_-]+', '', header)
    return header

  for example_name, converter_results in results.items():
    # Make sure the converters are in the right order.
    assert [c[0] for c in converter_results] == converters
    line = f'| {example_name} |'
    for converter_name, error_msg in converter_results:
      if not error_msg:
        line += ' YES |'
      else:
        error_header = (f'Error trace: model={example_name}, '
                        f'converter={converter_name}')
        error_lines.append('## ' + error_header)
        error_lines.append(f'```\n{error_msg}\n```')
        error_lines.append('[Back to top](#summary-table)')
        line += f' [NO]({_header2anchor(error_header)}) | '
    table_lines.append(line)

  dirname = os.path.dirname(__file__)
  input_path = os.path.join(dirname, 'converters_results.md.template')
  if FLAGS.output_path:
    output_path = FLAGS.output_path
  else:
    output_path = os.path.join(dirname, 'converters_results.md')

  with open(input_path) as f_in, open(output_path, 'w') as f_out:
    template = f_in.read()
    template = template.replace('{{generation_date}}',
                                str(datetime.date.today()))
    template = template.replace('{{table}}', '\n'.join(table_lines))
    template = template.replace('{{errors}}', '\n'.join(error_lines))
    f_out.write(template)

  logging.info('Written converter results to %s', output_path)


def test_converters():
  """Tests all converters and write output."""
  results = {}
  converters = {x: CONVERTERS[x] for x in FLAGS.converters}

  for example_name, test_case_fn in get_test_cases().items():
    if FLAGS.examples and example_name not in FLAGS.examples:
      continue
    test_case = test_case_fn()  # This will create the model's variables.
    converter_results = []
    for converter_name, converter_fn in converters.items():
      logging.info('===== Testing example %s, Converter %s',
                   example_name, converter_name)
      error_msg = ''
      try:
        converter_fn(test_case)
        logging.info('=== OK!')
      except Exception as e:  # pylint: disable=broad-except
        if FLAGS.fail_on_error:
          raise e
        error_msg = repr(e).replace('\\n', '\n')
        logging.info(
            '=== ERROR %s',
            error_msg if len(error_msg) < 250 else error_msg[:250] +
            '... (CROPPED)')
      converter_results.append((converter_name, error_msg))
    results[example_name] = converter_results

  if FLAGS.write_markdown:
    write_markdown(results)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  test_converters()


if __name__ == '__main__':
  app.run(main)
