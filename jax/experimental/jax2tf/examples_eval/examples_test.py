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
r"""Converts JAX examples to TFjs and TFLite using specifiable converters.

Run this as follows:

```
# Clone the Flax examples in a local directory.
git clone https://github.com/google/flax.git /tmp/flax/
FLAX_EXAMPLES=/tmp/flax/examples

# Run all converters on all suites and regenerate the table.
# The new output table is in README.md.
python3 examples_test.py --flax_examples_path=$FLAX_EXAMPLES

# Run only MNIST and WMT with the TFjs converter and don't write Markdown.
python3 examples_test.py \
    --examples=flax/mnist,flax/wmt \
    --converters=jax2tf_to_tfjs \
    --write_markdown=False \
    --flax_examples_path=$FLAX_EXAMPLES
```
"""
import sys
from typing import Sequence

from absl import app
from absl import flags
from jax.experimental.jax2tf.examples_eval import converters
from jax.experimental.jax2tf.examples_eval import examples_converter


CONVERTERS = {
    'jax2tf_to_tfjs': converters.jax2tf_to_tfjs,
    'jax2tf_to_tflite': converters.jax2tf_to_tflite
}

SUITES = ['flax']

flags.DEFINE_string('flax_examples_path',
                    '/Users/marcvanzee/git/flax/examples',
                    'Local path to Flax examples.')

flags.DEFINE_list(
    'converters', list(CONVERTERS.keys()),
    f'Which converters to test. Available converters: {list(CONVERTERS.keys())}'
)

flags.DEFINE_list(
    'example_suites',
    SUITES,
    f'Which example suites to test. Available suites: {SUITES}')

flags.DEFINE_list(
    'examples',
    [],
    ('List of suite/examples to test, e.g.: "flax/mnist,flax/seq2seq". '
     'If empty, will test all examples.'))

flags.DEFINE_bool(
    'write_markdown',
    True,
    'If true, write results as Markdown. Otherwise, only output to stdout.')

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  sys.path.append(FLAGS.flax_examples_path)

  results = {}

  for converter_name in FLAGS.converters:
    results[converter_name] = examples_converter.test_convert(
        converter_name=converter_name,
        converter_fn=CONVERTERS[converter_name],
        suite_names=FLAGS.example_suites,
        examples=FLAGS.examples)

  if FLAGS.write_markdown:
    examples_converter.write_markdown(results)


if __name__ == '__main__':
  app.run(main)
