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
import os
from tempfile import mkdtemp
from typing import Any, Callable, List, NamedTuple, Tuple

from jax.experimental import jax2tf # type: ignore
import tensorflow as tf # type: ignore
from tensorflowjs.converters import convert_tf_saved_model # type: ignore

ConversionResult = (
  NamedTuple("ConversionResult", [ ("test_name", str)
                                 , ("input_signature", Tuple[tf.TensorSpec,...])
                                 , ("succeeded", bool)
                                 ]))

conversion_results: List[ConversionResult] = []

def prettify_conversion_results(conversion_results: List[ConversionResult]) \
    -> str:
  conversion_results = list(set(conversion_results))
  def _pipewrap(columns):
    return '| ' + ' | '.join(columns) + ' |'

  column_names = ['Function', 'Input signature']

  header = [column_names, ['---'] * len(column_names)]
  table = []

  for conv_res in conversion_results:
    # Only gather failures
    if conv_res.succeeded:
      continue
    table.append([ conv_res.test_name
                 , ', '.join(list(map(lambda ts: ts.dtype.name,
                                      conv_res.input_signature)))
                 ])

  return ('\n'.join(line for line in map(_pipewrap, header + sorted(table)))
          .replace('_', '\\_'))

def pprint_conversion_summary(output_file: str) -> None:
  output = prettify_conversion_results(conversion_results)
  with open(output_file, 'w') as f:
    f.write(output)

def save_and_convert_model(func_jax: Callable, *args) -> None:
  def _make_tensorspec(arg: Any) -> tf.TensorSpec:
    try:
      return tf.TensorSpec.from_tensor(tf.convert_to_tensor(arg))
    except:
      pass
    # TODO(bchetioui): can we do better?
    dtype = arg.dtype if hasattr(arg, 'dtype') else tf.float32
    shape = arg.shape if hasattr(arg, 'shape') else ()
    return tf.TensorSpec(shape, dtype)

  # TODO(bchetioui): is this fine?
  test_name = os.environ.get('PYTEST_CURRENT_TEST')
  assert test_name is not None
  test_name = test_name.split(':')[-1].split(' (call)')[0]

  model = tf.Module()
  input_signature = list(map(_make_tensorspec, args))
  model.f = tf.function(jax2tf.convert(func_jax, with_gradient=True),
                        autograph=False,
                        input_signature=input_signature)

  model_dir, conversion_dir = mkdtemp(), mkdtemp()

  has_succeeded = True
  try:
    tf.saved_model.save(model, model_dir)
    convert_tf_saved_model(model_dir, conversion_dir)
  except Exception as e:
    print(f'Attempted to convert {test_name} but failed with: {e}')
    has_succeeded = False

  conversion_results.append(
      ConversionResult(test_name, tuple(input_signature), has_succeeded))
