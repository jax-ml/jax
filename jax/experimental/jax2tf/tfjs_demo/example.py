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
from absl import app # type: ignore
import os # type: ignore
from tempfile import mkdtemp
from typing import Any, Sequence

from jax import make_jaxpr # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflowjs.converters import convert_tf_saved_model # type: ignore

from jax.config import config # type: ignore
config.config_with_absl()

from jax.experimental import jax2tf # type: ignore

TfVal = Any
def generate_demo_file(path_to_model: str, input_signature: Sequence[TfVal],
                       *tensors: TfVal) -> None:
  assert len(tensors) == len(input_signature), (
      "Expected {len(input_signature)} tensors but got {len(tensors)}.")

  def make_arg_name(arg_id: int) -> str:
    return f'arg{arg_id}'

  def make_arg_decl(arg_id: int, tensor: TfVal) -> str:
    # TODO
    content = tensor.numpy().tolist()
    shape = list(tensor.shape)
    try:
      dtype = {tf.float32: 'float32', tf.int32: 'int32', tf.bool: 'bool',
               tf.complex64: 'complex64', tf.string: 'string'}[tensor.dtype]
    except:
      raise TypeError(f'dtype not supported: {tensor.dtype}')

    return (f"const {make_arg_name(arg_id)} = "
            f"tf.tensor({content}, {shape}, '{dtype}')")

  def make_dict_arg(arg_name: str, arg_var: str) -> str:
    return f"'{arg_name}': {arg_var}"

  stringified_arg_decl = (
    '\n'.join(list(map(lambda et: make_arg_decl(*et), enumerate(tensors)))))
  arg_dict = {}

  for param_id, arg_tuple in enumerate(zip(input_signature, tensors)):
    tensor_spec, tensor = arg_tuple
    arg_name = (tensor_spec.name if tensor_spec.name is not None
                else f"args_{param_id}:0")
    arg_var = make_arg_name(param_id)
    arg_dict[arg_name] = arg_var

  stringified_arg_dict = (
    ', '.join(list(map(lambda at: make_dict_arg(*at), arg_dict.items()))))

  with open('demo.js.template', 'rb') as template_file:
    output = template_file.read()

  output = (
    output.replace(b'{const_tensor_args_decl}', stringified_arg_decl.encode())
          .replace(b'{path_to_model}', path_to_model.encode())
          .replace(b'{args_as_dictionary}', stringified_arg_dict.encode()))

  with open('result.js', 'wb') as output_file:
    output_file.write(output)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Creating a saved model
  model = tf.Module(name="testModule")
  input_signature = [ tf.TensorSpec(shape=(10,), dtype=np.float32, name="test")
                    , tf.TensorSpec(shape=(10,), dtype=np.float32)
                    ]

  func_jax = lambda x, y: (x + y) * y
  a, b = np.ones((10,), dtype=np.float32), np.ones((10,), dtype=np.float32)
  print(make_jaxpr(func_jax)(a, b))

  model.predict = tf.function(jax2tf.convert(func_jax, with_gradient=True),
                              autograph=False, input_signature=input_signature)
  model_dir = mkdtemp()
  conversion_dir = os.path.join(os.path.dirname(__file__), './tfjs_models')

  print(model.predict(a, b))
  tf.saved_model.save(model, model_dir)
  convert_tf_saved_model(model_dir, conversion_dir)

  a, b = list(map(tf.convert_to_tensor, [a, b]))
  generate_demo_file(os.path.join(conversion_dir, 'model.json'),
                     input_signature, a, b)

if __name__ == "__main__":
  app.run(main)
