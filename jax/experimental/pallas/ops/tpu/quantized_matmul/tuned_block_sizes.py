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

import jax

# Below are tuned block sizes.

# key:
#    - tpu_version
#    - batch_size
#    - n_output_features
#    - n_input_features
#    - activation_dtype
#    - quantize_activation
# value:
#    - batch_block_size
#    - out_block_size
#    - in_block_size
TUNED_BLOCK_SIZES = {
    # go/keep-sorted start
    (6, 1024, 1280, 8192, 'bfloat16', True): (1024, 256, 8192),
    (6, 1024, 28672, 4096, 'bfloat16', True): (1024, 2048, 4096),
    (6, 1024, 4096, 14336, 'bfloat16', True): (1024, 256, 14336),
    (6, 1024, 4096, 4096, 'bfloat16', True): (1024, 512, 4096),
    (6, 1024, 6144, 4096, 'bfloat16', True): (1024, 768, 4096),
    (6, 1024, 7168, 8192, 'bfloat16', True): (1024, 512, 8192),
    (6, 1024, 8192, 1024, 'bfloat16', True): (1024, 4096, 1024),
    (6, 1024, 8192, 3584, 'bfloat16', True): (1024, 1024, 3584),
    (6, 128, 1280, 8192, 'bfloat16', True): (128, 1280, 2048),
    (6, 128, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 128, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 128, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 128, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 128, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 128, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 128, 8192, 3584, 'bfloat16', True): (128, 8192, 512),
    (6, 16, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 16, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 16, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 16, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 16, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 16, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 16, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 16, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    (6, 2048, 1280, 8192, 'bfloat16', True): (2048, 256, 8192),
    (6, 2048, 28672, 4096, 'bfloat16', True): (2048, 1024, 4096),
    (6, 2048, 4096, 14336, 'bfloat16', True): (2048, 4096, 512),
    (6, 2048, 4096, 4096, 'bfloat16', True): (2048, 512, 4096),
    (6, 2048, 6144, 4096, 'bfloat16', True): (2048, 512, 4096),
    (6, 2048, 7168, 8192, 'bfloat16', True): (2048, 256, 8192),
    (6, 2048, 8192, 1024, 'bfloat16', True): (256, 8192, 1024),
    (6, 2048, 8192, 3584, 'bfloat16', True): (2048, 512, 3584),
    (6, 256, 1280, 8192, 'bfloat16', True): (256, 256, 8192),
    (6, 256, 28672, 4096, 'bfloat16', True): (256, 2048, 4096),
    (6, 256, 4096, 14336, 'bfloat16', True): (256, 4096, 512),
    (6, 256, 4096, 4096, 'bfloat16', True): (256, 512, 4096),
    (6, 256, 6144, 4096, 'bfloat16', True): (256, 512, 4096),
    (6, 256, 7168, 8192, 'bfloat16', True): (256, 512, 8192),
    (6, 256, 8192, 1024, 'bfloat16', True): (256, 2048, 1024),
    (6, 256, 8192, 3584, 'bfloat16', True): (256, 8192, 512),
    (6, 32, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 32, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 32, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 32, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 32, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 32, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 32, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 32, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    (6, 512, 1280, 8192, 'bfloat16', True): (512, 256, 8192),
    (6, 512, 28672, 4096, 'bfloat16', True): (512, 2048, 4096),
    (6, 512, 4096, 14336, 'bfloat16', True): (512, 256, 14336),
    (6, 512, 4096, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 6144, 4096, 'bfloat16', True): (512, 1024, 4096),
    (6, 512, 7168, 8192, 'bfloat16', True): (512, 512, 8192),
    (6, 512, 8192, 1024, 'bfloat16', True): (512, 4096, 1024),
    (6, 512, 8192, 3584, 'bfloat16', True): (512, 2048, 3584),
    (6, 64, 1280, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 64, 28672, 4096, 'bfloat16', True): (128, 28672, 256),
    (6, 64, 4096, 14336, 'bfloat16', True): (128, 4096, 896),
    (6, 64, 4096, 4096, 'bfloat16', True): (128, 512, 4096),
    (6, 64, 6144, 4096, 'bfloat16', True): (128, 768, 4096),
    (6, 64, 7168, 8192, 'bfloat16', True): (128, 256, 8192),
    (6, 64, 8192, 1024, 'bfloat16', True): (128, 2048, 1024),
    (6, 64, 8192, 3584, 'bfloat16', True): (128, 1024, 3584),
    # go/keep-sorted end
}


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_key(
    batch_size,
    n_output_features,
    n_input_features,
    activation_dtype,
    quantize_activation,
):
  """Returns the key for the given parameters."""
  return (
      get_tpu_version(),
      batch_size,
      n_output_features,
      n_input_features,
      activation_dtype,
      quantize_activation,
  )


def get_tuned_block_sizes(
    batch_size,
    n_output_features,
    n_input_features,
    activation_dtype,
    quantize_activation,
):
  """Retrieve the tuned block sizes for the given parameters.

  Args:
      batch_size (int): The batch size.
      n_output_features (int): The number of output features.
      n_input_features (int): The number of input features.
      activation_dtype (str): The data type of the activation ('bfloat16' or
        'float32').
      quantize_activation (bool): Whether to quantize the activation.

  Returns:
      tuple: A tuple containing the batch_block_size, out_block_size, and
      in_block_size.
  """
  key = get_key(
      batch_size,
      n_output_features,
      n_input_features,
      activation_dtype,
      quantize_activation,
  )
  return TUNED_BLOCK_SIZES.get(key, (128, 128, 128))
