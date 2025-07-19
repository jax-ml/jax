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

"""Utility functions for ragged paged attention."""

import jax
from jax._src import dtypes


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(x, a):
  return cdiv(x, a) * a


def get_dtype_packing(dtype):
  bits = dtypes.bit_width(dtype)
  return 32 // bits


def next_power_of_2(x: int):
  """Finds the smallest power of 2 >= x using bit manipulation.

  Args:
    x: The input number (should be an integer).

  Returns:
    The smallest integer power of 2 that is >= x.
  """
  assert x > 0
  if x == 1:
    return 1
  return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
  """Returns the numeric version of the TPU, or -1 if not on TPU."""
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    return -1
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')]
  assert kind[:-1] == 'TPU v', kind
  return int(kind[-1])


def get_device_name(num_devices: int | None = None):
  name = ' '.join(jax.devices()[0].device_kind.split()[:2])
  if num_devices is not None:
    name += f'-{num_devices}'
  return name
