# Copyright 2024 The JAX Authors.
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

import numpy as np

import jax
import jax.extend as jex

from jax_ffi_example import _cpu_examples

for name, target in _cpu_examples.registrations().items():
  jex.ffi.register_ffi_target(name, target)


def array_attr(num: int):
  return jex.ffi.ffi_call(
      "array_attr",
      jax.ShapeDtypeStruct((), np.int32),
  )(array=np.arange(num, dtype=np.int32))


def dictionary_attr(**kwargs):
  return jex.ffi.ffi_call(
      "dictionary_attr",
      (jax.ShapeDtypeStruct((), np.int32), jax.ShapeDtypeStruct((), np.int32)),
  )(**kwargs)


def counter(index):
  return jex.ffi.ffi_call(
    "counter", jax.ShapeDtypeStruct((), jax.numpy.int32))(index=int(index))
