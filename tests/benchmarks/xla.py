# Copyright 2019 Google LLC
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


import enum
import pytest
import numpy as np

from jax import numpy as jnp
from jax.interpreters import xla


class AnEnum(enum.IntEnum):
  A = 123
  B = 456


_abstractify_args = [
  3,
  3.5,
  np.int32(3),
  np.uint32(7),
  np.random.randn(3, 4, 5, 6),
  np.arange(100, dtype=np.float32),
  jnp.int64(-3),
  jnp.array([1, 2, 3]),
  AnEnum.B,
]

@pytest.mark.parametrize("arg", _abstractify_args)
def test_abstractify(benchmark, arg):
  benchmark(xla.abstractify, arg)
