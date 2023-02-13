# Copyright 2018 The JAX Authors.
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

from functools import partial
import re
import sys
import unittest
import numpy as np
import os


from absl.testing import absltest

import jax
import jax.numpy as jnp
from jax import lax
from jax.config import config
from jax.interpreters import batching

import jax._src.lib
import jax._src.util
from jax._src import core
from jax._src import test_util as jtu

config.parse_flags_with_absl()
FLAGS = config.FLAGS

python_version = (sys.version_info[0], sys.version_info[1])



@jtu.with_config(
    jax_dynamic_shapes=True, jax_numpy_rank_promotion="allow")
class DynamicShapeTest(jtu.JaxTestCase):
  @unittest.skipIf(jtu.device_under_test() != "cpu", "cpu test")
  def test_basic_staging(self):
    def f(x, y):
      return x + y

    size = jax.lax.convert_element_type(2, core.bint(5))
    x = jnp.ones([size, 3])
    y = jnp.ones([size, 3])
    out = jax.jit(f, abstracted_axes=("n",), backend="cpu")(x, y)
    print(out)
    assert False


if __name__ == "__main__":
  os.environ["XLA_FLAGS"] = "--xla_cpu_use_xla_runtime"
  absltest.main(testLoader=jtu.JaxTestLoader())
