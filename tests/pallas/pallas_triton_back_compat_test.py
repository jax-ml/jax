# Copyright 2023 The JAX Authors.
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
"""Backwards compatibility tests for Pallas Triton kernels.

See module documentation for jax.experimental.jax2tf.tests.back_compat_test_util
for how these tests work and how to create/update the tests.
"""
import math
import os

import numpy as np

from absl.testing import absltest

from jax._src import test_util as jtu
from jax.config import config
import jax.numpy as jnp

from jax.experimental.jax2tf.tests import back_compat_test_util as bctu

from jax.experimental.pallas.ops import softmax

from jax.tests.pallas.back_compat_testdata import softmax as softmax_testdata

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# config.update("jax_traceback_filtering", "off")
config.parse_flags_with_absl()


class PallasTritonKernelTest(bctu.CompatTestBase):

  def test_softmax(self):
    shape = (8, 4)
    dtype = jnp.float16

    def f(x):
      return softmax.softmax(x, axis=-1)

    x = np.arange(math.prod(shape), dtype=dtype).reshape(shape)
    del x  # x is part of the testdata, here only for reference
    data = self.load_testdata(softmax_testdata.data_2023_09_03)
    atol, rtol = {
        jnp.bfloat16: (1e-2, 1e-4),
        jnp.float16: (1e-2, 1e-4),
        jnp.float32: (1e-7, 1e-6),
    }[dtype]
    self.run_one_test(f, data, atol=atol, rtol=rtol)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
