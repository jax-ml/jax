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
from jax import lax
from jax._src import test_multiprocess as jt_multiprocess
import numpy as np


class AxisIndexTest(jt_multiprocess.MultiProcessTest):

  def test(self):
    f = jax.pmap(lambda _: lax.axis_index("i"), axis_name="i")
    n = jax.local_device_count()
    xs = np.arange(n)
    out = f(xs * 0)
    np.testing.assert_equal(out, xs + (n * jax.process_index()))


if __name__ == "__main__":
  jt_multiprocess.main()
