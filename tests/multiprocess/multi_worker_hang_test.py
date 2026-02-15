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

"""Reproduction for b/167570287."""

import jax
from jax._src import test_multiprocess as jt_multiprocess
import jax.numpy as jnp


def f(x):
  x = jnp.dot(x, x)
  x = jax.lax.psum(x, "i")
  return x


f = jax.pmap(f, "i")


class MultiWorkerJaxHangTest(jt_multiprocess.MultiProcessTest):

  def testHang(self):
    i = 0
    n = jax.local_device_count()
    x = jnp.ones([n, 1000, 1000])
    while True:
      x = f(x)
      i += 1
      if i > 200 and jax.process_index() == 1:
        raise RuntimeError("Fake NaN!")


if __name__ == "__main__":
  # TODO(skyewm): modify multiprocess_test.py so we can reliably test both
  # the coordinator and the client process raising "Fake NaN". Currently it can
  # nondeterministically be either depending on which chips are assigned which
  # kernel device IDs, which changes the behavior of
  # --deepsea_hal_excluded_devs.
  jt_multiprocess.expect_failures_with_regex = (
      "(Fake NaN|detected fatal errors)")
  jt_multiprocess.main()
