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

"""Poll for error mode test."""

import os

import jax
from jax._src import distributed
from jax._src import test_multiprocess as jt_multiprocess
import jax.numpy as jnp


def f(x):
  x = jnp.dot(x, x)
  x = jax.lax.psum(x, "i")
  return x


f = jax.pmap(f, "i")


class MultiWorkerPollForErrorTest(jt_multiprocess.MultiProcessTest):

  def test_one_process_fails_without_shutdown(self):
    i = 0
    n = jax.local_device_count()
    x = jnp.ones([n, 1000, 1000])
    while True:
      x = f(x)
      i += 1
      if i > 200 and distributed.global_state.process_id == 1:
        # The following line will exit the program and prevent atexit-registered
        # functions (which includes a shutdown) from running.
        os._exit(0)  # pylint: disable=protected-access
        # Heartbeat stops. The heartbeat timeout error will occur after a short
        # period of time.


if __name__ == "__main__":

  # The heartbeat timeout error is propagated to all live tasks.
  jt_multiprocess.expect_failures_with_regex = (
      "Polled an error from coordination service"
  )
  jt_multiprocess.main()
