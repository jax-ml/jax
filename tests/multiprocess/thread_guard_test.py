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

"""Test thread guard for multiprocess arrays."""

import concurrent.futures

import jax
from jax._src import config
from jax._src import test_multiprocess as jt_multiprocess
from jax._src.lib import jaxlib_extension_version
import jax.numpy as jnp


@jax.jit
def f(x):
  y = jnp.square(x)
  z = jnp.cos(y)
  return jnp.sum(z)


@jax.jit
def g(x):
  y = jnp.square(x)
  z = jnp.sin(y)
  return z + 1


def _make_multiprocess_array():
  mesh = jax.make_mesh(
      (jax.device_count(),), ('i',),
      axis_types=(jax.sharding.AxisType.Explicit,), devices=jax.devices())
  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
  x = jnp.ones((jax.device_count(),))
  return jax.device_put(x, sharding)


class ThreadGuardTest(jt_multiprocess.MultiProcessTest):

  # Use a single test method since the thread guard affects global state and
  # tests can't run in parallel.
  def test_thread_guard(self):
    if jaxlib_extension_version < 392:
      self.skipTest(
          'Thread guard is supported only in jaxlib version >= 392. Jaxlib '
          f'version is {jaxlib_extension_version}')

    # Test slow JIT path.
    x = _make_multiprocess_array()
    with self.assertRaisesRegex(
        (RuntimeError, ValueError), 'thread guard was set'):
      with (config.thread_guard(True),
        concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor):
        y = executor.submit(f, x)
      jax.block_until_ready(y.result())

    # Test fast JIT path.
    x = _make_multiprocess_array()
    x = f(x)
    x = f(x)
    with self.assertRaisesRegex(
        (RuntimeError, ValueError), 'thread guard was set'):
      with (config.thread_guard(True),
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor):
        y = executor.submit(f, x)
      jax.block_until_ready(y.result())

    # Test local devices only.
    mesh = jax.make_mesh(
        (jax.local_device_count(),), ('i',),
        axis_types=(jax.sharding.AxisType.Explicit,),
        devices=jax.local_devices())
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    x = jnp.ones((jax.local_device_count(),))
    x = jax.device_put(x, sharding)

    with config.thread_guard(True):
      z = g(x)
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        y = executor.submit(f, x).result()
      out = y + z
      jax.block_until_ready(out)  # No cross-process arrays, so no error.

    # Test nested thread guard context managers.
    x = _make_multiprocess_array()
    with config.thread_guard(True):
      y = f(x)
      with config.thread_guard(True):
        z = g(y)  # No error when context manager is redundantly nested.
        jax.block_until_ready(z)
        with config.thread_guard(False):
          with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            w = executor.submit(f, x).result()
          jax.block_until_ready(w)  # No error, thread guard deactivated.

        # Thread guard is re-activated by the outer context manager.
        with self.assertRaisesRegex(
            (RuntimeError, ValueError), 'thread guard was set'):
          with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            v = executor.submit(g, w)
          jax.block_until_ready(v.result())

    # No error on the call in a different thread outside the context manager.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      y = executor.submit(f, x).result()
      jax.block_until_ready(y)

    # Test thread guard in a subthread.
    x = _make_multiprocess_array()

    def f_with_thread_guard_should_raise(x):
      with (config.thread_guard(True),
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor):
        return executor.submit(f, x).result()

    with self.assertRaisesRegex(
        (RuntimeError, ValueError), 'thread guard was set'):
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        y = executor.submit(f_with_thread_guard_should_raise, x).result()
        jax.block_until_ready(y)

    # Test nested thread guard context managers in different threads raises.
    x = _make_multiprocess_array()

    def f_with_thread_guard(x):
      with config.thread_guard(True):
        return f(x)

    with self.assertRaisesRegex(
        (RuntimeError, ValueError),
        'Nested thread guards in different threads are not supported'):
      with config.thread_guard(True):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
          y = executor.submit(f_with_thread_guard, x).result()
          jax.block_until_ready(y)


if __name__ == '__main__':
  jt_multiprocess.main()
