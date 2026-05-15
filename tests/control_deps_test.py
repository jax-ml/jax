# Copyright 2026 The JAX Authors.
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

import functools
from absl.testing import absltest
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.lax import parallel
from jax._src.lib import jaxlib_extension_version
from jax.experimental.control_deps import control_dep, schedule
import jax.numpy as jnp

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class ControlDepsTest(jtu.JaxTestCase):
  def setUp(self):
    if jaxlib_extension_version < 453:
      self.skipTest('Requires jaxlib_extension_version >= 453')

  def create_explicit_mesh(self, axes, names):
    axis_types = (jax.sharding.AxisType.Explicit,) * len(axes)
    return jtu.create_mesh(axes, names, iota_order=False, axis_types=axis_types)

  def test_math(self):
    def f_math(x, y, z):
      a = jnp.sin(x @ x)
      b = jnp.cos(y @ y)
      c = jnp.exp(z @ z)
      schedule([c, b, a])
      return a + b + c

    x = jnp.ones((67, 67))
    hlo = jax.jit(f_math).lower(x, x, x).as_text(dialect="hlo")
    self.assertIn('custom_call_target="control_dep"', hlo)

  def test_fsdp(self):
    k = 5
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ("i",))):
      @jax.jit
      @jax.shard_map(out_specs=(jax.P("i")), check_vma=False)
      def f_fsdp(x, ws):
        starts = []
        dones = []
        maths = []

        # This is a simple version of FSDP where x is like a set of activations
        # and ws is a list of weights, one per layer. We repeatedly all-gather
        # the weights for a layer and multiply with x.
        for w_shard in ws:
          f = parallel.all_gather_start(w_shard, "i", tiled=True)
          w = f.done()
          x = x @ w

          # Note that we pipe out the intermediate values.
          starts.append(f)
          dones.append(w)
          maths.append(x)

        # Here we schedule the code to run in a smart FSDP order where the all
        # gather for the next layer is overlapped with the math for the current
        # layer.
        deps = []
        for i in range(k + 1):
          if i == 0:
            deps.append(starts[0].x)
            deps.append(dones[0])
          elif i < k:
            deps.append(starts[i].x)
            deps.append(maths[i - 1])
            deps.append(dones[i])
          else:
            deps.append(maths[i - 1])
        schedule(deps)

        return x

      N = 128 * n
      x = jnp.ones((n * N, N), out_sharding=jax.P("i", None))
      ws = [jnp.ones((N, N), out_sharding=jax.P("i", None)) for _ in range(k)]
      hlo = jax.jit(f_fsdp).lower(x, ws).as_text(dialect="hlo")
      self.assertIn('custom_call_target="control_dep"', hlo)

  def test_scan_fsdp(self):
    k = 10
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ("i",))):
      # This test shows FSDP with scan.
      @jax.jit
      @jax.shard_map(out_specs=(jax.P("i")), check_vma=False)
      def f_scan_fsdp(x, ws):
        # Prologue.
        w_0 = jax.lax.all_gather(ws[0], "i", tiled=True)

        # Scan.
        def f(carry, w_shard):
          w, x = carry
          fut = parallel.all_gather_start(w_shard, "i", tiled=True)
          x = x @ w
          w_next = fut.done()
          schedule([fut.x, x, w_next])
          return (w_next, x), None
        (w, x), _ = jax.lax.scan(f, (w_0, x), ws[1:])

        # Epilogue.
        x = x @ w_0

        return x

      N = 128 * n
      x = jnp.ones((n * N, N), out_sharding=jax.P("i", None))
      ws = jnp.ones((k, N, N), out_sharding=jax.P(None, "i", None))
      hlo = jax.jit(f_scan_fsdp).lower(x, ws).as_text(dialect="hlo")
      self.assertIn('custom_call_target="control_dep"', hlo)

  def test_pipeline(self):
    k = 10
    n = jax.device_count()
    with jax.set_mesh(self.create_explicit_mesh((n,), ("i",))):

      @jax.jit
      @jax.shard_map(out_specs=(jax.P("i")), check_vma=False)
      def f_pipeline(xs, ws):
        starts = []
        dones = []
        maths = []

        # This shows a form of pipelining across microbatches. xs and ws are the
        # same size. We need to run xs[i] @ ws[i] for every i.
        for x, w_shard in zip(xs, ws):
          f = parallel.all_gather_start(w_shard, "i", tiled=True)
          w = f.done()
          y = x @ w

          starts.append(f)
          dones.append(w)
          maths.append(y)

        # We schedule things to run all the starts, then done and math in the
        # right order.
        schedule([s.x for s in starts])
        schedule([starts[-1].x, dones[0]])
        schedule(maths)
        for i in range(k - 1):
          control_dep(maths[i], dones[i + 1])

        return functools.reduce(lambda x, y: x + y, maths)

      N = 128 * n
      x = [jnp.ones((N, N), out_sharding=jax.P(None, None)) for _ in range(k)]
      ws = [jnp.ones((N, N), out_sharding=jax.P("i", None)) for _ in range(k)]
      hlo = jax.jit(f_pipeline).lower(x, ws).as_text(dialect="hlo")
      self.assertIn('custom_call_target="control_dep"', hlo)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
