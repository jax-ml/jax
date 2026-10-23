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

import jax
from jax import lax
import jax.numpy as jnp
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
from jax._src.lib import ifrt_version
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np


class PPermuteTest(jt_multiprocess.MultiProcessTest):

  def test_ppermute_in_loop_and_after_loop(self):
    if ifrt_version < 79:
      self.skipTest("Requires ifrt_version >= 79")
    n = jax.device_count()
    mesh = jtu.create_mesh((n,), ("x",))
    pairs = [(i, (i + 1) % n) for i in range(n)]
    swap = lambda v: lax.ppermute(v, "x", pairs)

    def program(a, b):
      # Regression test for https://github.com/openxla/xla/issues/50882.
      # On XLA:CPU, 10 local arrays of 65536 int32s exceed the sequential
      # execution thresholds so both the loop body and outer thunk executors
      # run on worker threads.
      local = tuple(jnp.full((1, 65536), k, jnp.int32) for k in range(10))
      a, local = lax.fori_loop(
          0,
          8,
          lambda i, c: (
              swap(c[0]) + 1,
              tuple(e * 3 + k for k, e in enumerate(c[1])),
          ),
          (a, local),
      )
      chain = []
      for _ in range(8):
        b = swap(b)
        chain.append(b)
      return a, chain, local

    spec = P("x", None)
    sh = NamedSharding(mesh, spec)
    put = lambda x: jax.make_array_from_callback(x.shape, sh, lambda i: x[i])
    fn = jax.jit(jax.shard_map(program, mesh=mesh, in_specs=spec, out_specs=spec))

    for trial in range(10):
      a_np = (
          np.arange(n, dtype=np.int32)[:, None] * 10_000 + 1_000_000 + trial
      ) + np.zeros((n, 1024), np.int32)
      b_np = a_np + 1_000_000

      a, chain, _ = fn(put(a_np), put(b_np))

      expected_a = np.roll(a_np, shift=8, axis=0) + 8
      want = [("a", a, expected_a)] + [
          (f"chain[{k}]", c, np.roll(b_np, shift=k + 1, axis=0))
          for k, c in enumerate(chain)
      ]
      for name, r, w in want:
        for s in r.addressable_shards:
          np.testing.assert_array_equal(
              s.data,
              w[s.index],
              err_msg=f"trial {trial} {name} shard {s.index} mismatch",
          )


if __name__ == "__main__":
  jt_multiprocess.main()
