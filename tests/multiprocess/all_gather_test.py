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

from absl.testing import parameterized
import jax
from jax import lax
from jax._src import test_multiprocess as jt_multiprocess
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np


class AllGatherTest(jt_multiprocess.MultiProcessTest):

  @parameterized.parameters(
      (np.int32,), (jnp.float32,), (jnp.float16,), (jnp.bfloat16,)
  )
  def test_all_gather_shard_map(self, dtype):
    mesh_shape = (jax.process_count(), jax.local_device_count())
    mesh = jtu.create_mesh(mesh_shape, ("x", "y"))
    spec = jax.P("x", "y")

    @jax.shard_map(
        mesh=mesh, in_specs=spec, out_specs=jax.P(None, None), check_vma=False
    )
    def f(x):
      out = lax.all_gather(x, "x", axis=0, tiled=True)
      return lax.all_gather(out, "y", axis=1, tiled=True)

    global_len = np.prod(mesh_shape)
    global_arr = jnp.arange(global_len, dtype=dtype).reshape(mesh_shape)
    sharding = jax.NamedSharding(mesh, spec)
    global_xs = jax.make_array_from_callback(
        mesh_shape, sharding, lambda index: global_arr[index]
    )

    out = f(global_xs)
    for actual in out.addressable_shards:
      jtu.check_close(actual.data, global_arr[actual.index])


if __name__ == "__main__":
  jt_multiprocess.main()
