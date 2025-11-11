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
import dataclasses
from typing import Any

from absl.testing import absltest
import jax
from jax import numpy as jnp
from jax._src import core as jax_core
from jax._src.pallas.pipelining import schedule_api
from jax._src.pallas.pipelining import test_util
from jax._src.state import types as state_types
import numpy as np


jax.config.parse_flags_with_absl()


@dataclasses.dataclass(frozen=True)
class MemoryRef:
  shape: tuple[int, ...]
  dtype: np.dtype
  memory_space: Any | None = None

  def get_ref_aval(self) -> state_types.AbstractRef:
    return state_types.AbstractRef(
        inner_aval=jax_core.ShapedArray(shape=self.shape, dtype=self.dtype),
        memory_space=self.memory_space,
    )


class ApiTest(absltest.TestCase):

  def test_basic_pipeline(self):
    # Use reads/writes to mimic the Ref effects of DMAs.
    copy_in = schedule_api.AsyncStage(max_in_flight=2)

    @copy_in.def_start
    def copy_in_start(_, x_ref, o_ref):
      del o_ref
      # dma_start creates a write_effect to x_ref
      x_ref[...] = jnp.ones_like(x_ref)

    @copy_in.def_end
    def copy_in_end(_, x_ref, o_ref):
      del o_ref
      # dma_end creates a write_effect to x_ref
      x_ref[...] = jnp.ones_like(x_ref)

    @schedule_api.stage(max_in_flight=2)
    def kernel_body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...] + 1.0

    copy_out = schedule_api.AsyncStage(max_in_flight=2)
    @copy_out.def_start
    def copy_out_start(_, x_ref, o_ref):
      del x_ref
      # dma_start creates a read_effect to o_ref
      _ = o_ref[...]

    @copy_out.def_end
    def copy_out_end(_, x_ref, o_ref):
      del x_ref
      # dma_end creates a read_effect to o_ref
      _ = o_ref[...]

    pipeline = schedule_api.schedule_pipeline(
        stages=(copy_in, kernel_body, copy_out),
        grid=(4,),
        args=(
            MemoryRef(shape=(128, 128), dtype=jnp.dtype(jnp.float32),
                      memory_space="VMEM"),
            MemoryRef(shape=(128, 128), dtype=jnp.dtype(jnp.float32),
                      memory_space="VMEM"),
        ),
        eval_fn=test_util.print_stage,
    )
    ref = jnp.ones((128, 128), jnp.float32)
    ref = jax.new_ref(ref)
    with test_util.capture_stdout() as stdout:
      pipeline(ref, ref)
    output = stdout().strip().split("\n")
    expected = [
        # step
        "[itr=0] copy_in_start",
        "[itr=1] copy_in_start",
        # step
        "[itr=0] copy_in_end",
        "[itr=0] kernel_body",
        "[itr=0] copy_out_start",
        "[itr=2] copy_in_start",
        # step
        "[itr=1] copy_in_end",
        "[itr=1] kernel_body",
        "[itr=1] copy_out_start",
        "[itr=3] copy_in_start",
        # step
        test_util.AnyOrder([
            "[itr=0] copy_out_end",
            "[itr=2] copy_in_end"]),
        "[itr=2] kernel_body",
        "[itr=2] copy_out_start",
        # step
        test_util.AnyOrder([
            "[itr=1] copy_out_end",
            "[itr=3] copy_in_end"]),
        "[itr=3] kernel_body",
        "[itr=3] copy_out_start",
        # step
        "[itr=2] copy_out_end",
        "[itr=3] copy_out_end",
    ]
    self.assertTrue(test_util.compare_lists(output, expected))


if __name__ == "__main__":
  absltest.main()
