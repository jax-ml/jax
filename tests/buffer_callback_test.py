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

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax.experimental import buffer_callback
import jax.numpy as jnp
import numpy as np

jax.config.parse_flags_with_absl()


class BufferCallbackTest(jtu.JaxTestCase):

  def test_buffer_callback(self):
    def callback(ctx: buffer_callback.ExecutionContext, out, *args):
      self.assertIsInstance(ctx, buffer_callback.ExecutionContext)
      assert ctx.stage() == buffer_callback.ExecutionStage.EXECUTE
      self.assertIsInstance(out, buffer_callback.Buffer)
      self.assertIsInstance(args[0], buffer_callback.Buffer)
      if jtu.test_device_matches(["cpu"]):
        with self.assertRaisesRegex(
            jax.errors.JaxRuntimeError, "XLA FFI GPU context is not available"
        ):
          ctx.stream()

    call = buffer_callback.buffer_callback(
        callback, jax.ShapeDtypeStruct((), np.float32)
    )
    call(jnp.zeros((1, 2)))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
