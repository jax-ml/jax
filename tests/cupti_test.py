# Copyright 2024 The JAX Authors.
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
from unittest import SkipTest
from jax._src import test_util as jtu
from jax._src.lib import cuda_versions
import jax
import jax.numpy as jnp
# from jax._src.cudnn import cudnn_fusion
from jax import jit

import tempfile
import pathlib
import jax.profiler
from jax._src import profiler



jax.config.parse_flags_with_absl()


class CuptiTracerTest(jtu.JaxTestCase):
  def setUp(self):
    if (not jtu.test_device_matches(["gpu"])):
      self.skipTest("Only works on GPUs")
    super().setUp()

  @jtu.run_on_devices("gpu")
  def test_cupti_activity_tracing(self):
    @jit
    def xy_plus_z(x, y, z):
        return jnp.float32(jax.lax.batch_matmul(jnp.bfloat16(x), y)) + z
    k = jax.random.key(0)
    s = 1, 16, 16
    jax.devices()
    x = jnp.int8(jax.random.normal(k, shape=s))
    y = jnp.bfloat16(jax.random.normal(k, shape=s))
    z = jnp.float32(jax.random.normal(k, shape=s))
    with tempfile.TemporaryDirectory() as tmpdir_string:
      tmpdir = pathlib.Path(tmpdir_string)
      with jax.profiler.trace(tmpdir):
        print(xy_plus_z(x, y, z))

      proto_path = tuple(tmpdir.rglob("*.xplane.pb"))
      proto_bytes = proto_path[0].read_bytes()
      if jtu.test_device_matches(["gpu"]):
        self.assertIn(b"/device:GPU", proto_bytes)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
