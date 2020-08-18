# Copyright 2019 Google LLC
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
import jax.numpy as jnp
from jax.tools.jax_to_hlo import jax_to_hlo
from jax.lib import xla_client
from jax import test_util as jtu


class JaxToHloTest(absltest.TestCase):

  def test_convert_axpy(self):

    def axpy(a, x, y):
      return a * x + y[:, jnp.newaxis]

    hlo_proto, hlo_text = jax_to_hlo(
        axpy, [
            ('y', xla_client.Shape('f32[128]')),
            ('a', xla_client.Shape('f32[]')),
            ('x', xla_client.Shape('f32[128,2]')),
        ])

    # Check that hlo_text contains a broadcast, add, and multiply.
    self.assertIn('broadcast', hlo_text)
    self.assertIn('add', hlo_text)
    self.assertIn('multiply', hlo_text)

    # Check that the HLO parameters are in the order we specified in the
    # jax_to_hlo call.
    self.assertIn('f32[128]{0} parameter(0)', hlo_text)
    self.assertIn('f32[] parameter(1)', hlo_text)
    self.assertIn('f32[128,2]{1,0} parameter(2)', hlo_text)

    # Check that the parameters are in the expected order.

    # TODO(jlebar): Ideally we'd check that hlo_proto can be deserialized to a
    # valid HLO proto, but we don't seem to have access to hlo_pb2 at the
    # moment, so the best we seem to be able to do is check that it's nonempty.
    assert hlo_proto

  def test_convert_with_constants(self):

    def fn(a, b, x, y):
      return a / b * x + y

    _, hlo_text = jax_to_hlo(
      fn,
      input_shapes=[
        ('x', xla_client.Shape('f32[128]')),
        ('y', xla_client.Shape('f32[128]')),
      ],
      constants={
        'a': 123456,
        'b': 4,
      })
    # Because we passed `a` and `b` as constants, they get constant-folded away
    # by Python/JAX to a/b = 30864.
    self.assertIn('constant(30864)', hlo_text)
    self.assertNotIn('123456', hlo_text)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
