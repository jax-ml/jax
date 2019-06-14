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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from jax.tools.jax_to_hlo import jax_to_hlo
from jaxlib import xla_client


class JaxToHloTest(absltest.TestCase):

  def test_convert_axpy(self):

    def axpy(a, x, y):
      return a * x + y

    hlo_proto, hlo_text = jax_to_hlo(
        axpy, {
            'a': xla_client.Shape('f32[]'),
            'x': xla_client.Shape('f32[128]'),
            'y': xla_client.Shape('f32[128]'),
        })

    # Check that hlo_text contains a broadcast, add, and multiply.
    self.assertIn('broadcast', hlo_text)
    self.assertIn('add', hlo_text)
    self.assertIn('multiply', hlo_text)

    # TODO(jlebar): Ideally we'd check that hlo_proto can be deserialized to a
    # valid HLO proto, but we don't seem to have access to hlo_pb2 at the
    # moment, so the best we seem to be able to do is check that it's nonempty.
    assert hlo_proto


if __name__ == '__main__':
  absltest.main()
