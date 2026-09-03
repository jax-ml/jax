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

import os
import unittest

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.lib import jaxlib_extension_version

try:
  import portpicker
except ImportError:
  portpicker = None

jax.config.parse_flags_with_absl()


@unittest.skipIf(not portpicker, "Test requires portpicker")
@jtu.thread_unsafe_test_class()  # jax.distributed uses global state.
class DistributedInitializeTest(jtu.JaxTestCase):

  def tearDown(self):
    jax.distributed.shutdown()
    super().tearDown()

  @jtu.skip_under_pytest(
      """Side effects from jax.distributed.initialize conflict with other tests
      in the same process. pytest runs multiple tests in the same process."""
  )
  def test_is_distributed_initialized(self):
    port = portpicker.pick_unused_port()
    self.assertFalse(jax.distributed.is_initialized())
    jax.distributed.initialize(f"localhost:{port}", 1, 0)
    self.assertTrue(jax.distributed.is_initialized())

  @jtu.skip_under_pytest("jax.distributed.initialize uses global state.")
  @unittest.skipIf(jaxlib_extension_version < 483, "Requires jaxlib 0.11.2")
  def test_mtls(self):
    # testdata/mtls_cert.pem is a self-signed certificate for localhost that
    # serves as both this process' identity and the CA bundle. Generated with:
    #   openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
    #     -keyout mtls_key.pem -out mtls_cert.pem -days 36500 \
    #     -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
    testdata = os.path.join(os.path.dirname(__file__), "testdata")
    cert_file = os.path.join(testdata, "mtls_cert.pem")
    key_file = os.path.join(testdata, "mtls_key.pem")
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f"localhost:{port}", 1, 0,
                               mtls_cert_file=cert_file,
                               mtls_key_file=key_file,
                               mtls_ca_file=cert_file,
                               verify_secure_credentials=True)
    self.assertTrue(jax.distributed.is_initialized())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
