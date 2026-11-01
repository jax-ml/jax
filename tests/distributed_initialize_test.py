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

import contextlib
import os
import unittest

from absl.testing import absltest
import jax
from jax._src import distributed as distributed_src
from jax._src import test_util as jtu
from jax._src.clusters import ClusterEnv, Mpi4pyCluster, SlurmCluster

try:
  import portpicker
except ImportError:
  portpicker = None

jax.config.parse_flags_with_absl()

mock = absltest.mock


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

  @jtu.skip_under_pytest("jax.distributed.initialize uses global state.")
  def test_client_round_trip(self):
    self.assertFalse(jax.distributed.is_initialized())
    self.assertIsNone(jax.distributed.client())
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f"localhost:{port}", 1, 0)
    client = jax.distributed.client()
    self.assertIs(client, distributed_src.global_state.client)
    value = "round-trip"
    client.key_value_set("distributed_client_round_trip", value)
    self.assertEqual(
        value, client.blocking_key_value_get("distributed_client_round_trip",
                                             1000))
    jax.distributed.shutdown()
    self.assertIsNone(jax.distributed.client())
    self.assertFalse(jax.distributed.is_initialized())


# Detection only reads cluster environment state. It does not call
# initialize(), so it does not need a free port.
@jtu.thread_unsafe_test_class()  # Patches process-global cluster detectors.
class DistributedDetectedClusterTest(jtu.JaxTestCase):

  def test_detected_cluster_absent(self):
    self.assertIsNone(jax.distributed.client())
    with contextlib.ExitStack() as stack:
      for env in ClusterEnv._cluster_types:
        stack.enter_context(
            mock.patch.object(env, "is_env_present", return_value=False))
      self.assertIsNone(jax.distributed.detected_cluster())
    self.assertIsNone(jax.distributed.client())
    self.assertFalse(jax.distributed.is_initialized())

  def test_detected_cluster_slurm(self):
    with contextlib.ExitStack() as stack:
      for env in ClusterEnv._cluster_types:
        if env is not SlurmCluster:
          stack.enter_context(
              mock.patch.object(env, "is_env_present", return_value=False))
      stack.enter_context(mock.patch.dict(os.environ, {
          "SLURM_JOB_ID": "7",
          "SLURM_STEP_NODELIST": "node001",
          "SLURM_NTASKS": "4",
          "SLURM_PROCID": "1",
          "SLURM_LOCALID": "0",
      }))
      self.assertEqual(jax.distributed.detected_cluster(), ("slurm", 1, 4))
    self.assertIsNone(jax.distributed.client())

  def test_detected_cluster_skips_mpi4py_opt_in(self):
    self.assertTrue(Mpi4pyCluster.opt_in_only_method)
    with contextlib.ExitStack() as stack:
      for env in ClusterEnv._cluster_types:
        stack.enter_context(mock.patch.object(
            env, "is_env_present", return_value=env is Mpi4pyCluster))
      self.assertIsNone(jax.distributed.detected_cluster())
    self.assertTrue(Mpi4pyCluster.opt_in_only_method)
    self.assertIsNone(jax.distributed.client())


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
