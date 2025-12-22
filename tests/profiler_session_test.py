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

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp

_TEST_SESSION_ID = 'my_custom_session_123'


@jtu.thread_unsafe_test_class()
class ProfilerSessionTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    # Ensure that any running profiler is stopped before starting the test.
    # This is in setUp rather than tearDown to defend against previous tests
    # that may have crashed or failed to clean up properly.
    try:
      jax.profiler.stop_trace()
    except RuntimeError:
      pass

  @parameterized.named_parameters(
      dict(testcase_name='without_session_id', session_id=None),
      dict(testcase_name='with_empty_session_id', session_id=''),
      dict(testcase_name='with_custom_session_id', session_id=_TEST_SESSION_ID),
  )
  def test_programmatic_profiling(self, session_id: str | None):
    tmpdir = pathlib.Path(self.create_tempdir())

    options = jax.profiler.ProfileOptions()
    if session_id is not None:
      options.session_id = session_id

    with jax.profiler.trace(tmpdir, profiler_options=options):
      jax.pmap(lambda x: jax.lax.psum(x + 1, 'i'), axis_name='i')(
          jnp.ones(jax.local_device_count())
      ).block_until_ready()

    profile_plugin_dir = tmpdir / 'plugins' / 'profile'
    self.assertTrue(profile_plugin_dir.exists(), f'Not found at {profile_plugin_dir}')

    subdirs = [x.name for x in profile_plugin_dir.iterdir() if x.is_dir()]
    self.assertLen(subdirs, 1)

    if session_id is None or not session_id:
      self.assertNotIn(_TEST_SESSION_ID, subdirs)
      self.assertNotIn('', subdirs)
      target_dir = subdirs[0]
    else:
      self.assertIn(session_id, subdirs)
      target_dir = session_id

    session_dir = profile_plugin_dir / target_dir
    pb_files = list(session_dir.glob('*.xplane.pb'))
    self.assertNotEmpty(pb_files, f'No .xplane.pb files found in {session_dir}')


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
