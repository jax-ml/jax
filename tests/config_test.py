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

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
from jax._src.cloud_tpu_init import cloud_tpu_init

jax.config.parse_flags_with_absl()
os.environ['JAX_TEST_ENUM_CONFIG_FROM_ENV_VAR'] = 'yyy'
os.environ['JAX_TEST_ENUM_FLAG_FROM_ENV_VAR'] = 'yyy'


jax_test_bool_config = config.bool_state(
    name='jax_test_bool_config',
    default=True,
    help='Configuration only used for tests.',
)

jax_test_enum_config = config.enum_state(
    name='jax_test_enum_config',
    enum_values=['default', 'xxx', 'yyy'],
    default='default',
    help='Configuration only used for tests.',
)

jax_test_enum_config_via_env_var = config.enum_state(
    name='jax_test_enum_config_from_env_var',
    enum_values=['default', 'xxx', 'yyy'],
    default='default',
    help='Configuration only used for tests.',
)

jax_test_enum_flag_via_env_var = config.enum_flag(
    name='jax_test_enum_flag_from_env_var',
    enum_values=['default', 'xxx', 'yyy'],
    default='default',
    help='Configuration only used for tests.',
)


class InvalidBool:
  def __bool__(self):
    raise ValueError("invalid bool")


class ConfigTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
      {"testcase_name": "_enum", "config_name": "jax_test_enum_config",
       "config_obj": jax_test_enum_config, "default": "default", "val1": "xxx",
       "val2": "yyy"},
      {"testcase_name": "_bool", "config_name": "jax_test_bool_config",
       "config_obj": jax_test_bool_config, "default": True, "val1": False,
       "val2": True},
  )
  def test_config_setting_via_update(self, config_name, config_obj, default, val1, val2):
    self.assertEqual(config_obj.value, default)

    jax.config.update(config_name, val1)
    self.assertEqual(config_obj.value, val1)

    jax.config.update(config_name, val2)
    self.assertEqual(config_obj.value, val2)

    jax.config.update(config_name, default)
    self.assertEqual(config_obj.value, default)

  @parameterized.named_parameters(
      {"testcase_name": "_enum", "config_obj": jax_test_enum_config,
       "default": "default", "val1": "xxx", "val2": "yyy"},
      {"testcase_name": "_bool", "config_obj": jax_test_bool_config,
       "default": True, "val1": False, "val2": True},
  )
  def test_config_setting_via_context(self, config_obj, default, val1, val2):
    self.assertEqual(config_obj.value, default)

    with config_obj(val1):
      self.assertEqual(config_obj.value, val1)

      with config_obj(val2):
        self.assertEqual(config_obj.value, val2)

      self.assertEqual(config_obj.value, val1)

    self.assertEqual(config_obj.value, default)

  def test_config_update_validation(self):
    self.assertEqual(jax_test_enum_config.value, 'default')
    with self.assertRaisesRegex(ValueError, 'new enum value must be in.*'):
      jax.config.update('jax_test_enum_config', 'invalid')
    # Error should raise before changing the value
    self.assertEqual(jax_test_enum_config.value, 'default')

  def test_config_context_validation(self):
    self.assertEqual(jax_test_enum_config.value, 'default')
    with self.assertRaisesRegex(ValueError, 'new enum value must be in.*'):
      with jax_test_enum_config('invalid'):
        pass
    self.assertEqual(jax_test_enum_config.value, 'default')

  def test_bool_config_update_validation(self):
    self.assertEqual(jax_test_bool_config.value, True)
    with self.assertRaisesRegex(ValueError, "invalid bool"):
      jax.config.update('jax_test_bool_config', InvalidBool())
    # Error should raise before changing the value
    self.assertEqual(jax_test_bool_config.value, True)

  def test_bool_config_context_validation(self):
    self.assertEqual(jax_test_bool_config.value, True)
    with self.assertRaisesRegex(ValueError, "invalid bool"):
      with jax_test_bool_config(InvalidBool()):
        pass
    self.assertEqual(jax_test_bool_config.value, True)

  def test_cloud_tpu_init(self):
    if not jtu.is_cloud_tpu():
      self.skipTest('Not running on a Cloud TPU VM.')

    # Context manager resets the jax_platforms config to its original value.
    with jtu.global_config_context(jax_platforms=None):
      cloud_tpu_init()
      self.assertEqual(config.jax_platforms.value, 'tpu,cpu')

    with jtu.global_config_context(jax_platforms='platform_A'):
      cloud_tpu_init()
      self.assertEqual(config.jax_platforms.value, 'platform_A')

  def test_config_state_setting_via_env_var(self):
    self.assertEqual(jax_test_enum_config_via_env_var.value, 'yyy')

  def test_config_flag_setting_via_env_var(self):
    self.assertEqual(jax_test_enum_flag_via_env_var.value, 'yyy')

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
