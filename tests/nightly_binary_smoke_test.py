# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that the nightly binary versions match the expected local overrides and the runtime initializes."""

import importlib
from importlib import metadata
import json
import os
import warnings

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb

jax.config.parse_flags_with_absl()


def _distribution_version(distribution_name: str) -> str:
  try:
    return metadata.version(distribution_name)
  except metadata.PackageNotFoundError:
    return '<not installed>'


def _module_locations(package_names: list[str]) -> dict[str, str]:
  locations = {
      'jax': getattr(jax, '__file__', '<missing>'),
  }

  package_to_module = {
      'jaxlib': 'jaxlib',
      'jax-cuda12-plugin': 'jax_cuda12_plugin',
      'jax-cuda12-pjrt': 'jax_plugins.xla_cuda12',
      'jax-cuda13-plugin': 'jax_cuda13_plugin',
      'jax-cuda13-pjrt': 'jax_plugins.xla_cuda13',
      'jax-rocm-plugin': 'jax_rocm_plugin',
      'jax-rocm-pjrt': 'jax_plugins.xla_rocm',
  }

  for package_name in package_names:
    module_name = package_to_module.get(
        package_name, package_name.replace('-', '_')
        )
    try:
      module = importlib.import_module(module_name)
      locations[package_name] = getattr(
          module,
          '__file__',
          list(getattr(module, '__path__', ['<missing>']))[0],
      )
    except Exception as exc:  # pylint: disable=broad-except
      locations[package_name] = f'<import failed: {type(exc).__name__}: {exc}>'
  return locations


@jtu.skip_under_pytest('Only intended to run under Bazel')
class NightlyBinarySmokeTest(jtu.JaxTestCase):

  def test_nightly_binary_versions_and_runtime_initialization(self):
    print('=== Nightly wheel resolution debug ===')
    print('cwd:', os.getcwd())
    print('TEST_SRCDIR:', os.environ.get('TEST_SRCDIR', '<unset>'))
    print('PYTHON_RUNFILES:', os.environ.get('PYTHON_RUNFILES', '<unset>'))

    expected_binary_version = os.environ.get('JAXCI_EXPECTED_BINARY_VERSION')
    self.assertIsNotNone(
        expected_binary_version,
        'JAXCI_EXPECTED_BINARY_VERSION must be set for the nightly guard.',
    )

    package_names = [
        name
        for name in os.environ.get('JAXCI_EXPECTED_BINARY_PACKAGES', '').split(',')
        if name
    ]
    self.assertNotEmpty(
        package_names,
        'JAXCI_EXPECTED_BINARY_PACKAGES must not be empty.',
    )

    distribution_versions = {
        package_name: _distribution_version(package_name)
        for package_name in package_names
    }
    print('Nightly binary smoke test distribution versions:')
    print(json.dumps(distribution_versions, indent=2, sort_keys=True))
    print('Nightly binary smoke test module locations:')
    print(json.dumps(_module_locations(package_names), indent=2, sort_keys=True))

    missing_distributions = [
        package_name
        for package_name, version in distribution_versions.items()
        if version == '<not installed>'
    ]
    self.assertEmpty(
        missing_distributions,
        'Expected nightly binary distributions are not installed: '
        f'{missing_distributions}',
    )

    mismatched_versions = {
        package_name: version
        for package_name, version in distribution_versions.items()
        if version != expected_binary_version
    }
    self.assertEmpty(
        mismatched_versions,
        'Installed binary distributions do not match the nightly override '
        f'version {expected_binary_version}: {mismatched_versions}',
    )

    warning_payload = []
    runtime_payload = {}
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')

      try:
        runtime_payload['default_backend'] = jax.default_backend()
      except Exception as exc:
        self.fail(
            'jax runtime initialization failed even though package metadata'
            ' matched the expected nightly binary version'
            f' {expected_binary_version}: {exc}'
        )

      backend = runtime_payload['default_backend']
      runtime_payload['device_count'] = jax.device_count()
      runtime_payload['devices'] = [str(device) for device in jax.devices()]
      try:
        runtime_payload[f'{backend}_devices'] = [
            str(device) for device in jax.devices(backend)
        ]
      except Exception as exc:  # pylint: disable=broad-except
        runtime_payload[f'{backend}_devices_error'] = f'{type(exc).__name__}: {exc}'
      runtime_payload['backend_keys'] = sorted(xb.backends().keys())
      runtime_payload['backend_errors'] = {
          key: str(value) for key, value in xb._backend_errors.items()
      }
      for warning in caught:
        warning_payload.append({
            'category': warning.category.__name__,
            'message': str(warning.message),
        })

    print('Nightly binary smoke test runtime payload:')
    print(json.dumps(runtime_payload, indent=2, sort_keys=True))
    print('Nightly binary smoke test captured warnings:')
    print(json.dumps(warning_payload, indent=2, sort_keys=True))

    self.assertNotEmpty(jax.devices())


if __name__ == '__main__':
  absltest.main()
