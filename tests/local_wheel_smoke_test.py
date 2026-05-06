# Copyright 2026 The JAX Authors.
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

"""Smoke-tests JAX wheel resolution for Bazel jobs.

Ensures the local wheel overrides are used at runtime, instead of ones from
matching lock file entries, if any such entries were present
during the resolution.
"""

from importlib import metadata
import json
import os

from absl.testing import absltest
import jax
from jax._src import test_util as jtu


jax.config.parse_flags_with_absl()

_EXPECTED_VERSIONS_ENV = 'JAXCI_EXPECTED_WHEEL_VERSIONS_JSON'
_NOT_INSTALLED = '<not installed>'


def _distribution_version(distribution_name: str) -> str:
  try:
    return metadata.version(distribution_name)
  except metadata.PackageNotFoundError:
    return _NOT_INSTALLED


@jtu.skip_under_pytest("Exclusive to Bazel's hermetic Python setup process.")
class LocalWheelSmokeTest(jtu.JaxTestCase):

  def test_local_wheel_versions(self):
    print('=== Local wheel resolution ===')

    expected_versions_json = os.environ.get(_EXPECTED_VERSIONS_ENV)
    self.assertIsNotNone(
        expected_versions_json,
        f'{_EXPECTED_VERSIONS_ENV} must be set for the smoke test.',
    )

    expected_versions: dict[str, str] | None = None
    try:
      expected_versions = json.loads(expected_versions_json)
    except json.JSONDecodeError as exc:
      self.fail(
          f'Could not parse {_EXPECTED_VERSIONS_ENV} as JSON: {exc}\n'
          f'Raw value: {expected_versions_json}'
      )

    self.assertIsInstance(expected_versions, dict)
    self.assertNotEmpty(
        expected_versions,
        f'{_EXPECTED_VERSIONS_ENV} must not be empty.',
    )

    invalid_entries = {
        package_name: version
        for package_name, version in expected_versions.items()
        if not isinstance(package_name, str) or not isinstance(version, str)
    }
    self.assertEmpty(
        invalid_entries,
        'Expected package-version map must contain only '
        f'string keys and values: {invalid_entries}',
    )

    expected_package_names = sorted(expected_versions)
    distribution_versions = {
        package_name: _distribution_version(package_name)
        for package_name in expected_package_names
    }
    print('Expected wheel versions:')
    print(json.dumps(expected_versions, indent=4, sort_keys=True))
    print('Installed distribution versions:')
    print(json.dumps(distribution_versions, indent=4, sort_keys=True))

    missing_distributions = [
        package_name
        for package_name, version in distribution_versions.items()
        if version == _NOT_INSTALLED
    ]
    self.assertEmpty(
        missing_distributions,
        'Expected distributions are not installed: '
        f'{missing_distributions}',
    )

    mismatched_versions = {
        package_name: {
            'expected': expected_versions[package_name],
            'actual': actual_version,
        }
        for package_name, actual_version in distribution_versions.items()
        if actual_version != expected_versions[package_name]
    }
    self.assertEmpty(
        mismatched_versions,
        'Installed distributions do not match the expected wheel versions: '
        f'{mismatched_versions}',
    )


if __name__ == '__main__':
  absltest.main()
