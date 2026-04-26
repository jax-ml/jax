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

import importlib
from importlib import metadata
import json
import os
from typing import Any
import warnings

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src import xla_bridge


jax.config.parse_flags_with_absl()

_EXPECTED_VERSIONS_ENV = "JAXCI_EXPECTED_WHEEL_VERSIONS_JSON"
_NOT_INSTALLED = "<not installed>"


def _distribution_version(distribution_name: str) -> str:
  try:
    return metadata.version(distribution_name)
  except metadata.PackageNotFoundError:
    return _NOT_INSTALLED


def _module_locations(package_names: list[str]) -> dict[str, str]:
  locations = {
      "jax": getattr(jax, "__file__", "<missing>"),
  }
  package_to_module = {
      "jaxlib": "jaxlib",
      "jax-cuda12-plugin": "jax_cuda12_plugin",
      "jax-cuda12-pjrt": "jax_plugins.xla_cuda12",
      "jax-cuda13-plugin": "jax_cuda13_plugin",
      "jax-cuda13-pjrt": "jax_plugins.xla_cuda13",
      "jax-rocm-plugin": "jax_rocm_plugin",
      "jax-rocm-pjrt": "jax_plugins.xla_rocm",
  }

  for package_name in package_names:
    if package_name == "jax":
      continue

    module_name = package_to_module.get(
      package_name, package_name.replace("-", "_")
    )
    try:
      module = importlib.import_module(module_name)
      module_file = getattr(module, "__file__", None)
      if module_file is not None:
        locations[package_name] = module_file
      else:
        module_paths = list(getattr(module, "__path__", []))
        locations[package_name] = (
          module_paths[0] if module_paths else "<missing>"
        )
    except Exception as exc:  # pylint: disable=broad-except
      locations[package_name] = (
          f"<import failed: {type(exc).__name__}: {exc}>"
      )
  return locations


@jtu.skip_under_pytest("Only intended to run under Bazel")
class LocalWheelSmokeTest(jtu.JaxTestCase):

  def test_local_wheel_versions_and_runtime_initialization(self):
    print("=== Local wheel resolution debug ===")

    expected_versions_json = os.environ.get(_EXPECTED_VERSIONS_ENV)
    self.assertIsNotNone(
        expected_versions_json,
        f"{_EXPECTED_VERSIONS_ENV} must be set for the smoke test.",
    )

    expected_versions: dict[str, str] | None = None
    try:
      expected_versions = json.loads(expected_versions_json)
    except json.JSONDecodeError as exc:
      self.fail(
          f"Could not parse {_EXPECTED_VERSIONS_ENV} as JSON: {exc}\n"
          f"Raw value: {expected_versions_json}"
      )

    self.assertIsInstance(expected_versions, dict)
    self.assertNotEmpty(
        expected_versions,
        f"{_EXPECTED_VERSIONS_ENV} must not be empty.",
    )

    invalid_entries = {
        package_name: version
        for package_name, version in expected_versions.items()
        if not isinstance(package_name, str) or not isinstance(version, str)
    }
    self.assertEmpty(
        invalid_entries,
        "Expected package-version map must contain only "
        f"string keys and values: {invalid_entries}",
    )

    expected_package_names = sorted(expected_versions)
    distribution_versions = {
        package_name: _distribution_version(package_name)
        for package_name in expected_package_names
    }
    print("Expected wheel versions:")
    print(json.dumps(expected_versions, indent=4, sort_keys=True))
    print("Installed distribution versions:")
    print(json.dumps(distribution_versions, indent=4, sort_keys=True))
    print("Module locations:")
    print(
        json.dumps(
            _module_locations(expected_package_names),
            indent=4,
            sort_keys=True,
        )
    )

    missing_distributions = [
        package_name
        for package_name, version in distribution_versions.items()
        if version == _NOT_INSTALLED
    ]
    self.assertEmpty(
        missing_distributions,
        "Expected distributions are not installed: "
        f"{missing_distributions}",
    )

    mismatched_versions = {
        package_name: {
            "expected": expected_versions[package_name],
            "actual": actual_version,
        }
        for package_name, actual_version in distribution_versions.items()
        if actual_version != expected_versions[package_name]
    }
    self.assertEmpty(
        mismatched_versions,
        "Installed distributions do not match the expected wheel versions: "
        f"{mismatched_versions}",
    )

    runtime_payload: dict[str, Any] = {}
    warning_payload: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")

      try:
        runtime_payload["default_backend"] = jax.default_backend()
      except Exception as exc:
        self.fail(
            "JAX runtime initialization failed after the expected "
            f"wheel versions were resolved: {type(exc).__name__}: {exc}"
        )

      runtime_payload["device_count"] = jax.device_count()
      runtime_payload["devices"] = [str(device) for device in jax.devices()]
      try:
        runtime_payload["gpu_devices"] = [
            str(device) for device in jax.devices("gpu")
        ]
      except Exception as exc:  # pylint: disable=broad-except
        runtime_payload["gpu_devices_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
      runtime_payload["backend_keys"] = sorted(xla_bridge.backends().keys())
      runtime_payload["backend_errors"] = {
          key: str(value)
          for key, value in getattr(xla_bridge, "_backend_errors", {}).items()
      }
      warning_payload = [
          {
              "category": warning.category.__name__,
              "message": str(warning.message),
          }
          for warning in caught
      ]

    print("Runtime payload:")
    print(json.dumps(runtime_payload, indent=4, sort_keys=True))
    print("Captured warnings:")
    print(json.dumps(warning_payload, indent=4, sort_keys=True))

    self.assertEqual(runtime_payload["default_backend"], "gpu")
    self.assertNotEmpty(runtime_payload.get("gpu_devices", []))


if __name__ == "__main__":
  absltest.main()
