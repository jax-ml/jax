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

"""Verifies that all modules in a wheel are importable.

This script is designed to be run after a wheel has been installed. It
discovers all modules within a specified package and attempts to import each
one. This helps catch packaging issues where modules are missing or have
unmet dependencies.
"""

import argparse
import importlib
import logging
import pkgutil

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
  """Arguments parser."""
  parser = argparse.ArgumentParser(
    description="Helper for the wheel package importing verification",
    fromfile_prefix_chars="@",
  )
  parser.add_argument(
    "--package-name", required=True, help="Name of the package to test"
  )
  return parser.parse_args()


def _discover_modules(package_name: str) -> list[str]:
  """Discovers all modules in a package.

  Uses pkgutil.walk_packages to find all modules in a given package. It
  includes an error handler to gracefully skip any modules that cause an
  error during the discovery process.

  Args:
    package_name: The name of the package to inspect (e.g., 'jax').

  Returns:
    A sorted list of the names of all discoverable modules in the package.
  """
  modules: set[str] = set()
  package = importlib.import_module(package_name)
  if hasattr(package, "__path__"):

    def onerror(name):
      """An error handler for walk_packages to log and continue."""
      logger.warning(
        "pkgutil.walk_packages failed on module %s. Skipping.", name
      )

    for _, name, _ in pkgutil.walk_packages(
      package.__path__, package.__name__ + ".", onerror=onerror
    ):
      modules.add(name)

  return sorted(modules)


def _is_c_extension(error: str) -> bool:
  """Returns True if the import error is from a C extension."""
  return (
    "dynamic module does not define module export function" in error.lower()
  )


def _has_optional_dependency(package_name: str, error: str) -> bool:
  """Returns True if a module is trying to import an optional dependency"""
  return "no module named" in error.lower() and package_name not in error


def verify_wheel_imports(args):
  """Verifies that all modules in the specified package can be imported.

  This function first discovers all modules in the package and then attempts
  to import each one. It logs any import failures and raises a RuntimeError if
  any modules fail to import.

  Args:
    args: An argparse.Namespace object containing the parsed command-line
      arguments.

  Raises:
    RuntimeError: If any modules fail to import.
  """
  modules = _discover_modules(args.package_name)

  modules_to_skip = [
    # requires `xprof` to be installed
    "jax.collect_profile",
    # it's dependendo on the Mosaic GPU bindings and will fail to import
    "jax._src.lib.mosaic_gpu"
  ]
  failed_imports = []

  for module in modules:
    if module not in modules_to_skip:
      try:
        importlib.import_module(module)
      except ModuleNotFoundError as e:
        # If the missing module is not part of the package being tested, it's
        # an optional dependency. We can safely skip it.
        if e.name and not e.name.startswith(args.package_name):
          logger.info(
            "Skipping module %s due to optional dependency: %s", module, e
          )
        else:
          logger.warning(
            "Module %s failed with an internal import error: %s", module, e
          )
          failed_imports.append(module)
      except ImportError as e:
        error_str = str(e)
        # Some errors are expected for optional parts of JAX. We check for
        # specific error strings here because JAX may raise generic
        # ImportError exceptions for these cases.
        if _is_c_extension(error_str):
          logger.info(
            "Skipping module %s due to optional dependencies or not being"
            " importable: %s",
            module,
            e,
          )
        else:
          logger.warning(
            "Module %s failed with an internal import error: %s", module, e
          )
          failed_imports.append(module)
      except Exception as e:  # pylint: disable=broad-exception-caught
        error_str = str(e)
        # Some modules define config options at import time. Since we import
        # all modules, we might try to define the same config option
        # multiple times, which raises a generic Exception. We check for the
        # error string to safely skip these import errors.
        if "already defined" in error_str:
          logger.info(
            "Skipping module %s due to already defined config option: %s",
            module,
            e,
          )
        else:
          logger.warning(
            "Module %s raised an exception of type %s: %s",
            module,
            type(e).__name__,
            e,
          )
          failed_imports.append(module)

  if failed_imports:
    raise RuntimeError(
      f"Failed to import {len(failed_imports)}/{len(modules)} modules"
      f" modules: {failed_imports}"
    )

  logger.info("Import of modules successfull")


if __name__ == "__main__":
  verify_wheel_imports(parse_args())
