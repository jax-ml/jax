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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

modules_to_skip = (
    # requires `xprof` to be installed
    "jax.collect_profile",
    # it's depending on the Mosaic GPU bindings and will fail to import
    "jax._src.lib.mosaic_gpu",
    # This is a C++ plugin that cannot be imported directly.
    "jax_plugins.xla_cuda12.xla_cuda_plugin",
    # TODO(ibarrola): Circular import, skip for the moment
    "jax._src.pallas.fuser",
    "jax._src.pallas.mosaic.lowering",
)


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments.

  Returns:
    An argparse.Namespace object containing the parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Helper for the wheel package importing verification",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--package-name", required=True, help="Name of the package to test"
  )
  parser.add_argument(
      "--pysources-path",
      required=True,
      help="Path of the manifest file containing all the modules",
  )
  return parser.parse_args()


def _get_modules_from_py_sources(
    *, pysources_path: str, package_name: str
) -> list[str]:
  """Converts file paths from a manifest to a list of module names.

  Args:
    pysources_path: Path to a manifest file containing one file path per line.
    package_name: The name of the package, used to construct module names.

  Returns:
    A list of module names (e.g., 'jax.experimental.pjit').
  """
  modules: list[str] = []
  with open(pysources_path) as f:
    for file_path in f:
      stripped_path = file_path.strip()
      if not stripped_path:
        continue

      # Paths in the manifest can be absolute from workspace root (e.g.,
      # 'third_party/py/jax/experimental/pjit.py') or relative to the package
      # (e.g., 'experimental/pjit.py'). We need to normalize them to module
      # names (e.g., 'jax.experimental.pjit').
      path_parts = stripped_path.split("/")
      try:
        # Assume the module path starts with the package name.
        start_index = path_parts.index(package_name)
        module_path = "/".join(path_parts[start_index:])
      except ValueError:
        # If the package name is not in the path, assume the path is relative
        # to the package root.
        module_path = f"{package_name}/{stripped_path}"

      if module_path.endswith("/__init__.py"):
        module_name = module_path.removesuffix("/__init__.py").replace("/", ".")
      elif module_path.endswith(".py"):
        module_name = module_path.removesuffix(".py").replace("/", ".")
      elif module_path.endswith(".so"):
        module_name = module_path.removesuffix(".so").replace("/", ".")
      else:
        continue  # Not a Python file.

      # A top-level __init__.py will result in an empty module_name, which
      # should correspond to the package itself.
      modules.append(module_name or package_name)
  return modules


def verify_wheel_imports(args: argparse.Namespace) -> None:
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
  modules = _get_modules_from_py_sources(
      pysources_path=args.pysources_path, package_name=args.package_name
  )
  failed_imports = []

  for module in modules:
    if any(module.startswith(m) for m in modules_to_skip):
      continue

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
      # The following error is caused by a missing CUDA driver in the test
      # environment. We can safely ignore it.
      if "libcuda.so" in str(e):
        logger.info("Skipping module %s due to missing CUDA driver: %s", module, e)
      else:
        logger.warning(
            "Module %s raised an exception of type %s: %s",
            module,
            type(e).__name__,
            e,
        )
        failed_imports.append(module)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Some modules define config options at import time. Since we import
      # all modules, we might try to define the same config option
      # multiple times, which raises a generic Exception. We check for the
      # error string to safely skip these import errors.
      if "already defined" in str(e):
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

  logger.info("Import of modules successful")


if __name__ == "__main__":
  verify_wheel_imports(parse_args())
