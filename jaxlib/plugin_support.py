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

from collections.abc import Sequence
import importlib
import re
from types import ModuleType
import warnings

from .version import __version__ as jaxlib_version


_PLUGIN_MODULE_NAME = {
    "cuda": "jax_cuda12_plugin",
    "rocm": "jax_rocm60_plugin",
}


def import_from_plugin(
    plugin_name: str, submodule_name: str, *, check_version: bool = True
) -> ModuleType | None:
  """Import a submodule from a known plugin with version checking.

  Args:
    plugin_name: The name of the plugin. The supported values are "cuda" or
      "rocm".
    submodule_name: The name of the submodule to import, e.g. "_triton".
    check_version: Whether to check that the plugin version is compatible with
      the jaxlib version. If the plugin is installed but the versions are not
      compatible, this function produces a warning and returns None.

  Returns:
    The imported submodule, or None if the plugin is not installed or if the
    versions are incompatible.
  """
  if plugin_name not in _PLUGIN_MODULE_NAME:
    raise ValueError(f"Unknown plugin: {plugin_name}")
  return maybe_import_plugin_submodule(
      [f".{plugin_name}", _PLUGIN_MODULE_NAME[plugin_name]],
      submodule_name,
      check_version=check_version,
  )


def check_plugin_version(
    plugin_name: str, jaxlib_version: str, plugin_version: str
) -> bool:
  # Regex to match a dotted version prefix 0.1.23.456.789 of a PEP440 version.
  # PEP440 allows a number of non-numeric suffixes, which we allow also.
  # We currently do not allow an epoch.
  version_regex = re.compile(r"[0-9]+(?:\.[0-9]+)*")

  def _parse_version(v: str) -> tuple[int, ...]:
    m = version_regex.match(v)
    if m is None:
      raise ValueError(f"Unable to parse version string '{v}'")
    return tuple(int(x) for x in m.group(0).split("."))

  if _parse_version(jaxlib_version) != _parse_version(plugin_version):
    warnings.warn(
        f"JAX plugin {plugin_name} version {plugin_version} is installed, but "
        "it is not compatible with the installed jaxlib version "
        f"{jaxlib_version}, so it will not be used.",
        RuntimeWarning,
    )
    return False
  return True


def maybe_import_plugin_submodule(
    plugin_module_names: Sequence[str],
    submodule_name: str,
    *,
    check_version: bool = True,
) -> ModuleType | None:
  for plugin_module_name in plugin_module_names:
    try:
      module = importlib.import_module(
          f"{plugin_module_name}.{submodule_name}",
          package="jaxlib",
      )
    except ImportError:
      continue
    else:
      if not check_version:
        return module
      try:
        version_module = importlib.import_module(
            f"{plugin_module_name}.version",
            package="jaxlib",
        )
      except ImportError:
        return module
      plugin_version = getattr(version_module, "__version__", "")
      if check_plugin_version(
          plugin_module_name, jaxlib_version, plugin_version
      ):
        return module
  return None
