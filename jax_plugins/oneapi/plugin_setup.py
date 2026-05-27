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

"""Setup script for JAX OneAPI plugin package."""

import importlib
import os
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
oneapi_version = 0  # placeholder
project_name = f"jax-oneapi{oneapi_version}-plugin"
package_name = f"jax_oneapi{oneapi_version}_plugin"

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(package_name)
__version__ = _version_module._get_version_for_build()
_cmdclass = _version_module._get_cmdclass(package_name)

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description=f"JAX Plugin for Intel GPUs (OneAPI:{oneapi_version})",
    long_description="",
    long_description_content_type="text/markdown",
    author="MiniGoel",
    author_email="mini.goel@intel.com",
    packages=[package_name],
    python_requires=">=3.9",
    install_requires=[f"jax-oneapi{oneapi_version}-pjrt=={__version__}"],
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        package_name: [
            "*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
