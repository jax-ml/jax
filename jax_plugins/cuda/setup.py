# Copyright 2023 The JAX Authors.
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

import importlib
import os
from setuptools import setup, find_namespace_packages

__version__ = None
cuda_version = 0  # placeholder
project_name = f"jax-cuda{cuda_version}-pjrt"
package_name = f"jax_plugins.xla_cuda{cuda_version}"

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(f"jax_plugins/xla_cuda{cuda_version}")
__version__ = _version_module._get_version_for_build()

packages = find_namespace_packages(
    include=[
        package_name,
        f"{package_name}.*",
    ]
)

setup(
    name=project_name,
    version=__version__,
    description="JAX XLA PJRT Plugin for NVIDIA GPUs",
    long_description="",
    long_description_content_type="text/markdown",
    author="JAX team",
    author_email="jax-dev@google.com",
    packages=packages,
    install_requires=[],
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    package_data={
        package_name: ["xla_cuda_plugin.so"],
    },
    zip_safe=False,
    entry_points={
        "jax_plugins": [
            f"xla_cuda{cuda_version} = {package_name}",
        ],
    },
)
