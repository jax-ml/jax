# Copyright 2018 The JAX Authors.
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

from setuptools import setup, find_packages

project_name = 'jax'

_current_jaxlib_version = '0.6.1'
# The following should be updated after each new jaxlib release.
_latest_jaxlib_version_on_pypi = '0.6.1'

_libtpu_version = '0.0.15.*'

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()
_jax_version = _version_module._version  # JAX version, with no .dev suffix.
_cmdclass = _version_module._get_cmdclass(project_name)
_minimum_jaxlib_version = _version_module._minimum_jaxlib_version

# If this is a pre-release ("rc" wheels), append "rc0" to
# _minimum_jaxlib_version and _current_jaxlib_version so that we are able to
# install the rc wheels.
if _version_module._is_prerelease():
  _minimum_jaxlib_version += "rc0"
  _current_jaxlib_version += "rc0"

with open('README.md', encoding='utf-8') as f:
  _long_description = f.read()

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description='Differentiate, compile, and transform Numpy code.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(exclude=["examples"]),
    package_data={'jax': ['py.typed', "*.pyi", "**/*.pyi"]},
    python_requires='>=3.10',
    install_requires=[
        f'jaxlib >={_minimum_jaxlib_version}, <={_jax_version}',
        'ml_dtypes>=0.5.0',
        'numpy>=1.26',
        'opt_einsum',
        'scipy>=1.12',
    ],
    extras_require={
        # Minimum jaxlib version; used in testing.
        'minimum-jaxlib': [f'jaxlib=={_minimum_jaxlib_version}'],

        # A CPU-only jax doesn't require any extras, but we keep this extra
        # around for compatibility.
        'cpu': [],

        # Used only for CI builds that install JAX from github HEAD.
        'ci': [f'jaxlib=={_latest_jaxlib_version_on_pypi}'],

        # Cloud TPU VM jaxlib can be installed via:
        # $ pip install "jax[tpu]"
        'tpu': [
          f'jaxlib>={_current_jaxlib_version},<={_jax_version}',
          f'libtpu=={_libtpu_version}',
          'requests',  # necessary for jax.distributed.initialize
        ],

        'cuda': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin[with-cuda]>={_current_jaxlib_version},<={_jax_version}",
        ],

        'cuda12': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin[with-cuda]>={_current_jaxlib_version},<={_jax_version}",
        ],

        # Target that does not depend on the CUDA pip wheels, for those who want
        # to use a preinstalled CUDA.
        'cuda12-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin>={_current_jaxlib_version},<={_jax_version}",
        ],

        # ROCm support for ROCm 6.0 and above.
        'rocm': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-rocm60-plugin>={_current_jaxlib_version},<={_jax_version}",
        ],

        # For automatic bootstrapping distributed jobs in Kubernetes
        'k8s': [
          'kubernetes',
        ],
    },
    url='https://github.com/jax-ml/jax',
    license='Apache-2.0',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    zip_safe=False,
)
