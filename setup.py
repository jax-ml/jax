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

_current_jaxlib_version = '0.11.2'

# The following should be updated after each new jaxlib release.
_latest_jaxlib_version_on_pypi = '0.11.2'

_libtpu_version = '0.0.50.*'

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

# GPU targets for which TheRock publishes a rocm-sdk-device-gfx* wheel. The
# same set is on the ROCm 7 and ROCm 10 indexes.
_gfx_targets = (
    "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
    "gfx1036",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201", "gfx1250",
)

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description='Differentiate, compile, and transform Numpy code.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(include=["jax", "jax.*"]),
    package_data={'jax': ['py.typed', "*.pyi", "**/*.pyi"]},
    python_requires='>=3.12',
    install_requires=[
        f'jaxlib >={_minimum_jaxlib_version}, <={_jax_version}',
        'ml_dtypes>=0.5.0',
        'numpy>=2.2',
        'opt_einsum',
        'scipy>=1.15',
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

        'cuda13': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda13-plugin[with-cuda]>={_current_jaxlib_version},<={_jax_version}",
        ],

        # Target that does not depend on the CUDA pip wheels, for those who want
        # to use a preinstalled CUDA.
        'cuda12-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin>={_current_jaxlib_version},<={_jax_version}",
        ],

        'cuda13-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda13-plugin>={_current_jaxlib_version},<={_jax_version}",
        ],

        # TheRock ROCm wheels are not on PyPI; pass --extra-index-url for the
        # stable AMD index of that ROCm line, alongside PyPI rather than
        # replacing it:
        #   rocm / rocm7: https://repo.amd.com/rocm/whl-multi-arch/
        #   rocm10:       https://stable.repo.amd.com/rocm/whl-next/
        # These three pull device code for every target. Naming one instead,
        # as in jax[rocm10-device-gfx950], downloads only that target.
        # TODO(gulsumgudukbay): add a rocm8 extra once those wheels ship.
        'rocm': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-rocm7-plugin[with-rocm,device-all]=={_jax_version}.*",
        ],

        'rocm7': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-rocm7-plugin[with-rocm,device-all]=={_jax_version}.*",
        ],

        'rocm10': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-rocm10-plugin[with-rocm,device-all]=={_jax_version}.*",
        ],

        # The same, per GPU target: jax[rocm7-device-gfx950] and so on.
        **{
          f'rocm{line}-device-{target}': [
            f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
            f"jax-rocm{line}-plugin[with-rocm,device-{target}]"
            f"=={_jax_version}.*",
          ]
          for line in ('7', '10') for target in _gfx_targets
        },

        # Preinstalled ROCm, typically /opt/rocm. Does not pull TheRock wheels.
        'rocm7-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-rocm7-plugin=={_jax_version}.*",
        ],

        'oneapi': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-oneapi-plugin[with-oneapi]>={_current_jaxlib_version},<={_jax_version}",
        ],

        # For automatic bootstrapping distributed jobs in Kubernetes
        'k8s': [
          'kubernetes',
        ],

        # For including XProf server
        'xprof': [
          'xprof',
        ],
    },
    url='https://github.com/jax-ml/jax',
    license='Apache-2.0',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    zip_safe=False,
)
