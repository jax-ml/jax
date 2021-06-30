# Copyright 2018 Google LLC
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

from setuptools import setup, find_packages

# The following should be updated with each new jaxlib release.
_current_jaxlib_version = '0.1.68'
_available_cuda_versions = ['101', '102', '110', '111']

_dct = {}
with open('jax/version.py') as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']
_minimum_jaxlib_version = _dct['_minimum_jaxlib_version']

_libtpu_version = '0.1.dev20210615'
_libtpu_wheel_url = "https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/"\
    f"wheels/libtpu-nightly/libtpu_nightly-{_libtpu_version}-py3-none-any.whl"

import sys
def _jaxlib_wheel_url(cuda_ver):
    python_ver = f"{sys.version_info.major}{sys.version_info.minor}"
    return f"https://storage.googleapis.com/jax-releases/"\
           f"cuda{cuda_ver}/jaxlib-{_current_jaxlib_version}"\
           f"+cuda{cuda_ver}-cp{python_ver}-none-manylinux2010_x86_64.whl"

setup(
    name='jax',
    version=__version__,
    description='Differentiate, compile, and transform Numpy code.',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(exclude=["examples"]),
    package_data={'jax': ['py.typed']},
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17',
        'absl-py',
        'opt_einsum',
    ],
    extras_require={
        # Minimum jaxlib version; used in testing.
        'minimum-jaxlib': [f'jaxlib=={_minimum_jaxlib_version}'],

        # CPU-only jaxlib can be installed via:
        # $ pip install jax[cpu]
        'cpu': [f'jaxlib>={_minimum_jaxlib_version}'],

        # Cloud TPU VM jaxlib can be installed via:
        # $ pip install jax[tpu]
        'tpu': [f'jaxlib=={_current_jaxlib_version}',
                f'libtpu-nightly @ {_libtpu_wheel_url}'],

        # CUDA installations require specifying the CUDA version, e.g.
        # $ pip install jax[cuda102]
        **{f'cuda{version}': f"jaxlib @ {_jaxlib_wheel_url(version)}"
           for version in _available_cuda_versions}
    },
    url='https://github.com/google/jax',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    zip_safe=False,
)
