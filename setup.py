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
_current_jaxlib_version = '0.1.75'
_available_cuda_versions = ['11']
_default_cuda_version = '11'
_available_cudnn_versions = ['82', '805']
_default_cudnn_version = '82'
_libtpu_version = '0.1.dev20211208'

_dct = {}
with open('jax/version.py') as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']
_minimum_jaxlib_version = _dct['_minimum_jaxlib_version']

setup(
    name='jax',
    version=__version__,
    description='Differentiate, compile, and transform Numpy code.',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(exclude=["examples"]),
    package_data={'jax': ['py.typed']},
    python_requires='>=3.7',
    install_requires=[
        'absl-py',
        'numpy>=1.18',
        'opt_einsum',
        'scipy>=1.2.1',
        'typing_extensions',
    ],
    extras_require={
        # Minimum jaxlib version; used in testing.
        'minimum-jaxlib': [f'jaxlib=={_minimum_jaxlib_version}'],

        # CPU-only jaxlib can be installed via:
        # $ pip install jax[cpu]
        'cpu': [f'jaxlib=={_current_jaxlib_version}'],

        # Cloud TPU VM jaxlib can be installed via:
        # $ pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/jax_releases.html
        'tpu': [f'jaxlib=={_current_jaxlib_version}',
                f'libtpu-nightly=={_libtpu_version}',
                # Required by cloud_tpu_init.py
                'requests'],

        # CUDA installations require adding jax releases URL; e.g.
        # Cuda installation defaulting to a CUDA and Cudnn version defined above.
        # $ pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_releases.html
        'cuda': [f"jaxlib=={_current_jaxlib_version}+cuda{_default_cuda_version}.cudnn{_default_cudnn_version}"],

        # CUDA installations require adding jax releases URL; e.g.
        # $ pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
        # $ pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html
        **{f'cuda{cuda_version}_cudnn{cudnn_version}': f"jaxlib=={_current_jaxlib_version}+cuda{cuda_version}.cudnn{cudnn_version}"
           for cuda_version in _available_cuda_versions for cudnn_version in _available_cudnn_versions}
    },
    url='https://github.com/google/jax',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    zip_safe=False,
)
