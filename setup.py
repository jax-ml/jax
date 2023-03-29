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

from distutils import spawn
import subprocess
import os
import sys

from setuptools import setup

_current_jaxlib_version = '0.4.7'
# The following should be updated with each new jaxlib release.
_latest_jaxlib_version_on_pypi = '0.4.7'
_available_cuda11_cudnn_versions = ['82', '86']
_default_cuda11_cudnn_version = '86'
_default_cuda12_cudnn_version = '88'
_libtpu_version = '0.1.dev20230327'

_dct = {}
with open('jax/version.py', encoding='utf-8') as f:
  exec(f.read(), _dct)
_minimum_jaxlib_version = _dct['_minimum_jaxlib_version']

if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
else:
  protoc = spawn.find_executable('protoc')

def generate_proto(source):
  if not protoc or not os.path.exists(source):
    return
  protoc_command = [protoc, '-I.', '--python_out=.', source]
  if subprocess.call(protoc_command) != 0:
    sys.exit(-1)

generate_proto("jax/experimental/australis/executable.proto")
generate_proto("jax/experimental/australis/petri.proto")

setup(
    extras_require={
        # Minimum jaxlib version; used in testing.
        'minimum-jaxlib': [f'jaxlib=={_minimum_jaxlib_version}'],

        # CPU-only jaxlib can be installed via:
        # $ pip install jax[cpu]
        'cpu': [f'jaxlib=={_current_jaxlib_version}'],

        # Used only for CI builds that install JAX from github HEAD.
        'ci': [f'jaxlib=={_latest_jaxlib_version_on_pypi}'],

        # Cloud TPU VM jaxlib can be installed via:
        # $ pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        'tpu': [f'jaxlib=={_current_jaxlib_version}',
                f'libtpu-nightly=={_libtpu_version}',
                # Required by cloud_tpu_init.py
                'requests'],

        # $ pip install jax[australis]
        'australis': ['protobuf>=3.13,<4'],

        # CUDA installations require adding the JAX CUDA releases URL, e.g.,
        # Cuda installation defaulting to a CUDA and Cudnn version defined above.
        # $ pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        'cuda': [f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{_default_cuda11_cudnn_version}"],

        'cuda11_pip': [
          f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{_default_cuda11_cudnn_version}",
          "nvidia-cublas-cu11",
          "nvidia-cuda-nvcc-cu11",
          "nvidia-cuda-runtime-cu11",
          "nvidia-cudnn-cu11",
          "nvidia-cufft-cu11",
          "nvidia-cusolver-cu11",
          "nvidia-cusparse-cu11",
        ],

        'cuda12_pip': [
          f"jaxlib=={_current_jaxlib_version}+cuda12.cudnn{_default_cuda12_cudnn_version}",
          "nvidia-cublas-cu12",
          "nvidia-cuda-nvcc-cu12",
          "nvidia-cuda-runtime-cu12",
          "nvidia-cudnn-cu12",
          "nvidia-cufft-cu12",
          "nvidia-cusolver-cu12",
          "nvidia-cusparse-cu12",
        ],

        # Target that does not depend on the CUDA pip wheels, for those who want
        # to use a preinstalled CUDA.
        'cuda11_local': [
          f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{_default_cuda11_cudnn_version}",
        ],
        'cuda12_local': [
          f"jaxlib=={_current_jaxlib_version}+cuda12.cudnn{_default_cuda12_cudnn_version}",
        ],

        # CUDA installations require adding jax releases URL; e.g.
        # $ pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        # $ pip install jax[cuda11_cudnn86] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        **{f'cuda11_cudnn{cudnn_version}': f"jaxlib=={_current_jaxlib_version}+cuda11.cudnn{cudnn_version}"
           for cudnn_version in _available_cuda11_cudnn_versions}
    },
)
