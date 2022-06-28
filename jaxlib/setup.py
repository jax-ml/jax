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

from setuptools import setup
import os

__version__ = None
project_name = 'jaxlib'

with open('jaxlib/version.py') as f:
  exec(f.read(), globals())

cuda_version = os.environ.get("JAX_CUDA_VERSION")
cudnn_version = os.environ.get("JAX_CUDNN_VERSION")
if cuda_version and cudnn_version:
  __version__ += f"+cuda{cuda_version.replace('.', '')}-cudnn{cudnn_version.replace('.', '')}"

nightly = os.environ.get('JAXLIB_NIGHTLY')
if nightly:
  project_name = 'jaxlib-nightly'

setup(
    name=project_name,
    version=__version__,
    description='XLA library for JAX',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=['jaxlib', 'jaxlib.xla_extension'],
    python_requires='>=3.7',
    install_requires=['scipy>=1.5', 'numpy>=1.19', 'absl-py'],
    url='https://github.com/google/jax',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_data={
        'jaxlib': [
            '*.so',
            '*.pyd*',
            'py.typed',
            'cuda/*',
            'cuda/nvvm/libdevice/libdevice*',
            'mlir/*.py',
            'mlir/dialects/*.py',
            'mlir/transforms/*.py',
            'mlir/_mlir_libs/*.dll',
            'mlir/_mlir_libs/*.dylib',
            'mlir/_mlir_libs/*.so',
            'mlir/_mlir_libs/*.pyd',
            'rocm/*',
        ],
        'jaxlib.xla_extension': ['*.pyi'],
    },
    zip_safe=False,
)
