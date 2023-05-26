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

from setuptools import setup
from setuptools.dist import Distribution
import os

__version__ = None
project_name = 'jaxlib'

with open('jaxlib/version.py') as f:
  exec(f.read(), globals())

with open('README.md') as f:
  _long_description = f.read()

cuda_version = os.environ.get("JAX_CUDA_VERSION")
cudnn_version = os.environ.get("JAX_CUDNN_VERSION")
if cuda_version and cudnn_version:
  __version__ += f"+cuda{cuda_version.replace('.', '')}-cudnn{cudnn_version.replace('.', '')}"

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    description='XLA library for JAX',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=['jaxlib', 'jaxlib.xla_extension'],
    python_requires='>=3.8',
    install_requires=['scipy>=1.7', 'numpy>=1.21', 'ml_dtypes>=0.1.0'],
    extras_require={
      'cuda11_pip': [
        "nvidia-cublas-cu11>=11.11",
        "nvidia-cuda-cupti-cu11>=11.8",
        "nvidia-cuda-nvcc-cu11>=11.8",
        "nvidia-cuda-runtime-cu11>=11.8",
        "nvidia-cudnn-cu11>=8.8",
        "nvidia-cufft-cu11>=10.9",
        "nvidia-cusolver-cu11>=11.4",
        "nvidia-cusparse-cu11>=11.7",
      ],
      'cuda12_pip': [
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12>=8.9",
        "nvidia-cufft-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
      ],
    },
    url='https://github.com/google/jax',
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={
        'jaxlib': [
            '*.so',
            '*.pyd*',
            'py.typed',
            'cpu/*',
            'cuda/*',
            'cuda/nvvm/libdevice/libdevice*',
            'mlir/*.py',
            'mlir/dialects/*.py',
            'mlir/_mlir_libs/*.dll',
            'mlir/_mlir_libs/*.dylib',
            'mlir/_mlir_libs/*.so',
            'mlir/_mlir_libs/*.pyd',
            'mlir/_mlir_libs/*.py',
            'rocm/*',
        ],
        'jaxlib.xla_extension': ['*.pyi'],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
