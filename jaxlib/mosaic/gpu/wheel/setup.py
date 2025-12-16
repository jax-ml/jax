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

import importlib
import os
from setuptools import setup, find_namespace_packages

__version__ = None
cuda_version = 0  # placeholder
project_name = f"mosaic_gpu-cuda{cuda_version}"
package_name = f"mosaic_gpu.mosaic_gpu_cuda{cuda_version}"

cuda_wheel_suffix = ''  # placeholder

nvidia_cublas_version = ''  # placeholder
nvidia_cuda_cupti_version = ''  # placeholder
nvidia_cuda_nvcc_version = ''  # placeholder
nvidia_cuda_runtime_version = ''  # placeholder
nvidia_cudnn_version = ''  # placeholder
nvidia_cufft_version = ''  # placeholder
nvidia_cusolver_version = ''  # placeholder
nvidia_cusparse_version = ''  # placeholder
nvidia_nccl_version = ''  # placeholder
nvidia_nvjitlink_version = ''  # placeholder
nvidia_cuda_nvrtc_version = ''  # placeholder
nvidia_nvshmem_version = ''  # placeholder

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(f"mosaic_gpu/mosaic_gpu_cuda{cuda_version}")
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
    description="Mosaic GPU Support Plugin",
    long_description="",
    long_description_content_type="text/markdown",
    author="JAX team",
    author_email="jax-dev@google.com",
    packages=packages,
    install_requires=[],
    extras_require={
      'with-cuda': [
          # Using the same deps as JAX for now - can likely be trimmed down.
          f"nvidia-cublas{cuda_wheel_suffix}{nvidia_cublas_version}",
          f"nvidia-cuda-cupti{cuda_wheel_suffix}{nvidia_cuda_cupti_version}",
          f"nvidia-cuda-nvcc{cuda_wheel_suffix}{nvidia_cuda_nvcc_version}",
          f"nvidia-cuda-runtime{cuda_wheel_suffix}{nvidia_cuda_runtime_version}",
          f"nvidia-cudnn-cu{cuda_version}{nvidia_cudnn_version}",
          f"nvidia-cufft{cuda_wheel_suffix}{nvidia_cufft_version}",
          f"nvidia-cusolver{cuda_wheel_suffix}{nvidia_cusolver_version}",
          f"nvidia-cusparse{cuda_wheel_suffix}{nvidia_cusparse_version}",
          f"nvidia-nccl-cu{cuda_version}{nvidia_nccl_version}",
          f"nvidia-nvjitlink{cuda_wheel_suffix}{nvidia_nvjitlink_version}",
          f"nvidia-cuda-nvrtc{cuda_wheel_suffix}{nvidia_cuda_nvrtc_version}",
          f"nvidia-nvshmem-cu{cuda_version}{nvidia_nvshmem_version}",
      ] + (["nvidia-nvvm"] if cuda_version == 13 else []),
    },
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    package_data={
        package_name: ["*.so"],
    },
    zip_safe=False,
    entry_points={
        "mosaic_gpu": [
            f"mosaic_gpu_cuda{cuda_version} = {package_name}",
        ],
    },
)
