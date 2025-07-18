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
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
cuda_version = 0  # placeholder
project_name = f"jax-cuda{cuda_version}-plugin"
package_name = f"jax_cuda{cuda_version}_plugin"
cuda_whl_sfx = "-cu12" if cuda_version == 12 else ""

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
    description="JAX Plugin for NVIDIA GPUs",
    long_description="",
    long_description_content_type="text/markdown",
    author="JAX team",
    author_email="jax-dev@google.com",
    packages=[package_name],
    python_requires=">=3.11",
    install_requires=[f"jax-cuda{cuda_version}-pjrt=={__version__}"],
    extras_require={
      'with-cuda': [
          f"nvidia-cublas{cuda_whl_sfx}>=12.1.3.1",
          f"nvidia-cuda-cupti{cuda_whl_sfx}>=12.1.105",
          f"nvidia-cuda-nvcc{cuda_whl_sfx}>=12.6.85",
          f"nvidia-cuda-runtime{cuda_whl_sfx}>=12.1.105",
          f"nvidia-cudnn-cu{cuda_version}>=9.8,<10.0",
          f"nvidia-cufft{cuda_whl_sfx}>=11.0.2.54",
          f"nvidia-cusolver{cuda_whl_sfx}>=11.4.5.107",
          f"nvidia-cusparse{cuda_whl_sfx}>=12.1.0.106",
          f"nvidia-nccl-cu{cuda_version}>=2.18.1",
          # nvjitlink is not a direct dependency of JAX, but it is a transitive
          # dependency via, for example, cuSOLVER. NVIDIA's cuSOLVER packages
          # do not have a version constraint on their dependencies, so the
          # package doesn't get upgraded even though not doing that can cause
          # problems (https://github.com/jax-ml/jax/issues/18027#issuecomment-1756305196)
          # Until NVIDIA add version constraints, add a version constraint
          # here.
          f"nvidia-nvjitlink{cuda_whl_sfx}>=12.1.105",
          # nvrtc is a transitive and undeclared dep of cudnn.
          f"nvidia-cuda-nvrtc{cuda_whl_sfx}>=12.1.55",
          # NVSHMEM is used by Mosaic GPU collectives and can be used by XLA to
          # speed up collectives too.
          f"nvidia-nvshmem-cu{cuda_version}>=3.2.5",
      ] + (["nvidia-nvvm"] if cuda_version == 13 else []),
    },
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    package_data={
        package_name: [
            "*",
            "nvvm/libdevice/libdevice*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
