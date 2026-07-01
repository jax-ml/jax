# Copyright 2026 The JAX Authors.
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

"""Setup script for JAX OneAPI plugin package."""

import importlib
import os
from setuptools import setup
from setuptools.dist import Distribution

__version__ = None
project_name = "jax-oneapi-plugin"
package_name = "jax_oneapi_plugin"

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(package_name)
__version__ = _version_module._get_version_for_build()
_cmdclass = _version_module._get_cmdclass(package_name)

# oneAPI runtime dependency pins
dpcpp_cpp_rt_version = "==2025.1.1"
intel_sycl_rt_version = "==2025.1.1"
tbb_version = "==2022.1.0"
intel_cmplr_lib_rt_version = "==2025.1.1"
intel_cmplr_lib_ur_version = "==2025.1.1"
intel_cmplr_lic_rt_version = "==2025.1.1"
intel_opencl_rt_version = "==2025.1.1"
intel_openmp_version = "==2025.1.1"
impi_rt_version = "==2021.15.0"
intel_pti_version = "==0.12.0"
mkl_version = "==2025.1.0"
onemkl_sycl_blas_version = "==2025.1.0"
onemkl_sycl_dft_version = "==2025.1.0"
onemkl_sycl_lapack_version = "==2025.1.0"
onemkl_sycl_rng_version = "==2025.1.0"
onemkl_sycl_sparse_version = "==2025.1.0"
umf_version = "==0.10.0"
tcmlib_version = "==1.3.0"

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description="JAX Plugin for Intel GPUs",
    long_description="",
    long_description_content_type="text/markdown",
    author="MiniGoel",
    author_email="mini.goel@intel.com",
    packages=[package_name],
    python_requires=">=3.12",
    install_requires=[f"jax-oneapi-pjrt=={__version__}"],
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
    extras_require={
        "with-oneapi": [
            f"dpcpp-cpp-rt{dpcpp_cpp_rt_version}",
            f"intel-sycl-rt{intel_sycl_rt_version}",
            f"tbb{tbb_version}",
            f"intel-cmplr-lib-rt{intel_cmplr_lib_rt_version}",
            f"intel-cmplr-lib-ur{intel_cmplr_lib_ur_version}",
            f"intel-cmplr-lic-rt{intel_cmplr_lic_rt_version}",
            f"intel-opencl-rt{intel_opencl_rt_version}",
            f"intel-openmp{intel_openmp_version}",
            f"impi-rt{impi_rt_version}",
            f"intel-pti{intel_pti_version}",
            f"mkl{mkl_version}",
            f"onemkl-sycl-blas{onemkl_sycl_blas_version}",
            f"onemkl-sycl-dft{onemkl_sycl_dft_version}",
            f"onemkl-sycl-lapack{onemkl_sycl_lapack_version}",
            f"onemkl-sycl-rng{onemkl_sycl_rng_version}",
            f"onemkl-sycl-sparse{onemkl_sycl_sparse_version}",
            f"umf{umf_version}",
            f"tcmlib{tcmlib_version}",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    package_data={
        package_name: [
            "*",
        ],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
