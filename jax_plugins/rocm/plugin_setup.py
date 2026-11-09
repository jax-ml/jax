# Copyright 2024 The JAX Authors.
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

"""Setup script for JAX ROCm plugin package."""

import importlib
import os
import sys
from setuptools import setup
from setuptools.dist import Distribution

# The PEP 517 backend does not put the source root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rocm_version import DEFAULT_ROCM_PATH, detect_rocm_version

__version__ = None
rocm_version = 0  # placeholder
project_name = f"jax-rocm{rocm_version}-plugin"
package_name = f"jax_rocm{rocm_version}_plugin"

# Hermetic wheel actions expose WHEEL_VERSION_SUFFIX, but not ROCM_PATH.
rocm_path = os.getenv("ROCM_PATH", DEFAULT_ROCM_PATH)
rocm_tag = os.getenv("ROCM_VERSION_EXTRA")
rocm_detected_version = detect_rocm_version(rocm_path, rocm_tag)

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(package_name)
__version__ = _version_module._get_version_for_build()
if rocm_tag:
  __version__ = __version__ + "+rocm" + rocm_tag
_cmdclass = _version_module._get_cmdclass(package_name)

class BinaryDistribution(Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True

_rocm_min_version = "7.14" if rocm_version == 7 else str(rocm_version)
_rocm_req = f">={_rocm_min_version},<{rocm_version + 1}"

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

# rocm-sdk-libraries is host-side only; the kernels live in
# rocm-sdk-device-gfx*.
_extras_require = {
    "with-rocm": [f"rocm[libraries]{_rocm_req}"],
    "device-all": [f"rocm[device-all]{_rocm_req}"],
    **{f"device-{t}": [f"rocm[device-{t}]{_rocm_req}"] for t in _gfx_targets},
}

_description = f"JAX Plugin for AMD GPUs (ROCm:{rocm_detected_version})"

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description=_description,
    long_description=_description,
    long_description_content_type="text/plain",
    author="ROCm JAX Devs",
    author_email="dl.dl-JAX@amd.com",
    packages=[package_name],
    python_requires=">=3.12",
    install_requires=[f"jax-rocm{rocm_version}-pjrt=={__version__}"],
    url="https://github.com/jax-ml/jax",
    license="Apache-2.0",
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
    extras_require=_extras_require,
)
