# Copyright 2020 Google LLC
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

# Script that builds a jaxlib wheel, intended to be run via bazel run as part
# of the jaxlib build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import glob
import os
import platform
import shutil
import subprocess
import sys
import tempfile

from bazel_tools.tools.python.runfiles import runfiles

parser = argparse.ArgumentParser()
parser.add_argument(
  "--sources_path",
  default=None,
  help="Path in which the wheel's sources should be prepared. Optional. If "
       "omitted, a temporary directory will be used.")
parser.add_argument(
  "--output_path",
  default=None,
  required=True,
  help="Path to which the output wheel should be written. Required.")
args = parser.parse_args()

r = runfiles.Create()


def _is_windows():
  return sys.platform.startswith("win32")


def _copy_so(src_file, dst_dir, dst_filename=None):
  src_filename = os.path.basename(src_file)
  if not dst_filename:
    if _is_windows() and src_filename.endswith(".so"):
      dst_filename = src_filename[:-3] + ".pyd"
    else:
      dst_filename = src_filename
  dst_file = os.path.join(dst_dir, dst_filename)
  if _is_windows():
    shutil.copyfile(src_file, dst_file)
  else:
    shutil.copy(src_file, dst_file)


def _copy_normal(src_file, dst_dir, dst_filename=None):
  src_filename = os.path.basename(src_file)
  dst_file = os.path.join(dst_dir, dst_filename or src_filename)
  if _is_windows():
    shutil.copyfile(src_file, dst_file)
  else:
    shutil.copy(src_file, dst_file)


def copy_file(src_file, dst_dir, dst_filename=None):
  if src_file.endswith(".so"):
    _copy_so(src_file, dst_dir, dst_filename=dst_filename)
  else:
    _copy_normal(src_file, dst_dir, dst_filename=dst_filename)


_XLA_EXTENSION_STUBS = [
    "__init__.pyi",
    "jax_jit.pyi",
    "ops.pyi",
    "outfeed_receiver.pyi",
    "pmap_lib.pyi",
    "profiler.pyi",
    "pytree.pyi",
]


def patch_copy_xla_extension_stubs(dst_dir):
  # This file is required by PEP-561. It marks jaxlib as package containing
  # type stubs.
  with open(os.path.join(dst_dir, "py.typed"), "w"):
    pass
  # The -stubs suffix is required by PEP-561.
  xla_extension_dir = os.path.join(dst_dir, "xla_extension-stubs")
  os.makedirs(xla_extension_dir)
  # Create a dummy __init__.py to convince setuptools that
  # xla_extension-stubs is a package.
  with open(os.path.join(xla_extension_dir, "__init__.py"), "w"):
    pass
  for stub_name in _XLA_EXTENSION_STUBS:
    with open(r.Rlocation(
        "org_tensorflow/tensorflow/compiler/xla/python/xla_extension/" + stub_name)) as f:
      src = f.read()
    src = src.replace(
        "from tensorflow.compiler.xla.python import xla_extension",
        "from .. import xla_extension"
    )
    with open(os.path.join(xla_extension_dir, stub_name), "w") as f:
      f.write(src)


def patch_copy_xla_client_py(dst_dir):
  with open(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_client.py")) as f:
    src = f.read()
    src = src.replace("from tensorflow.compiler.xla.python import xla_extension as _xla",
                      "from . import xla_extension as _xla")
    with open(os.path.join(dst_dir, "xla_client.py"), "w") as f:
      f.write(src)


def patch_copy_tpu_client_py(dst_dir):
  with open(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.py")) as f:
    src = f.read()
    src = src.replace("from tensorflow.compiler.xla.python import xla_extension as _xla",
                      "from . import xla_extension as _xla")
    src = src.replace("from tensorflow.compiler.xla.python import xla_client",
                      "from . import xla_client")
    src = src.replace(
        "from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client",
        "from . import tpu_client_extension as _tpu_client")
    with open(os.path.join(dst_dir, "tpu_client.py"), "w") as f:
      f.write(src)

def verify_mac_libraries_dont_reference_chkstack():
  """Verifies that xla_extension.so doesn't depend on ____chkstk_darwin.

  We don't entirely know why this happens, but in some build environments
  we seem to target the wrong Mac OS version.
  https://github.com/google/jax/issues/3867

  This check makes sure we don't release wheels that have this dependency.
  """
  if platform.system() != "Darwin":
    return
  nm = subprocess.run(
    ["nm", "-g",
     r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_extension.so")
     ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
  if nm.returncode != 0:
    raise RuntimeError(f"nm process failed: {nm.stdout} {nm.stderr}")
  if "____chkstk_darwin" in nm.stdout:
      raise RuntimeError(
        "Mac wheel incorrectly depends on symbol ____chkstk_darwin, which "
        "means that it isn't compatible with older MacOS versions.")


def prepare_wheel(sources_path):
  """Assembles a source tree for the wheel in `sources_path`."""
  jaxlib_dir = os.path.join(sources_path, "jaxlib")
  os.makedirs(jaxlib_dir)
  copy_to_jaxlib = functools.partial(copy_file, dst_dir=jaxlib_dir)

  verify_mac_libraries_dont_reference_chkstack()
  copy_to_jaxlib(r.Rlocation("__main__/build/LICENSE.txt"),
                 dst_dir=sources_path)
  copy_file(r.Rlocation("__main__/jaxlib/setup.py"), dst_dir=sources_path)
  copy_file(r.Rlocation("__main__/jaxlib/setup.cfg"), dst_dir=sources_path)
  copy_to_jaxlib(r.Rlocation("__main__/jaxlib/init.py"),
                 dst_filename="__init__.py")
  copy_to_jaxlib(r.Rlocation("__main__/jaxlib/lapack.so"))
  copy_to_jaxlib(r.Rlocation("__main__/jaxlib/_pocketfft.so"))
  copy_to_jaxlib(r.Rlocation("__main__/jaxlib/pocketfft_flatbuffers_py_generated.py"))
  copy_to_jaxlib(r.Rlocation("__main__/jaxlib/pocketfft.py"))
  if r.Rlocation("__main__/jaxlib/cusolver_kernels.so") is not None:
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cusolver_kernels.so"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cublas_kernels.so"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cuda_lu_pivot_kernels.so"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cuda_prng_kernels.so"))
  if r.Rlocation("__main__/jaxlib/cusolver_kernels.pyd") is not None:
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cusolver_kernels.pyd"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cublas_kernels.pyd"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cuda_lu_pivot_kernels.pyd"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cuda_prng_kernels.pyd"))
  if r.Rlocation("__main__/jaxlib/cusolver.py") is not None:
    libdevice_dir = os.path.join(jaxlib_dir, "cuda", "nvvm", "libdevice")
    os.makedirs(libdevice_dir)
    copy_file(r.Rlocation("local_config_cuda/cuda/cuda/nvvm/libdevice/libdevice.10.bc"),
              dst_dir=libdevice_dir)
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cusolver.py"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cuda_linalg.py"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cuda_prng.py"))
  if r.Rlocation("__main__/jaxlib/rocblas_kernels.so") is not None:
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/rocblas_kernels.so"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/rocsolver.py"))
  if r.Rlocation("__main__/jaxlib/cusparse_kernels.so") is not None:
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cusparse_kernels.so"))
    copy_to_jaxlib(r.Rlocation("__main__/jaxlib/cusparse.py"))
  copy_to_jaxlib(r.Rlocation("__main__/jaxlib/version.py"))

  if _is_windows():
    copy_to_jaxlib(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_extension.pyd"))
  else:
    copy_to_jaxlib(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_extension.so"))
  patch_copy_xla_extension_stubs(jaxlib_dir)
  patch_copy_xla_client_py(jaxlib_dir)

  if not _is_windows():
    copy_to_jaxlib(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client_extension.so"))
    patch_copy_tpu_client_py(jaxlib_dir)


def build_wheel(sources_path, output_path):
  """Builds a wheel in `output_path` using the source tree in `sources_path`."""
  if platform.system() == "Windows":
    cpu_name = "amd64"
    platform_name = "win"
  else:
    platform_name, cpu_name = {
      ("Linux", "x86_64"): ("manylinux2010", "x86_64"),
      ("Linux", "aarch64"): ("manylinux2014", "aarch64"),
      ("Darwin", "x86_64"): ("macosx_10_9", "x86_64"),
      ("Darwin", "arm64"): ("macosx_11_0", "arm64"),
    }[(platform.system(), platform.machine())]
  python_tag_arg = (f"--python-tag=cp{sys.version_info.major}"
                    f"{sys.version_info.minor}")
  platform_tag_arg = f"--plat-name={platform_name}_{cpu_name}"
  cwd = os.getcwd()
  os.chdir(sources_path)
  subprocess.run([sys.executable, "setup.py", "bdist_wheel",
                 python_tag_arg, platform_tag_arg])
  os.chdir(cwd)
  for wheel in glob.glob(os.path.join(sources_path, "dist", "*.whl")):
    output_file = os.path.join(output_path, os.path.basename(wheel))
    sys.stderr.write(f"Output wheel: {output_file}\n\n")
    sys.stderr.write("To install the newly-built jaxlib wheel, run:\n")
    sys.stderr.write(f"  pip install {output_file}\n\n")
    shutil.copy(wheel, output_path)


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jaxlib")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(sources_path)
  build_wheel(sources_path, args.output_path)
finally:
  if tmpdir:
    tmpdir.cleanup()
