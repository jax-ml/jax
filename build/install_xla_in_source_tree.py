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


import os
import sys
import shutil
import argparse

from bazel_tools.tools.python.runfiles import runfiles

parser = argparse.ArgumentParser()
parser.add_argument("target")
args = parser.parse_args()

r = runfiles.Create()


def _is_windows():
  return sys.platform.startswith("win32")


def _copy_so(src_file, dst_dir):
  src_filename = os.path.basename(src_file)
  if _is_windows() and src_filename.endswith(".so"):
    dst_filename = src_filename[:-3] + ".pyd"
  else:
    dst_filename = src_filename
  dst_file = os.path.join(dst_dir, dst_filename)
  shutil.copyfile(src_file, dst_file)


def _copy_normal(src_file, dst_dir):
  src_filename = os.path.basename(src_file)
  dst_file = os.path.join(dst_dir, src_filename)
  shutil.copyfile(src_file, dst_file)


def copy(src_file, dst_dir=os.path.join(args.target, "jaxlib")):
  if src_file.endswith(".so"):
    _copy_so(src_file, dst_dir)
  else:
    _copy_normal(src_file, dst_dir)


def patch_copy_xla_client_py(dst_dir=os.path.join(args.target, "jaxlib")):
  with open(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_client.py")) as f:
    src = f.read()
    src = src.replace("from tensorflow.compiler.xla.python import xla_extension as _xla",
                      "from . import xla_extension as _xla")
    src = src.replace("from tensorflow.compiler.xla.python.xla_extension import ops",
                      "from .xla_extension import ops")
    with open(os.path.join(dst_dir, "xla_client.py"), "w") as f:
      f.write(src)


def patch_copy_tpu_client_py(dst_dir=os.path.join(args.target, "jaxlib")):
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


copy(r.Rlocation("__main__/jaxlib/lapack.so"))
copy(r.Rlocation("__main__/jaxlib/_pocketfft.so"))
copy(r.Rlocation("__main__/jaxlib/pocketfft_flatbuffers_py_generated.py"))
copy(r.Rlocation("__main__/jaxlib/pocketfft.py"))
if r.Rlocation("__main__/jaxlib/cusolver_kernels.so") is not None:
  copy(r.Rlocation("__main__/jaxlib/cusolver_kernels.so"))
  copy(r.Rlocation("__main__/jaxlib/cublas_kernels.so"))
  copy(r.Rlocation("__main__/jaxlib/cusolver_kernels.so"))
  copy(r.Rlocation("__main__/jaxlib/cuda_prng_kernels.so"))
if r.Rlocation("__main__/jaxlib/cusolver_kernels.pyd") is not None:
  copy(r.Rlocation("__main__/jaxlib/cusolver_kernels.pyd"))
  copy(r.Rlocation("__main__/jaxlib/cublas_kernels.pyd"))
  copy(r.Rlocation("__main__/jaxlib/cusolver_kernels.pyd"))
  copy(r.Rlocation("__main__/jaxlib/cuda_prng_kernels.pyd"))
copy(r.Rlocation("__main__/jaxlib/version.py"))
copy(r.Rlocation("__main__/jaxlib/cusolver.py"))
copy(r.Rlocation("__main__/jaxlib/cuda_prng.py"))

if _is_windows():
  copy(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_extension.pyd"))
else:
  copy(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/xla_extension.so"))
patch_copy_xla_client_py()

if not _is_windows():
  copy(r.Rlocation("org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client_extension.so"))
  patch_copy_tpu_client_py()
