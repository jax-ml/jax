# Copyright 2020 The JAX Authors.
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
import os
import platform
import re
import subprocess
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from jax.tools import build_utils

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
parser.add_argument(
  "--cpu",
  default=None,
  required=True,
  help="Target CPU architecture. Required.")
parser.add_argument(
  "--editable",
  action="store_true",
  help="Create an 'editable' jaxlib build instead of a wheel.")
parser.add_argument(
    "--include_gpu_plugin_extension",
    default=False,
    help="Whether to include gpu plugin extension.",
)
args = parser.parse_args()

r = runfiles.Create()


def _is_mac():
  return platform.system() == "Darwin"


pyext = "pyd" if build_utils.is_windows() else "so"


def exists(src_file):
  return r.Rlocation(src_file) is not None


def patch_copy_mlir_import(src_file, dst_dir):
  src_file = r.Rlocation(src_file)
  src_filename = os.path.basename(src_file)
  with open(src_file) as f:
    src = f.read()

  with open(os.path.join(dst_dir, src_filename), 'w') as f:
    replaced = re.sub(r'^from mlir(\..*)? import (.*)', r'from jaxlib.mlir\1 import \2', src, flags=re.MULTILINE)
    f.write(replaced)

_XLA_EXTENSION_STUBS = [
    "__init__.pyi",
    "jax_jit.pyi",
    "ops.pyi",
    "outfeed_receiver.pyi",
    "pmap_lib.pyi",
    "profiler.pyi",
    "pytree.pyi",
    "transfer_guard_lib.pyi",
]
_OPTIONAL_XLA_EXTENSION_STUBS = [
]


def patch_copy_xla_extension_stubs(dst_dir):
  # This file is required by PEP-561. It marks jaxlib as package containing
  # type stubs.
  with open(os.path.join(dst_dir, "py.typed"), "w"):
    pass
  xla_extension_dir = os.path.join(dst_dir, "xla_extension")
  os.makedirs(xla_extension_dir)
  for stub_name in _XLA_EXTENSION_STUBS:
    stub_path = r.Rlocation(
        "xla/xla/python/xla_extension/" + stub_name)
    stub_path = str(stub_path)  # Make pytype accept os.path.exists(stub_path).
    if stub_name in _OPTIONAL_XLA_EXTENSION_STUBS and not os.path.exists(stub_path):
      continue
    with open(stub_path) as f:
      src = f.read()
    src = src.replace(
        "from xla.python import xla_extension",
        "from .. import xla_extension"
    )
    with open(os.path.join(xla_extension_dir, stub_name), "w") as f:
      f.write(src)


def verify_mac_libraries_dont_reference_chkstack():
  """Verifies that xla_extension.so doesn't depend on ____chkstk_darwin.

  We don't entirely know why this happens, but in some build environments
  we seem to target the wrong Mac OS version.
  https://github.com/google/jax/issues/3867

  This check makes sure we don't release wheels that have this dependency.
  """
  if not _is_mac():
    return
  nm = subprocess.run(
    ["nm", "-g",
     r.Rlocation("xla/xla/python/xla_extension.so")
    ],
    capture_output=True, text=True,
    check=False)
  if nm.returncode != 0:
    raise RuntimeError(f"nm process failed: {nm.stdout} {nm.stderr}")
  if "____chkstk_darwin" in nm.stdout:
    raise RuntimeError(
      "Mac wheel incorrectly depends on symbol ____chkstk_darwin, which "
      "means that it isn't compatible with older MacOS versions.")


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(os.path.join(sources_path, "setup.cfg"), "w") as f:
    f.write(f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat-name={tag}
""")


def prepare_wheel(sources_path, *, cpu, include_gpu_plugin_extension):
  """Assembles a source tree for the wheel in `sources_path`."""
  jaxlib_dir = os.path.join(sources_path, "jaxlib")
  os.makedirs(jaxlib_dir)
  copy_to_jaxlib = functools.partial(build_utils.copy_file, dst_dir=jaxlib_dir, runfiles=r)

  verify_mac_libraries_dont_reference_chkstack()
  build_utils.copy_file("__main__/jaxlib/tools/LICENSE.txt", dst_dir=sources_path, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/README.md", dst_dir=sources_path, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/setup.py", dst_dir=sources_path, runfiles=r)
  write_setup_cfg(sources_path, cpu)
  copy_to_jaxlib("__main__/jaxlib/init.py", dst_filename="__init__.py")
  copy_to_jaxlib(f"__main__/jaxlib/cpu_feature_guard.{pyext}")
  if include_gpu_plugin_extension:
    copy_to_jaxlib(f"__main__/jaxlib/cuda_plugin_extension.{pyext}")
  copy_to_jaxlib(f"__main__/jaxlib/utils.{pyext}")
  copy_to_jaxlib("__main__/jaxlib/lapack.py")
  copy_to_jaxlib("__main__/jaxlib/hlo_helpers.py")
  copy_to_jaxlib("__main__/jaxlib/ducc_fft.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_prng.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_linalg.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_rnn.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_triton.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_common_utils.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_solver.py")
  copy_to_jaxlib("__main__/jaxlib/gpu_sparse.py")
  copy_to_jaxlib("__main__/jaxlib/tpu_mosaic.py")
  copy_to_jaxlib("__main__/jaxlib/version.py")
  copy_to_jaxlib("__main__/jaxlib/xla_client.py")
  copy_to_jaxlib(f"__main__/jaxlib/xla_extension.{pyext}")
  cpu_dir = os.path.join(jaxlib_dir, "cpu")
  os.makedirs(cpu_dir)
  build_utils.copy_file(f"__main__/jaxlib/cpu/_lapack.{pyext}", dst_dir=cpu_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/cpu/_ducc_fft.{pyext}", dst_dir=cpu_dir, runfiles=r)

  cuda_dir = os.path.join(jaxlib_dir, "cuda")
  if exists(f"__main__/jaxlib/cuda/_solver.{pyext}"):
    libdevice_dir = os.path.join(cuda_dir, "nvvm", "libdevice")
    os.makedirs(libdevice_dir)
    build_utils.copy_file("local_config_cuda/cuda/cuda/nvvm/libdevice/libdevice.10.bc", dst_dir=libdevice_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/cuda/_solver.{pyext}", dst_dir=cuda_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/cuda/_blas.{pyext}", dst_dir=cuda_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/cuda/_linalg.{pyext}", dst_dir=cuda_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/cuda/_prng.{pyext}", dst_dir=cuda_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/cuda/_rnn.{pyext}", dst_dir=cuda_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/cuda/_triton.{pyext}", dst_dir=cuda_dir, runfiles=r)
  rocm_dir = os.path.join(jaxlib_dir, "rocm")
  if exists(f"__main__/jaxlib/rocm/_solver.{pyext}"):
    os.makedirs(rocm_dir)
    build_utils.copy_file(f"__main__/jaxlib/rocm/_solver.{pyext}", dst_dir=rocm_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/rocm/_blas.{pyext}", dst_dir=rocm_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/rocm/_linalg.{pyext}", dst_dir=rocm_dir, runfiles=r)
    build_utils.copy_file(f"__main__/jaxlib/rocm/_prng.{pyext}", dst_dir=rocm_dir, runfiles=r)
  if exists(f"__main__/jaxlib/cuda/_sparse.{pyext}"):
    build_utils.copy_file(f"__main__/jaxlib/cuda/_sparse.{pyext}", dst_dir=cuda_dir, runfiles=r)
  if exists(f"__main__/jaxlib/rocm/_sparse.{pyext}"):
    build_utils.copy_file(f"__main__/jaxlib/rocm/_sparse.{pyext}", dst_dir=rocm_dir, runfiles=r)

  mosaic_dir = os.path.join(jaxlib_dir, "mosaic")
  mosaic_python_dir = os.path.join(mosaic_dir, "python")
  os.makedirs(mosaic_dir)
  os.makedirs(mosaic_python_dir)
  copy_to_jaxlib("__main__/jaxlib/mosaic/python/apply_vector_layout.py", dst_dir=mosaic_python_dir)
  copy_to_jaxlib("__main__/jaxlib/mosaic/python/infer_memref_layout.py", dst_dir=mosaic_python_dir)
  copy_to_jaxlib("__main__/jaxlib/mosaic/python/tpu.py", dst_dir=mosaic_python_dir)
  build_utils.copy_file("__main__/jaxlib/mosaic/python/_tpu_ops_ext.py", dst_dir=mosaic_python_dir, runfiles=r)
  # TODO (sharadmv,skyewm): can we avoid patching this file?
  patch_copy_mlir_import("__main__/jaxlib/mosaic/python/_tpu_gen.py", dst_dir=mosaic_python_dir)

  mlir_dir = os.path.join(jaxlib_dir, "mlir")
  mlir_dialects_dir = os.path.join(jaxlib_dir, "mlir", "dialects")
  mlir_libs_dir = os.path.join(jaxlib_dir, "mlir", "_mlir_libs")
  os.makedirs(mlir_dir)
  os.makedirs(mlir_dialects_dir)
  os.makedirs(mlir_libs_dir)
  build_utils.copy_file("__main__/jaxlib/mlir/ir.py", dst_dir=mlir_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/passmanager.py", dst_dir=mlir_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_builtin_ops_ext.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_builtin_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_chlo_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_mhlo_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_stablehlo_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_ods_common.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_func_ops_ext.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_func_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_ml_program_ops_ext.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_ml_program_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_sparse_tensor_enum_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_sparse_tensor_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/sparse_tensor.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/builtin.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/chlo.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/arith.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_arith_enum_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_arith_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_arith_ops_ext.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/math.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_math_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/memref.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_memref_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_memref_ops_ext.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/scf.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_scf_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_scf_ops_ext.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/vector.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_vector_enum_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/_vector_ops_gen.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/mhlo.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/stablehlo.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/func.py", dst_dir=mlir_dialects_dir, runfiles=r)
  build_utils.copy_file("__main__/jaxlib/mlir/dialects/ml_program.py", dst_dir=mlir_dialects_dir, runfiles=r)

  build_utils.copy_file("__main__/jaxlib/mlir/_mlir_libs/__init__.py", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlir.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_chlo.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlirHlo.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlirDialectsSparseTensor.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_mlirSparseTensorPasses.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_tpu_ext.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_stablehlo.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  build_utils.copy_file(f"__main__/jaxlib/mlir/_mlir_libs/_site_initialize_0.{pyext}", dst_dir=mlir_libs_dir, runfiles=r)
  if build_utils.is_windows():
    build_utils.copy_file("__main__/jaxlib/mlir/_mlir_libs/jaxlib_mlir_capi.dll", dst_dir=mlir_libs_dir, runfiles=r)
  elif _is_mac():
    build_utils.copy_file("__main__/jaxlib/mlir/_mlir_libs/libjaxlib_mlir_capi.dylib", dst_dir=mlir_libs_dir, runfiles=r)
  else:
    build_utils.copy_file("__main__/jaxlib/mlir/_mlir_libs/libjaxlib_mlir_capi.so", dst_dir=mlir_libs_dir, runfiles=r)
  patch_copy_xla_extension_stubs(jaxlib_dir)


tmpdir = None
sources_path = args.sources_path
if sources_path is None:
  tmpdir = tempfile.TemporaryDirectory(prefix="jaxlib")
  sources_path = tmpdir.name

try:
  os.makedirs(args.output_path, exist_ok=True)
  prepare_wheel(
      sources_path,
      cpu=args.cpu,
      include_gpu_plugin_extension=args.include_gpu_plugin_extension,
  )
  package_name = "jaxlib"
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(sources_path, args.output_path, package_name)
finally:
  if tmpdir:
    tmpdir.cleanup()
