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

import ctypes
import functools
import importlib
import logging
import os
import pathlib
import traceback
from typing import Any

from jax._src.lib import triton
from jax._src.lib import xla_client
import jax._src.xla_bridge as xb

cuda_plugin_extension = None
cuda_versions = None

def _import_extensions():
  global cuda_plugin_extension
  global cuda_versions

  # cuda_plugin_extension locates inside jaxlib. `jaxlib` is for testing without
  # preinstalled jax cuda plugin packages.
  for pkg_name in ['jax_cuda13_plugin', 'jax_cuda12_plugin', 'jaxlib.cuda']:
    try:
      cuda_plugin_extension = importlib.import_module(
          f'{pkg_name}.cuda_plugin_extension'
      )
      cuda_versions = importlib.import_module(
          f'{pkg_name}._versions'
      )
    except ImportError:
      cuda_plugin_extension = None
      cuda_versions = None
    else:
      break

logger = logging.getLogger(__name__)


def _get_library_path():
  installed_path = (
      pathlib.Path(__file__).resolve().parent / 'xla_cuda_plugin.so'
  )
  if installed_path.exists():
    return installed_path

  local_path = os.path.join(
      os.path.dirname(__file__), 'pjrt_c_api_gpu_plugin.so'
  )
  if not os.path.exists(local_path):
    runfiles_dir = os.getenv('RUNFILES_DIR', None)
    if runfiles_dir:
      local_path = os.path.join(
          runfiles_dir, '__main__/jax_plugins/cuda/pjrt_c_api_gpu_plugin.so'
      )

  if os.path.exists(local_path):
    logger.debug(
        'Native library %s does not exist. This most likely indicates an issue'
        ' with how %s was built or installed. Fallback to local test'
        ' library %s',
        installed_path,
        __package__,
        local_path,
    )
    return local_path

  logger.debug(
      'WARNING: Native library %s and local test library path %s do not'
      ' exist. This most likely indicates an issue with how %s was built or'
      ' installed or missing src files.',
      installed_path,
      local_path,
      __package__,
  )
  return None


def _load(module, libraries):
  try:
    m = importlib.import_module(f"nvidia.{module}")
  except ImportError:
    m = None

  for lib in libraries:
    excs = []
    if m is not None:
      path = pathlib.Path(m.__path__[0]) / "lib" / lib
      try:
        ctypes.cdll.LoadLibrary(path)
        continue
      except OSError as e:
        excs.append(e)

    # TODO(phawkins): check the non-Python path here and error if not found.
    # # Try again, without the Python module path.
    # try:
    #   ctypes.cdll.LoadLibrary(lib)
    #   continue
    # except OSError as e:
    #   excs.append(e)
    #
    # raise ExceptionGroup(f"Unable to load CUDA library {lib}", excs)  # noqa: F821


def _load_nvidia_libraries():
  """Attempts to load NVIDIA's libraries.

  We prefer the Python packages, if present. If not, we fall back to loading
  them from LD_LIBRARY_PATH. By loading the libraries here, later lookups will
  find these copies."""
  _load("cuda_runtime", ["libcudart.so.12"])
  _load("cu13", ["libcudart.so.13"])
  # cuda_nvrtc isn't directly a dependency of JAX, but CUDNN appears to need it
  # and at least in CUDA 12.9 has RUNPATHs misconfigured to refer to
  # nvidia/nvrtc instead of nvidia/cuda_nvrtc.
  _load("cuda_nvrtc", ["libnvrtc.so.12"])
  _load("cu13", ["libnvrtc.so.13"])
  _load("cublas", ["libcublas.so.12", "libcublasLt.so.12"])
  _load("cu13", ["libcublas.so.13", "libcublasLt.so.13"])
  _load("nccl", ["libnccl.so.2"])
  _load("cuda_cupti", ["libcupti.so.12"])
  _load("cu13", ["libcupti.so.13"])
  _load("cusparse", ["libcusparse.so.12"])
  _load("cu13", ["libcusparse.so.12"])
  _load("cusolver", ["libcusolver.so.11"])
  _load("cu13", ["libcusolver.so.12"])
  _load("cufft", ["libcufft.so.11"])
  _load("cu13", ["libcufft.so.12"])
  _load("nvshmem", ["libnvshmem_host.so.3"])
  _load("cudnn", ["libcudnn.so.9"])


def _check_cuda_versions(raise_on_first_error: bool = False,
                         debug: bool = False):
  assert cuda_versions is not None
  results: list[dict[str, Any]] = []

  def _make_msg(name: str,
                runtime_version: int,
                build_version: int,
                min_supported: int,
                debug_msg: bool = False):
    if debug_msg:
      return (f"Package: {name}\n"
              f"Version JAX was built against: {build_version}\n"
              f"Minimum supported: {min_supported}\n"
              f"Installed version: {runtime_version}")
    if min_supported:
      req_str = (f"The local installation version must be no lower than "
                 f"{min_supported}.")
    else:
      req_str = ("The local installation must be the same version as "
                 "the version against which JAX was built.")
    msg = (f"Outdated {name} installation found.\n"
           f"Version JAX was built against: {build_version}\n"
           f"Minimum supported: {min_supported}\n"
           f"Installed version: {runtime_version}\n"
           f"{req_str}")
    return msg


  def _version_check(name: str,
                     get_version,
                     get_build_version,
                     scale_for_comparison: int = 1,
                     min_supported_version: int = 0) -> int | None:
    """Checks the runtime CUDA component version against the JAX one.

    Args:
      name: Of the CUDA component.
      get_version: A function to get the local runtime version of the component.
      get_build_version: A function to get the build version of the component.
      scale_for_comparison: For rounding down a version to ignore patch/minor.
      min_supported_version: An absolute minimum version required. Must be
        passed without rounding down.

    Returns: the runtime version, or None if the component is not found.

    Raises:
      RuntimeError: If the component is not found, or is of unsupported version,
        and if raising the error is not deferred till later.
    """

    build_version = get_build_version()
    try:
      version = get_version()
    except Exception as e:
      err_msg = f"Unable to load {name}. Is it installed?"
      if raise_on_first_error:
        raise RuntimeError(err_msg) from e
      err_msg += f"\n{traceback.format_exc()}"
      results.append({"name": name, "installed": False, "msg": err_msg})
      return

    if not min_supported_version:
      min_supported_version = build_version // scale_for_comparison
    passed = min_supported_version <= version

    if not passed or debug:
      msg = _make_msg(name=name,
                      runtime_version=version,
                      build_version=build_version,
                      min_supported=min_supported_version,
                      debug_msg=passed)
      if not passed and raise_on_first_error:
        raise RuntimeError(msg)
      else:
        record = {"name": name,
                  "installed": True,
                  "msg": msg,
                  "passed": passed,
                  "build_version": build_version,
                  "version": version,
                  "minimum_supported": min_supported_version}
        results.append(record)
    return version

  _version_check("CUDA", cuda_versions.cuda_runtime_get_version,
                 cuda_versions.cuda_runtime_build_version,
                 scale_for_comparison=10,
                 min_supported_version=12010)
  cudnn_version = _version_check(
      "cuDNN",
      cuda_versions.cudnn_get_version,
      cuda_versions.cudnn_build_version,
      # NVIDIA promise both backwards and forwards compatibility for cuDNN patch
      # versions:
      # https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/forward-compatibility.html#cudnn-api-compatibility
      scale_for_comparison=100,
  )
  _version_check("cuFFT", cuda_versions.cufft_get_version,
                 cuda_versions.cufft_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100)
  # TODO(phawkins): for some reason this check fails with a cusolver internal
  # error when fetching the version. This may be a path error from our stubs.
  # Figure out what's happening here and re-enable.
  # _version_check("cuSOLVER", cuda_versions.cusolver_get_version,
  #                cuda_versions.cusolver_build_version,
  #                # Ignore patch versions.
  #                scale_for_comparison=100,
  #                min_supported_version=11400)
  _version_check("cuPTI", cuda_versions.cupti_get_version,
                 cuda_versions.cupti_build_version,
                 min_supported_version=18)
  cublas_version = _version_check("cuBLAS", cuda_versions.cublas_get_version,
                 cuda_versions.cublas_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=120100)
  _version_check("cuSPARSE", cuda_versions.cusparse_get_version,
                 cuda_versions.cusparse_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=12100)

  # https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-10-1
  if (cudnn_version is not None and cudnn_version == 91000
      and cuda_versions.cudnn_build_version() != 91000):
    msg = ("cuDNN 9.10.0 had a binary backward-compatibility issue due to reordered enum "
           f"values affecting block-scale datatypes. Found runtime version {cudnn_version} "
           f"and build version {cuda_versions.cudnn_build_version()}. Please upgrade to "
           "9.10.1 or above.")
    if raise_on_first_error:
      raise RuntimeError(msg)
    else:
      results.append({"installed": True, "msg": msg, "passed": False})
  # xb.local_device_count() cannot safely be called at this point
  if xb.CUDA_VISIBLE_DEVICES.value == "all":
    local_device_count = cuda_versions.cuda_device_count()
  else:
    local_device_count = len(xb.CUDA_VISIBLE_DEVICES.value.split(","))
  # https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-10-0
  if (cudnn_version is not None and cudnn_version < 91001
      and cublas_version is not None and cublas_version >= 120900
      and local_device_count > 1):
    msg = (f"cuDNN < 9.10.0 ({cudnn_version} found) had an issue that caused some multi-GPU "
            "matmuls, in which the same finalized execution plan is used across different "
            f"GPUs, to be functionally incorrect when run with cublasLt >= 12.9 ({cublas_version} "
            "found). Please upgrade to 9.10.1 or above.")
    if raise_on_first_error:
      raise RuntimeError(msg)
    else:
      results.append({"installed": True, "msg": msg, "passed": False})

  errors = []
  debug_results = []
  for result in results:
    message: str = result['msg']
    if not result['installed'] or not result['passed']:
      errors.append(message)
    else:
      debug_results.append(message)

  join_str = f'\n{"-" * 50}\n'
  if debug_results:
    print(f'CUDA components status (debug):\n'
          f'{join_str.join(debug_results)}')
  if errors:
    raise RuntimeError(f'Unable to use CUDA because of the '
                       f'following issues with CUDA components:\n'
                       f'{join_str.join(errors)}')


def initialize():
  _load_nvidia_libraries()
  _import_extensions()
  path = _get_library_path()
  if path is None:
    return

  if not os.getenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK"):
    _check_cuda_versions(raise_on_first_error=True)
  else:
    print('Skipped CUDA versions constraints check due to the '
          'JAX_SKIP_CUDA_CONSTRAINTS_CHECK env var being set.')

  options = xla_client.generate_pjrt_gpu_plugin_options()
  c_api = xb.register_plugin(
      'cuda', priority=500, library_path=str(path), options=options
  )
  if cuda_plugin_extension:
    xla_client.register_custom_call_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_call_target, c_api
        ),
    )
    for _name, _value in cuda_plugin_extension.ffi_registrations().items():
      xla_client.register_custom_call_target(
          _name, _value, platform='CUDA', api_version=1
      )
    xla_client.register_custom_type_id_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.register_custom_type_id, c_api
        ),
    )
    triton.register_compilation_handler(
        "CUDA",
        functools.partial(
            cuda_plugin_extension.compile_triton_to_asm, c_api
        ),
    )
  else:
    logger.warning('cuda_plugin_extension is not found.')
