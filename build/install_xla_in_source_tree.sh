#!/bin/bash
#
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

# Script that installs JAX's XLA dependencies inside the JAX source tree.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <target directory>"
  exit 1
fi
TARGET="$1"

if [[ ! -d "${TARGET}/jaxlib" ]]; then
  echo "Target directory ${TARGET} does not have a jaxlib directory"
  exit 1
fi

# Copy the XLA dependencies into jax/lib, fixing up some imports to point to the
# new location.
cp -f "$(rlocation __main__/jaxlib/lapack.so)" "${TARGET}/jaxlib"
cp -f "$(rlocation __main__/jaxlib/pytree.so)" "${TARGET}/jaxlib"
if [[ -x "$(rlocation __main__/jaxlib/cusolver_kernels.so)" ]]; then
  cp -f "$(rlocation __main__/jaxlib/cublas_kernels.so)" "${TARGET}/jaxlib"
  cp -f "$(rlocation __main__/jaxlib/cusolver_kernels.so)" "${TARGET}/jaxlib"
fi
cp -f "$(rlocation __main__/jaxlib/version.py)" "${TARGET}/jaxlib"
cp -f "$(rlocation __main__/jaxlib/cusolver.py)" "${TARGET}/jaxlib"
cp -f "$(rlocation org_tensorflow/tensorflow/compiler/xla/python/xla_extension.so)" \
  "${TARGET}/jaxlib"
cp -f "$(rlocation org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client_extension.so)" \
  "${TARGET}/jaxlib"
sed \
  -e 's/from tensorflow.compiler.xla.python import xla_extension as _xla/from . import xla_extension as _xla/' \
  -e 's/from tensorflow.compiler.xla.python.xla_extension import ops/from .xla_extension import ops/' \
  < "$(rlocation org_tensorflow/tensorflow/compiler/xla/python/xla_client.py)" \
  > "${TARGET}/jaxlib/xla_client.py"
sed \
  -e 's/from tensorflow.compiler.xla.python import xla_extension as _xla/from . import xla_extension as _xla/' \
  -e 's/from tensorflow.compiler.xla.python import xla_client/from . import xla_client/' \
  -e 's/from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client/from . import tpu_client_extension as _tpu_client/' \
  < "$(rlocation org_tensorflow/tensorflow/compiler/xla/python/tpu_driver/client/tpu_client.py)" \
  > "${TARGET}/jaxlib/tpu_client.py"
