#!/usr/bin/env bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# A script to run GPU tests via pytest in parallel, controlled with an
# environment variable. Wraps the Bazel test binary to invoke pytest with the
# hermetic Python interpreter, enabling pytest-style negative filtering (-k).
#
# Required environment variables:
#     TF_GPU_COUNT = Number of GPUs available (auto-detected via rocminfo).

ROCMINFO=$(find "external/local_config_rocm/rocm/rocm_dist/" -name "rocminfo" -path "*/bin/rocminfo")
TF_GPU_COUNT=$($ROCMINFO | grep "Name: *gfx*" | wc -l)
TF_TESTS_PER_GPU=${TF_TESTS_PER_GPU:-8}

# Resolve hermetic Python, test .py file, and PYTHONPATH from runfiles.
setup_pytest_env() {
  local test_bin="$1"
  HERMETIC_PY_REL=$(grep "^PYTHON_BINARY = " "$test_bin" | sed "s/PYTHON_BINARY = '\\(.*\\)'/\\1/")
  HERMETIC_PY="${TEST_SRCDIR}/${HERMETIC_PY_REL}"
  TEST_PY=$(echo "$test_bin" | sed -E 's/_(gpu|cpu|tpu)$//').py

  PYPATH="${TEST_SRCDIR}/__main__"
  for d in "${TEST_SRCDIR}"/pypi_*/site-packages; do
    [[ -d "$d" ]] && PYPATH="${PYPATH}:${d}"
  done
  for d in "${TEST_SRCDIR}"/__main__/jaxlib/tools/*_py_import_unpacked_wheel; do
    [[ -d "$d" ]] && PYPATH="${PYPATH}:${d}"
  done
  export PYTHONPATH="${PYPATH}${PYTHONPATH:+:$PYTHONPATH}"
  export RUNFILES_DIR="${TEST_SRCDIR}"
}

# Separate pytest args (-k, -v, -x, etc.) from absltest/JAX flags.
# JAX flags are exported as environment variables instead.
parse_pytest_args() {
  PYTEST_ARGS=()
  local skip_next=false
  for arg in "$@"; do
    if $skip_next; then
      PYTEST_ARGS+=("$arg")
      skip_next=false
    elif [[ "$arg" == "-k" || "$arg" == "-v" || "$arg" == "-x" || "$arg" == "-s" ]]; then
      PYTEST_ARGS+=("$arg")
      [[ "$arg" == "-k" ]] && skip_next=true
    elif [[ "$arg" == --jax_test_dut=* ]]; then
      export JAX_TEST_DUT="${arg#*=}"
    elif [[ "$arg" == --jax_platform_name=* ]]; then
      export JAX_PLATFORMS="${arg#*=}"
    fi
  done
}

if [[ $TF_GPU_COUNT == 0 ]]; then
    echo "Execute with no GPU support (pytest mode)"
    TEST_BIN="$1"; shift
    setup_pytest_env "$TEST_BIN"
    parse_pytest_args "$@"
    exec "$HERMETIC_PY" -m pytest "$TEST_PY" "${PYTEST_ARGS[@]}"
fi

function is_absolute {
  [[ "$1" = /* ]] || [[ "$1" =~ ^[a-zA-Z]:[/\\].* ]]
}

export TF_PER_DEVICE_MEMORY_LIMIT_MB=${TF_PER_DEVICE_MEMORY_LIMIT_MB:-4096}

RUNFILES_MANIFEST_FILE="${TEST_SRCDIR}/MANIFEST"
function rlocation() {
  if is_absolute "$1" ; then
    echo "$1"
  elif [[ -e "$TEST_SRCDIR/$1" ]]; then
    echo "$TEST_SRCDIR/$1"
  elif [[ -e "$RUNFILES_MANIFEST_FILE" ]]; then
    echo "$(grep "^$1 " "$RUNFILES_MANIFEST_FILE" | sed 's/[^ ]* //')"
  fi
}

TEST_BINARY="$(rlocation $TEST_WORKSPACE/${1#./})"
shift

mkdir -p /var/lock
for j in $(seq 0 $((TF_TESTS_PER_GPU-1))); do
  for i in $(seq 0 $((TF_GPU_COUNT-1))); do
    exec {lock_fd}>/var/lock/gpulock${i}_${j} || exit 1
    if flock -n "$lock_fd"; then
      (
        export CUDA_VISIBLE_DEVICES=$i
        export HIP_VISIBLE_DEVICES=$i
        echo "Running test $TEST_BINARY $* on GPU $CUDA_VISIBLE_DEVICES (pytest mode)"
        setup_pytest_env "$TEST_BINARY"
        parse_pytest_args "$@"
        "$HERMETIC_PY" -m pytest "$TEST_PY" "${PYTEST_ARGS[@]}"
      )
      return_code=$?
      flock -u "$lock_fd"
      exit $return_code
    fi
  done
done

echo "Cannot find a free GPU to run the test $* on, exiting with failure..."
exit 1
