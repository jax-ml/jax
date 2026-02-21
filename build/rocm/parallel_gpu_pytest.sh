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

# Helper functions for `--config=rocm_pytest`: rules_python test stubs embed
# runfiles-relative paths for the hermetic Python (`PYTHON_BINARY`) and test
# entrypoint (`MAIN`). We resolve those paths across runfiles layouts so pytest
# can apply node-level filtering like `-m "not mosaic_gpu"` reliably.
extract_python_stub_var() {
  # `--config=rocm_pytest` sets `--run_under=//build/rocm:parallel_gpu_pytest`.
  # Not every Bazel "test" target is a Python/pytest test, so we first detect
  # rules_python-generated stubs and extract the embedded Python entrypoints.
  local var_name="$1"
  local stub_file="$2"
  # Typical rules_python stubs embed values like:
  #   PYTHON_BINARY = '...'
  #   MAIN = '...'
  local line=""
  local val=""
  line="$(grep -m1 -E "^[[:space:]]*${var_name} = " "$stub_file" 2>/dev/null || true)"
  # Newer rules_python stubs may not define `MAIN = ...`; instead they embed the
  # entrypoint as `main_rel_path = '__main__/path/to/test.py'`.
  if [[ -z "${line}" && "${var_name}" == "MAIN" ]]; then
    line="$(grep -m1 -E "^[[:space:]]*main_rel_path = " "$stub_file" 2>/dev/null || true)"
  fi
  [[ -z "${line}" ]] && return 0
  # Handle common forms like:
  #   VAR = '...'
  #   VAR = "..."
  #   VAR = r'...'
  #   VAR = r"..."
  # and allow trailing comments.
  if [[ "${var_name}" == "MAIN" && "${line}" =~ ^[[:space:]]*main_rel_path[[:space:]]*= ]]; then
    val="$(printf '%s\n' "${line}" | sed -E "s/^[[:space:]]*main_rel_path = r?['\\\"](.*)['\\\"][[:space:]]*(#.*)?$/\\1/" || true)"
  else
    val="$(printf '%s\n' "${line}" | sed -E "s/^[[:space:]]*${var_name} = r?['\\\"](.*)['\\\"][[:space:]]*(#.*)?$/\\1/" || true)"
  fi
  [[ "${val}" == "${line}" ]] && val=""
  printf '%s' "${val}"
}

resolve_runfiles_path() {
  # Resolve a stub-provided runfiles-relative path to an absolute path.
  # We prefer `${TEST_SRCDIR}/<rel>` but fall back to `${TEST_SRCDIR}/__main__/<rel>`
  # since some stubs refer to paths rooted at the main repo.
  local rel="$1"
  if [[ -z "$rel" ]]; then
    return 1
  fi
  if [[ -e "${TEST_SRCDIR}/${rel}" ]]; then
    echo "${TEST_SRCDIR}/${rel}"
    return 0
  fi
  if [[ -e "${TEST_SRCDIR}/__main__/${rel}" ]]; then
    echo "${TEST_SRCDIR}/__main__/${rel}"
    return 0
  fi
  return 1
}

# Resolve hermetic Python, test .py file, and PYTHONPATH from runfiles.
setup_pytest_env() {
  local test_bin="$1"
  HERMETIC_PY_REL="$(extract_python_stub_var "PYTHON_BINARY" "$test_bin" || true)"
  MAIN_REL="$(extract_python_stub_var "MAIN" "$test_bin" || true)"

  # If this isn't a rules_python stub, we can't run it via pytest.
  if [[ -z "${HERMETIC_PY_REL}" ]]; then
    return 1
  fi

  HERMETIC_PY="$(resolve_runfiles_path "${HERMETIC_PY_REL}")"
  if [[ -z "${HERMETIC_PY:-}" || ! -x "${HERMETIC_PY}" ]]; then
    return 1
  fi

  TEST_PY="$(resolve_runfiles_path "${MAIN_REL}")"
  if [[ -z "${TEST_PY:-}" || ! -f "${TEST_PY}" ]]; then
    return 1
  fi

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
  # Split args into:
  # - `PYTEST_ARGS`: forwarded to `pytest` (supports node-level filtering, e.g.
  #   `-m "not mosaic_gpu"` to skip Mosaic GPU tests on ROCm).
  # - `PASSTHRU_ARGS`: forwarded to non-python test binaries (so `--config=rocm_pytest`
  #   doesn't break non-pytest targets by passing `-m/-k/...`).
  PYTEST_ARGS=()
  PASSTHRU_ARGS=()
  local skip_next=false
  local saw_marker=false
  for arg in "$@"; do
    if $skip_next; then
      PYTEST_ARGS+=("$arg")
      skip_next=false
    elif [[ "$arg" == "-k" || "$arg" == "-m" || "$arg" == "-v" || "$arg" == "-x" || "$arg" == "-s" || "$arg" == -r* ]]; then
      PYTEST_ARGS+=("$arg")
      [[ "$arg" == "-m" ]] && saw_marker=true
      [[ "$arg" == "-k" || "$arg" == "-m" ]] && skip_next=true
    elif [[ "$arg" == --jax_test_dut=* ]]; then
      export JAX_TEST_DUT="${arg#*=}"
    elif [[ "$arg" == --jax_platform_name=* ]]; then
      export JAX_PLATFORM_NAME="${arg#*=}"
    else
      PASSTHRU_ARGS+=("$arg")
    fi
  done
}

run_pytest() {
  # Run the underlying Python test under pytest so marker/node-level filtering
  # from `--config=rocm_pytest` works uniformly on ROCm.
  # Ensure pytest discovers the repo-level `conftest.py` in runfiles. Without
  # this, `-m "not mosaic_gpu"` can become a no-op because the `mosaic_gpu`
  # marker is applied by `conftest.py`.
  # Bazel sharding support: when a test target is sharded (shard_count > 1),
  # Bazel expects the test runner to indicate sharding support by touching
  # TEST_SHARD_STATUS_FILE. We only do this when we actually run pytest.
  if [[ -n "${TEST_SHARD_STATUS_FILE:-}" ]]; then
    touch "${TEST_SHARD_STATUS_FILE}" 2>/dev/null || true
  fi

  local pytest_rootdir=""
  if [[ -n "${TEST_SRCDIR:-}" && -f "${TEST_SRCDIR}/__main__/conftest.py" ]]; then
    pytest_rootdir="${TEST_SRCDIR}/__main__"
  elif [[ -n "${TEST_SRCDIR:-}" && -f "${TEST_SRCDIR}/conftest.py" ]]; then
    pytest_rootdir="${TEST_SRCDIR}"
  fi

  if [[ -n "${pytest_rootdir}" ]]; then
    "$HERMETIC_PY" -m pytest --rootdir="${pytest_rootdir}" --confcutdir="${pytest_rootdir}" "$TEST_PY" "${PYTEST_ARGS[@]}"
  else
    "$HERMETIC_PY" -m pytest "$TEST_PY" "${PYTEST_ARGS[@]}"
  fi
  local rc=$?
  # pytest uses exit code 5 when no tests were selected/collected (e.g. marker
  # filters deselect everything). Treat that as success for Bazel runs.
  if [[ $rc -eq 5 ]]; then
    return 0
  fi
  return $rc
}

run_passthru() {
  # For non-python tests we still run under the GPU lock, but we must not pass
  # pytest-only arguments (-m/-k/...) that came from --config=rocm_pytest.
  "$TEST_BINARY" "${PASSTHRU_ARGS[@]}"
}

if [[ $TF_GPU_COUNT == 0 ]]; then
    echo "Execute with no GPU support (pytest mode)"
    TEST_BIN="$1"; shift
    parse_pytest_args "$@"
    if setup_pytest_env "$TEST_BIN"; then
      run_pytest
      exit $?
    else
      TEST_BINARY="$TEST_BIN"
      run_passthru
      exit $?
    fi
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
        parse_pytest_args "$@"
        if setup_pytest_env "$TEST_BINARY"; then
          run_pytest
        else
          run_passthru
        fi
      )
      return_code=$?
      flock -u "$lock_fd"
      exit $return_code
    fi
  done
done

echo "Cannot find a free GPU to run the test $* on, exiting with failure..."
exit 1
