#!/bin/bash
# Script for git bisect run INSIDE the GitHub Runner container
# Tests H100 CUDA timeout targets

export JAXCI_BAZEL_TARGETS="
//tests:array_test_gpu
//tests:custom_partitioning_test_gpu
//tests:debugging_primitives_test_gpu
//tests:export_test_gpu
//tests:memories_test_gpu
//tests:profiler_test_gpu
//tests:python_callback_test_gpu
//tests:ragged_collective_test_gpu
//tests:shard_alike_test_gpu
//tests/multiprocess:thread_guard_test_gpu
//tests/pallas:gpu_pallas_distributed_test_gpu
//tests:pjit_test_gpu
//tests:pmap_test_gpu
//tests:shard_map_test_gpu
"
export JAXCI_HERMETIC_PYTHON_VERSION="3.14"
export JAXCI_XLA_TRACK="pinned"
export JAXCI_TEST_TAG_FILTERS="multiaccelerator"
export JAXCI_LOCAL_TEST_JOBS="8"
export JAXCI_EXCLUDE_TEST_TARGETS="PmapTest.testSizeOverflow|.*InterpretTest.*"

echo "Running target tests inside runner containment:"
echo "$JAXCI_BAZEL_TARGETS"

# Locate repository root
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1

echo "Repository root: $(pwd)"

bash ./ci/run_bazel_cuda_targeted_tests.sh
