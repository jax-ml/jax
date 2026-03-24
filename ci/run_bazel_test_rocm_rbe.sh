#!/bin/bash
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
# Runs Bazel GPU tests with RBE. This runs single accelerator tests with one
# GPU apiece on RBE.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Run Bazel GPU tests with RBE (single accelerator tests with one GPU apiece).
echo "Running RBE GPU tests..."

TAG_FILTERS="jax_test_gpu,-config-cuda-only,-manual"

TESTS_TO_IGNORE=(
    -//tests/pallas:pallas_test_gpu
    -//tests/pallas:ops_test_gpu
    -//tests/pallas:ops_test_mgpu_gpu
    -//tests/pallas:pallas_shape_poly_test_gpu
    -//tests/pallas:pallas_vmap_test_gpu
    -//tests/pallas:triton_pallas_test_gpu
    -//tests:export_harnesses_multi_platform_test_gpu
    -//tests:jet_test_gpu
    -//tests:lax_autodiff_test_gpu
    -//tests:lax_numpy_setops_test_gpu
    -//tests:lax_numpy_test_gpu
    -//tests:lax_test_gpu
    -//tests:linalg_test_gpu
    -//tests:logging_test_gpu
    -//tests:random_lax_test_gpu
    -//tests:scipy_signal_test_gpu
    -//tests:stax_test_gpu
    -//tests:ode_test_gpu
    -//tests:lobpcg_test_gpu
    -//tests:scipy_stats_test_gpu
    -//tests:nn_test_gpu
    -//tests:lax_scipy_sparse_test_gpu
    -//tests:lax_scipy_spectral_dac_test_gpu
    -//tests:lax_scipy_special_functions_test_gpu
    -//tests:cholesky_update_test_gpu
    -//tests:api_test_gpu
    -//tests:ann_test_gpu
    -//tests:experimental_rnn_test_gpu
    -//tests:lax_vmap_test_gpu
    -//tests:qdwh_test_gpu
    -//tests:scaled_dot_test_gpu
    -//tests:scipy_spatial_test_gpu
    -//tests:shape_poly_test_gpu
    -//tests:sparsify_test_gpu
    -//tests:lax_numpy_reducers_test_gpu
    -//tests:scipy_optimize_test_gpu
)

for arg in "$@"; do
    if [[ "$arg" == "--config=multi_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},multiaccelerator"
    fi
    if [[ "$arg" == "--config=single_gpu" ]]; then
        TAG_FILTERS="${TAG_FILTERS},gpu,-multiaccelerator"
    fi
    if [[ "$arg" == "--//jax:build_jaxlib=false" ]]; then
        # tests to ignore for pre-built plugin wheels
        TESTS_TO_IGNORE+=(
            -//tests:buffer_callback_test_gpu
        )
    fi
done

bazel --bazelrc=build/rocm/rocm.bazelrc test \
    --config=rocm \
    --config=rocm_rbe_dynamic \
    --test_env=XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    --test_output=errors \
    --test_env=TF_CPP_MIN_LOG_LEVEL=0 \
    --test_env=JAX_EXCLUDE_TEST_TARGETS=PmapTest.testSizeOverflow \
    --build_tag_filters=${TAG_FILTERS} \
    --test_tag_filters=${TAG_FILTERS} \
    --remote_download_regex='.*test\.xml$' \
    --test_env=JAX_SKIP_SLOW_TESTS=true \
    --repo_env=TF_ROCM_AMDGPU_TARGETS="gfx908,gfx90a,gfx942,gfx950" \
    --color=yes \
    $@ \
    --spawn_strategy=local \
    -- \
    //tests:gpu_tests \
    //tests:backend_independent_tests \
    //tests/pallas:gpu_tests \
    //tests/pallas:backend_independent_tests \
    //jaxlib/tools:check_gpu_wheel_sources_test \
    "${TESTS_TO_IGNORE[@]}"

ci/utilities/collect_bazel_test_xmls.sh test-artifacts
exit "${bazel_retval:-0}"