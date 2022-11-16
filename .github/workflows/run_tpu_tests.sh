#!/usr/bin/env bash

# Must be run from jax checkout

set -eux

num_tpus=$(python3 -c 'import jax; print(jax.device_count())')

# Single-accelerator tests can run on one chip each
bazel test \
  --repo_env=PYTHON_BIN_PATH=$(which python3) \
  --//jax:build_jaxlib=false \
  --run_under "$(pwd)/build/parallel_accelerator_execute.sh" \
  --test_output=errors \
  --test_env=JAX_ACCELERATOR_COUNT="${num_tpus}" \
  --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
  # Number of jobs (i.e. test processes) should equal accelerator_count *
  # tests_per_accelerator for maximum parallelism
  --jobs="${num_tpus}" \
  --test_tag_filters=-multiaccelerator \
  --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
  //tests:tpu_tests //tests:backend_independent_tests

# Run multi-accelerator on all available chips
bazel test \
  --repo_env=PYTHON_BIN_PATH=$(which python3) \
  --//jax:build_jaxlib=false \
  --test_output=errors \
  --jobs=1 \
  --test_tag_filters=multiaccelerator \
  //tests:tpu_tests //tests:backend_independent_tests
