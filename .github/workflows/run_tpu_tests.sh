#!/usr/bin/env bash

# Must be run from jax checkout

set -eux

echo $PATH

num_tpus=$(python3 -c 'import jax; print(jax.device_count())')

bazel test \
  --repo_env=PYTHON_BIN_PATH=python3 \
  --//jax:build_jaxlib=false \
  --run_under "$(pwd)/build/parallel_accelerator_execute.sh" \
  --test_output=errors \
  --test_env=JAX_ACCELERATOR_COUNT="${num_tpus}" \
  --jobs="${num_tpus}" \
  --test_env=JAX_TESTS_PER_ACCELERATOR=1 \
  --test_tag_filters=-multiaccelerator \
  --test_env=ALLOW_MULTIPLE_LIBTPU_LOAD=true \
  //tests:tpu_tests //tests:backend_independent_tests
