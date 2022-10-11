#!/usr/bin/env bash
# Copyright 2022 The JAX Authors.
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

set -eux

ROCM_TF_FORK_REPO="https://github.com/ROCmSoftwarePlatform/tensorflow-upstream"
ROCM_TF_FORK_BRANCH="develop-upstream"
rm -rf /tmp/tensorflow-upstream || true
git clone -b ${ROCM_TF_FORK_BRANCH} ${ROCM_TF_FORK_REPO} /tmp/tensorflow-upstream
if [ ! -v TENSORFLOW_ROCM_COMMIT ]; then
    echo "The TENSORFLOW_ROCM_COMMIT environment variable is not set, using top of branch"
elif [ ! -z "$TENSORFLOW_ROCM_COMMIT" ]
then
      echo "Using tensorflow-rocm at commit: $TENSORFLOW_ROCM_COMMIT"
      cd /tmp/tensorflow-upstream
      git checkout $TENSORFLOW_ROCM_COMMIT
      cd -
fi


python3 ./build/build.py --enable_rocm --rocm_path=${ROCM_PATH} --bazel_options=--override_repository=org_tensorflow=/tmp/tensorflow-upstream
pip3 install --force-reinstall dist/*.whl  # installs jaxlib (includes XLA)
pip3 install --force-reinstall .  # installs jax
