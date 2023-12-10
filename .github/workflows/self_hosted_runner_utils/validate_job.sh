#!/bin/bash

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

# More or less copied from
# https://github.com/openxla/iree/tree/main/build_tools/github_actions/runner/config

set -euo pipefail

ALLOWED_EVENTS=(
  "schedule"
  "workflow_dispatch"
)

# Tests if the first argument is contained in the array in the second argument.
# Usage `is_contained "element" "${array[@]}"`
is_contained() {
  local e;
  local match="$1"
  shift
  for e in "$@"; do
    if [[ "${e}" == "${match}" ]]; then
      return 0
    fi
  done
  return 1
}

if ! is_contained "${GITHUB_EVENT_NAME}" "${ALLOWED_EVENTS[@]}"; then
  echo "Event type '${GITHUB_EVENT_NAME}' is not allowed on this runner. Aborting workflow."
  # clean up any nefarious stuff we may have fetched in job setup.
  cd ~/actions-runner/_work
  rm -rfv _actions/ _temp/
  exit 1
fi
