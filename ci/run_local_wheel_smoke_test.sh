#!/bin/bash
# Copyright 2026 The JAX Authors.
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

# Runs the local-wheel Bazel smoke test.
# Wrap the entire output in a GitHub actions group when on GHA, to differentiate
# the output from other test logs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

start_github_actions_group() {
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::group::$1"
  fi
}

end_github_actions_group() {
  if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    echo "::endgroup::"
  fi
}

cd "$repo_root"

expected_wheel_versions_json="$(
  python3 "$script_dir/parse_wheel_metadata.py" --wheel-dir="$repo_root/dist"
)"

start_github_actions_group "Local wheel smoke test"
echo "Running local wheel smoke test..."
if bazel "$@" \
    "--test_env=JAXCI_EXPECTED_WHEEL_VERSIONS_JSON=$expected_wheel_versions_json" \
    --test_output=all \
    //tests:local_wheel_smoke_test_gpu; then
  echo "Local wheel smoke test passed."
  end_github_actions_group
  exit 0
else
  smoke_status=$?
  echo "Local wheel smoke test failed."
  end_github_actions_group
  exit "$smoke_status"
fi
