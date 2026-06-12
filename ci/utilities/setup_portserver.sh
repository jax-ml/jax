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
# distributed under/the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Starts the JAX portserver in the background and configures cleanup on exit.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: ci/utilities/setup_portserver.sh must be sourced, not executed directly." >&2
  exit 1
fi

if [[ ! -f "build/portserver.py" ]]; then
  echo "ERROR: build/portserver.py not found. Make sure you are in the JAX root directory." >&2
  exit 1
fi

echo "::group::Start Portserver" >&2
${PYTHON_BIN:-python3} build/portserver.py &
SERVER_PID=$!
cleanup_portserver() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_portserver EXIT
echo "::endgroup::" >&2
