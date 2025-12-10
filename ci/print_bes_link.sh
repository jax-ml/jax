#!/bin/bash
# Copyright 2025 The JAX Authors.
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

if [[ -n "$OUTPUT_BEP_FILE" ]]; then
  url="$(grep -m1 -o 'https://source\.cloud\.google\.com/results/invocations/[a-zA-Z0-9-]*' $OUTPUT_BEP_FILE)" || true # don't allow a pipe fail on this
  if [[ -n "$url" ]]; then
    echo "BES link: $url" >> $GITHUB_STEP_SUMMARY
    echo "::notice:: BES link: $url"

  else
    echo "Could not parse build id from the invocation" 
  fi
else
    echo "No BEP link is propagated"
fi
