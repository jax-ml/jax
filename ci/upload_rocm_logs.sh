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
# Upload ROCm pytest artifacts to S3.
#
# Artifacts:
#   - logs.tar.gz: archive of the local logs/ directory (pytest JSON + future logs)
#   - run-manifest.json: small run metadata for indexing/debugging
#   - _SUCCESS: written last to indicate the upload set is complete
#
# S3 layout (deterministic + unique per run/attempt):
#   <org>/<repo>/<branch>/<nightly|continuous>/<DATE>_<run_id>_<attempt>/<combo>/
set -euo pipefail

: "${S3_BUCKET_NAME:?}"
: "${INPUT_PYTHON:?}"
: "${INPUT_ROCM_VERSION:?}"
: "${INPUT_ROCM_TAG:?}"
: "${INPUT_RUNNER:?}"
: "${IS_NIGHTLY:?}"  # nightly|continuous

TEST_LOGS_ROOT="jax-ci-test-logs"

norm() { printf '%s' "$1" | tr '.-' '_' ; }

# Timestamp: GitHub run_started_at from public Actions API; fall back to system date.
RUN_STARTED_AT="$(
  curl -fsSL \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}" \
  | sed -n 's/.*"run_started_at"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  | head -n1 || true
)"

DATE="${RUN_STARTED_AT%%T*}"
[[ -n "${DATE}" ]] || DATE="$(date -u +%F)"

# GPU count from runner name (e.g. linux-x86-64-8gpu-amd -> 8).
GPU_COUNT=""
if [[ "${INPUT_RUNNER}" =~ ([0-9]+)gpu ]]; then
  GPU_COUNT="${BASH_REMATCH[1]}"
fi

GPU_PART="${GPU_COUNT:+gpu_${GPU_COUNT}}"
GPU_PART="${GPU_PART:-${INPUT_RUNNER}}"

RUN_KEY="${DATE}_${GITHUB_RUN_ID}_${GITHUB_RUN_ATTEMPT}"
COMBO="py$(norm "${INPUT_PYTHON}")-rocm$(norm "${INPUT_ROCM_VERSION}")-${GPU_PART}"
PREFIX="${GITHUB_REPOSITORY}/${GITHUB_REF_NAME}/${IS_NIGHTLY}/${RUN_KEY}/${COMBO}"

DEST="s3://${S3_BUCKET_NAME}/${TEST_LOGS_ROOT}/${PREFIX}"

echo "Uploading ROCm pytest artifacts"

# Upload archive first (created in YAML)
ARCHIVE="logs.tar.gz"
[[ -f "${ARCHIVE}" ]] || { echo "Missing ${ARCHIVE}"; exit 2; }

echo "Uploading logs.tar.gz"
aws s3 cp --only-show-errors "${ARCHIVE}" "${DEST}/${ARCHIVE}"

PYTHON="${JAXCI_PYTHON:-python3}"
# Packages/wheels (best-effort)
PKGS_RAW="$(
  "${PYTHON}" -m pip list --format=freeze 2>/dev/null \
  | grep -E '^(jax|jaxlib)==|pjrt|plugin' \
  || true
)"
PKGS_ONE_LINE="$(printf "%s" "${PKGS_RAW}" | tr '\n' '|' | sed 's/|$//')"

WHEELS_RAW="$(sha256sum dist/*.whl 2>/dev/null || true)"
WHEELS_ONE_LINE="$(printf "%s" "${WHEELS_RAW}" | tr '\n' '|' | sed 's/|$//')"

# Base image digest (best-effort)
GHCR_REPO="rocm/jax-base-ubu24.${INPUT_ROCM_TAG}"
IMAGE="ghcr.io/${GHCR_REPO}:latest"
DIGEST=""

TOKEN="$(
  curl -fsSL "https://ghcr.io/token?service=ghcr.io&scope=repository:${GHCR_REPO}:pull" \
  | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
  || true
)"
if [[ -n "${TOKEN}" ]]; then
  DIGEST="$(
    curl -fsSL -D - \
      -H "Authorization: Bearer ${TOKEN}" \
      -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
      "https://ghcr.io/v2/${GHCR_REPO}/manifests/latest" -o /dev/null \
    | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}' \
    | tr -d $'\r' \
    | head -n1 || true
  )"
fi

RUN_URL="https://github.com/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"

echo "Uploading run-manifest.json"
aws s3 cp --only-show-errors - "${DEST}/run-manifest.json" <<EOF
{
  "schema_version": 1,
  "run_started_at": "${RUN_STARTED_AT}",
  "run_completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "github_run_url": "${RUN_URL}",
  "github_repository": "${GITHUB_REPOSITORY}",
  "github_ref_name": "${GITHUB_REF_NAME}",
  "github_ref": "${GITHUB_REF}",
  "github_sha": "${GITHUB_SHA}",
  "github_event_name": "${GITHUB_EVENT_NAME}",
  "github_run_id": "${GITHUB_RUN_ID}",
  "github_run_attempt": "${GITHUB_RUN_ATTEMPT}",
  "github_run_number": "${GITHUB_RUN_NUMBER}",
  "github_workflow": "${GITHUB_WORKFLOW}",
  "is_nightly": "${IS_NIGHTLY}",
  "github_job": "${GITHUB_JOB}",
  "python_version": "${INPUT_PYTHON}",
  "rocm_version": "${INPUT_ROCM_VERSION}",
  "rocm_tag": "${INPUT_ROCM_TAG}",
  "gpu_count": ${GPU_COUNT:-null},
  "runner": "${INPUT_RUNNER}",
  "base_image_name": "${IMAGE}",
  "base_image_digest": "${DIGEST}",
  "jax_packages_raw": "${PKGS_ONE_LINE}",
  "wheels_sha_raw": "${WHEELS_ONE_LINE}"
}
EOF

echo "Writing _SUCCESS"
printf '' | aws s3 cp --only-show-errors - "${DEST}/_SUCCESS"
