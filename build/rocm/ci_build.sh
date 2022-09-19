#!/usr/bin/env bash
# Copyright 2022 Google LLC
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

# Usage: ci_build.sh [--dockerfile <DOCKERFILE_PATH> --keep_image]
#                    <COMMAND>
#
# DOCKERFILE_PATH: (Optional) Path to the Dockerfile used for docker build.
#                  If this optional value is not supplied (via the --dockerfile flag)
#                  Dockerfile.rocm (located in the same directory as this script)
#                  will be used.
# KEEP_IMAGE: (Optional) If this flag is set, the container will be committed as an image
#
# COMMAND: Command to be executed in the docker container

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/build_common.sh"
CONTAINER_TYPE="rocm"

DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile.rocm"
DOCKER_CONTEXT_PATH="${SCRIPT_DIR}"
KEEP_IMAGE="--rm"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --dockerfile)
      DOCKERFILE_PATH="$2"
      DOCKER_CONTEXT_PATH=$(dirname "${DOCKERFILE_PATH}")
      shift 2
      ;;
    --keep_image)
      KEEP_IMAGE=""
      shift 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${DOCKERFILE_PATH}" ]]; then
  die "Invalid Dockerfile path: \"${DOCKERFILE_PATH}\""
fi

ROCM_EXTRA_PARAMS="--device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 16G"

# Helper function to traverse directories up until given file is found.
function upsearch (){
  test / == "$PWD" && return || \
    test -e "$1" && echo "$PWD" && return || \
    cd .. && upsearch "$1"
}

# Set up WORKSPACE and BUILD_TAG. Jenkins will set them for you or we pick
# reasonable defaults if you run it outside of Jenkins.
WORKSPACE="${WORKSPACE:-$(upsearch WORKSPACE)}"
BUILD_TAG="${BUILD_TAG:-jax_ci}"

# Determine the docker image name
DOCKER_IMG_NAME="${BUILD_TAG}.${CONTAINER_TYPE}"

# Under Jenkins matrix build, the build tag may contain characters such as
# commas (,) and equal signs (=), which are not valid inside docker image names.
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | sed -e 's/=/_/g' -e 's/,/-/g')

# Convert to all lower-case, as per requirement of Docker image names
DOCKER_IMG_NAME=$(echo "${DOCKER_IMG_NAME}" | tr '[:upper:]' '[:lower:]')

# Print arguments.
echo "WORKSPACE: ${WORKSPACE}"
echo "COMMAND: ${POSITIONAL_ARGS[*]}"
echo "BUILD_TAG: ${BUILD_TAG}"
echo "  (docker container name will be ${DOCKER_IMG_NAME})"
echo ""

echo "Building container (${DOCKER_IMG_NAME})..."
docker build -t ${DOCKER_IMG_NAME} \
    -f "${DOCKERFILE_PATH}" "${DOCKER_CONTEXT_PATH}"

# Check docker build status
if [[ $? != "0" ]]; then
  die "ERROR: docker build failed. Dockerfile is at ${DOCKERFILE_PATH}"
fi

# Run the command inside the container.
echo "Running '${POSITIONAL_ARGS[*]}' inside ${DOCKER_IMG_NAME}..."


docker run ${KEEP_IMAGE} --name ${DOCKER_IMG_NAME} --pid=host \
  -v ${WORKSPACE}:/workspace \
  -w /workspace \
  ${ROCM_EXTRA_PARAMS} \
  "${DOCKER_IMG_NAME}" \
  ${POSITIONAL_ARGS[@]}

if [[ "${KEEP_IMAGE}" != "--rm" ]] && [[ $? == "0" ]]; then
  echo "Committing the docker container as jax-rocm"
  docker stop ${DOCKER_IMG_NAME}
  docker commit ${DOCKER_IMG_NAME} jax-rocm
  docker rm ${DOCKER_IMG_NAME}
fi

echo "ROCm build was successful!"
