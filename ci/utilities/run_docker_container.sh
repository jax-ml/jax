#!/bin/bash
# Copyright 2024 The JAX Authors.
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
# Sets up a Docker container for JAX CI.
<<<<<<< HEAD

# This script creates and starts a Docker container named "jax" for internal
# JAX CI jobs. These jobs primarily handle building and publishing JAX artifacts
# to PyPI and/or GCS.

# Note: GitHub Actions workflows do not utilize this script, as they leverage
# built-in containerization features to run jobs within a container. However,
# they use the same Docker image to maintain consistency. This script also helps
# ensure that local build environments mirror the behavior of CI builds.
# Usage:
#     ./ci/utilities/run_docker_container.sh
#     docker exec -it jax <build-script>
#     E.g: docker exec -it jax ./ci/build_artifacts.sh jaxlib
=======
#
# This script creates and starts a Docker container named "jax" for internal
# JAX CI jobs.
#
# Note: While GitHub action workflows use the same Docker images, they do not
# run this script as they leverage built-in containerization features to run
# jobs within a container.
# Usage:
#     ./ci/utilities/run_docker_container.sh
#     docker exec jax <build-script>
#     E.g: docker exec jax ./ci/build_artifacts.sh jaxlib
>>>>>>> 74eca1346d0f0a4750bbd8b1b881a72680e69970
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

source ./ci/envs/docker.env

# Keep the existing "jax" container if it's already present.
if ! docker container inspect jax >/dev/null 2>&1 ; then
  # Simple retry logic for docker-pull errors. Sleeps if a pull fails.
  # Pulling an already-pulled container image will finish instantly, so
  # repeating the command costs nothing.
  docker pull "$JAXCI_DOCKER_IMAGE" || sleep 15
  docker pull "$JAXCI_DOCKER_IMAGE"

<<<<<<< HEAD
  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Docker on Windows doesn't support the `host` networking mode, and so
    # port-forwarding is required for the container to detect it's running on GCE.
=======
  # Docker on Windows doesn't support the `host` networking mode, and so
  # port-forwarding is required for the container to detect it's running on GCE.
  # This is needed for RBE configs to work.
  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
>>>>>>> 74eca1346d0f0a4750bbd8b1b881a72680e69970
    export IP_ADDR=$(powershell -command "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'vEthernet (nat)').IPAddress")
    netsh interface portproxy add v4tov4 listenaddress=$IP_ADDR listenport=80 connectaddress=169.254.169.254 connectport=80
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -e GCE_METADATA_HOST=$IP_ADDR"
  fi

<<<<<<< HEAD
  # Create a temporary file to store all JAXCI variables
=======
  # Create a temporary file to pass any user defined JAXCI_ variables to the
  # container.
>>>>>>> 74eca1346d0f0a4750bbd8b1b881a72680e69970
  JAXCI_TEMP_ENVFILE_DIR=$(mktemp)
  env | grep "JAXCI_" > "$JAXCI_TEMP_ENVFILE_DIR"

  # On Windows, convert MSYS Linux-like paths to Windows paths.
  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
<<<<<<< HEAD
    echo 'Converting MSYS Linux-like paths to Windows paths (for Bazel, Python, etc.)'
=======
    echo 'Converting MSYS Linux-like paths to Windows paths for setting up the Docker container'
>>>>>>> 74eca1346d0f0a4750bbd8b1b881a72680e69970
    # Convert all "JAXCI.*DIR" variables
    source <(python ./ci/utilities/convert_msys_paths_to_win_paths.py --convert $(env | grep "JAXCI.*DIR" | awk -F= '{print $1}'))
  fi

  # Start the container.
  docker run $JAXCI_DOCKER_ARGS --name jax \
<<<<<<< HEAD
          --env-file $JAXCI_TEMP_ENVFILE_DIR \
=======
          --env-file "$JAXCI_TEMP_ENVFILE_DIR" \
>>>>>>> 74eca1346d0f0a4750bbd8b1b881a72680e69970
          -w "$JAXCI_DOCKER_WORK_DIR" -itd --rm \
          -v "$JAXCI_JAX_GIT_DIR:$JAXCI_DOCKER_WORK_DIR" \
          "$JAXCI_DOCKER_IMAGE" \
          bash

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Allow requests from the container.
    CONTAINER_IP_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' jax)
    netsh advfirewall firewall add rule name="Allow Metadata Proxy" dir=in action=allow protocol=TCP localport=80 remoteip="$CONTAINER_IP_ADDR"
  fi
fi