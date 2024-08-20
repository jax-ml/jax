#!/bin/bash
# Copyright 2024 JAX Authors. All Rights Reserved.
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
#
# Set up Docker for JAX CI jobs.

# Keep the existing "jax" container if it's already present.
if ! docker container inspect jax >/dev/null 2>&1 ; then
  # Simple retry logic for docker-pull errors. Sleeps if a pull fails.
  # Pulling an already-pulled container image will finish instantly, so
  # repeating the command costs nothing.
  docker pull "$JAXCI_DOCKER_IMAGE" || sleep 15
  docker pull "$JAXCI_DOCKER_IMAGE"

  JAXCI_DOCKER_ARGS=""
  # Enable GPU on Docker if building or testing either the CUDA plugin or PJRT
  # on Linux x86 machines.
  if ( [[ $(uname -a) == *Linux*x86* ]] ) && \
   ( [[ "$JAXCI_BUILD_PLUGIN_ENABLE" == 1 ]] || \
   [[ "$JAXCI_BUILD_PJRT_ENABLE" == 1 ]] ); then
    JAXCI_DOCKER_ARGS="--gpus all"
  fi

  container_workdir_path="/jax"

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # On Windows, the paths need to be converted to Windows-style paths.
    export JAXCI_GIT_DIR=$(cygpath -w $JAXCI_GIT_DIR)
    export container_workdir_path=$(cygpath -w /c/jax/)
    export JAXCI_OUTPUT_DIR=$(cygpath -w /c/jax/pkg)

    # Docker on Windows doesn't support the `host` networking mode, and so
    # port-forwarding is required for the container to detect it's running on GCE.
    export IP_ADDR=$(powershell -command "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'vEthernet (nat)').IPAddress")
    netsh interface portproxy add v4tov4 listenaddress=$IP_ADDR listenport=80 connectaddress=169.254.169.254 connectport=80
    JAXCI_DOCKER_ARGS="-e GCE_METADATA_HOST=$IP_ADDR"
  fi

  docker run $JAXCI_DOCKER_ARGS --name jax -w $container_workdir_path -itd --rm \
      -v "$JAXCI_GIT_DIR:$container_workdir_path" \
      -e JAXCI_OUTPUT_DIR=$JAXCI_OUTPUT_DIR \
      "$JAXCI_DOCKER_IMAGE" \
    bash

  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    # Allow requests from the container.
    CONTAINER_IP_ADDR=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' jax)
    netsh advfirewall firewall add rule name="Allow Metadata Proxy" dir=in action=allow protocol=TCP localport=80 remoteip="$CONTAINER_IP_ADDR"
  fi
fi
jaxrun() { docker exec jax "$@"; }