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
  # Enable GPU on Docker if building or testing either the CUDA plugin or PJRT.
  # Only Linux x86 CI machines have GPUs attached to them. 
  if ( [[ $(uname -s) == "Linux" ]] && [[ $(uname -m) == "x86_64" ]] ) && \
   ( [[ "$JAXCI_BUILD_PLUGIN_ENABLE" == 1 ]] || \
   [[ "$JAXCI_BUILD_PJRT_ENABLE" == 1 ]] ); then
  JAXCI_DOCKER_ARGS="--gpus all"
  fi

  container_path="/jax"
  if [[ "$(uname -s)" =~ "MSYS_NT" ]]; then
    JAXCI_GIT_DIR=$(cygpath -w $JAXCI_GIT_DIR)
    container_path=$(cygpath -w /c/jax/)
    JAXCI_OUTPUT_DIR=$(cygpath -w /c/jax/pkg)
  fi

  if [[ `uname -s | grep -P '^MSYS_NT'` ]]; then
  # Docker on Windows doesn't support the `host` networking mode, and so
  # port-forwarding is required for the container to detect it's running on GCE.
  export IP_ADDR=$(powershell -command "(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'vEthernet (nat)').IPAddress")
  netsh interface portproxy add v4tov4 listenaddress=$IP_ADDR listenport=80 connectaddress=169.254.169.254 connectport=80
  # A local firewall rule for the container is added in
  # ci/official/utilities/setup_docker.sh.
  else
    # The volume mapping flag below shares the user's gcloud credentials, if any,
    # with the container, in case the user has credentials stored there.
    # This would allow Bazel to authenticate for RBE.
    # Note: TF's CI does not have any credentials stored there.
    JAXCI_DOCKER_ARGS="$JAXCI_DOCKER_ARGS -v $HOME/.config/gcloud:/root/.config/gcloud"
  fi

  JAXCI_GIT_DIR=${JAXCI_GIT_DIR:-/tmpfs/src/github/jax}
  docker run $JAXCI_DOCKER_ARGS --name jax -w $container_path -itd --rm \
      -v "$JAXCI_GIT_DIR:$container_path" \
      -e JAXCI_OUTPUT_DIR=$JAXCI_OUTPUT_DIR \
      "$JAXCI_DOCKER_IMAGE" \
    bash
fi
jaxrun() { docker exec jax "$@"; }