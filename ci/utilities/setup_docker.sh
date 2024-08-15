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
   [[ "$JAXCI_BUILD_PLUGIN_ENABLE" == 1 ]] || \
   [[ "$JAXCI_BUILD_PJRT_ENABLE" == 1 ]]; then
  JAXCI_DOCKER_ARGS="--gpus all"
  fi

  JAXCI_GIT_DIR=${JAXCI_GIT_DIR:-/tmpfs/src/jax}
  docker run $JAXCI_DOCKER_ARGS --name jax -w "/jax" -itd --rm \
      -v "$JAXCI_GIT_DIR:/jax" \
      -v $HOME/.config/gcloud:/root/.config/gcloud \
      -e JAXCI_OUTPUT_DIR=$JAXCI_OUTPUT_DIR \
      "$JAXCI_DOCKER_IMAGE" \
    bash
fi
jaxrun() { docker exec jax "$@"; }