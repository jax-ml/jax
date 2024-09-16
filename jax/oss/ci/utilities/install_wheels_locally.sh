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
# Install wheels stored in `JAXCI_OUTPUT_DIR` locally using the Python binary
# set in JAXCI_PYTHON.

# When running tests with Pytests, install wheels using the Python binary set
# in JAXCI_PYTHON.
if [[ $JAXCI_RUN_PYTEST_CPU == 1 ]] || [[ $JAXCI_RUN_PYTEST_GPU == 1 ]]; then
  # Install the wheels inside the output directory. On Windows, the path needs
  # to be in Linux format.
  if [[ ! $(uname -s) =~ "MSYS_NT" ]]; then
    bash -c "$JAXCI_PYTHON -m pip install $JAXCI_OUTPUT_DIR/*.whl"
  else
    bash -c "$JAXCI_PYTHON -m pip install $(cygpath $JAXCI_OUTPUT_DIR)/*.whl"
  fi

  # Install JAX package at the current commit.
  "$JAXCI_PYTHON" -m pip install -U -e .
fi
