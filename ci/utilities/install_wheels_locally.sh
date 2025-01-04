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
#
# Install wheels stored in `JAXCI_OUTPUT_DIR` on the system using the Python
# binary set in JAXCI_PYTHON. Use the absolute path to the `find` utility to
# avoid using the Windows version of `find` on Msys.
WHEELS=( $(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \( -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" \)) )

if [[ -z "$WHEELS" ]]; then
  echo "ERROR: No wheels found under $JAXCI_OUTPUT_DIR"
  exit 1
fi

echo "Installing the following wheels:"
echo "${WHEELS[@]}"

# On Windows, convert MSYS Linux-like paths to Windows paths.
if [[ $(uname -s) =~ "MSYS_NT" ]]; then
  "$JAXCI_PYTHON" -m pip install $(cygpath -w "${WHEELS[@]}")
else
  "$JAXCI_PYTHON" -m pip install "${WHEELS[@]}"
fi

echo "Installing the JAX package in editable mode at the current commit..."
# Install JAX package at the current commit.
"$JAXCI_PYTHON" -m pip install -U -e .
