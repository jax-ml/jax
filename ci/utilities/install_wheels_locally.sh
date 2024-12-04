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
<<<<<<< HEAD
# Install wheels stored in `JAXCI_OUTPUT_DIR` locally using the Python binary
# set in JAXCI_PYTHON. Use the absolute path to the `find` utility to avoid
# using the Windows version of `find` on Msys.
=======
# Install wheels stored in `JAXCI_OUTPUT_DIR` on the system using the Python
# binary set in JAXCI_PYTHON. Use the absolute path to the `find` utility to
# avoid using the Windows version of `find` on Msys.
>>>>>>> 5ade371c88a1f879556ec29867b173da49ae57f0
WHEELS=( $(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \( -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" \)) )

if [[ -z "$WHEELS" ]]; then
  echo "ERROR: No wheels found under $JAXCI_OUTPUT_DIR"
  exit 1
fi

echo "Installing the following wheels:"
echo "${WHEELS[@]}"
"$JAXCI_PYTHON" -m pip install "${WHEELS[@]}"

echo "Installing the JAX package in editable mode at the current commit..."
# Install JAX package at the current commit.
"$JAXCI_PYTHON" -m pip install -U -e .
