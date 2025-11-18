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

WHEELS=( $(/usr/bin/find "$JAXCI_OUTPUT_DIR/" -type f \(  -name "*jax*py3*" -o -name "*jaxlib*" -o -name "*jax*cuda*pjrt*" -o -name "*jax*cuda*plugin*" \)) )

for i in "${!WHEELS[@]}"; do
  if [[ "${WHEELS[$i]}" == *jax*py3*none*any.whl ]]; then
    # Append an extra to the end of the JAX wheel path to install those
    # packages as well from PyPI. E.g. jax[tpu] will install the libtpu package
    # from PyPI. See ci/envs/README.md for more details.
    if [[ -n "$JAXCI_JAX_PYPI_EXTRAS" ]]; then
      WHEELS[$i]="${WHEELS[$i]}[$JAXCI_JAX_PYPI_EXTRAS]"
    fi
  fi
done

if [[ -n "${WHEELS[@]}" ]]; then
  echo "Installing the following wheels:"
  echo "${WHEELS[@]}"

  # Install `uv` if it's not already installed. `uv` is much faster than pip for
  # installing Python packages.
  if ! command -v uv >/dev/null 2>&1; then
    pip install uv~=0.5.30
  fi

  # On Windows, convert MSYS Linux-like paths to Windows paths.
  if [[ $(uname -s) =~ "MSYS_NT" ]]; then
    "$JAXCI_PYTHON" -m uv pip install $(cygpath -w "${WHEELS[@]}")
  else
    "$JAXCI_PYTHON" -m uv pip install "${WHEELS[@]}"
  fi
else
  # Note that we don't exit here because the wheels may have been installed
  # earlier in a different step in the CI job.
  echo "INFO: No wheels found under $JAXCI_OUTPUT_DIR"
  echo "INFO: Skipping local wheel installation."
fi