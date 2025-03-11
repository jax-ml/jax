#!/bin/bash

# Check for user-supplied arguments.
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <jax_home_directory> <version>"
    exit 1
fi

# Set JAX_HOME and RELEASE_VERSION from user arguments.
JAX_HOME=$1
RELEASE_VERSION=$2
WHEELHOUSE="$JAX_HOME/wheelhouse"

# Projects to upload separately to PyPI.
PROJECTS=("jax_rocm60_pjrt" "jax_rocm60_plugin")

# PyPI API Token.
PYPI_API_TOKEN=${PYPI_API_TOKEN:-"pypi-replace_with_token"}

# Ensure the specified JAX_HOME and wheelhouse directories exists.
if [[ ! -d "$JAX_HOME" ]]; then
    echo "Error: The specified JAX_HOME directory does not exist: $JAX_HOME"
    exit 1
fi
if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "Error: The wheelhouse directory does not exist: $WHEELHOUSE"
    exit 1
fi

upload_and_release_project() {
    local project=$1

    echo "Searching for wheels matching project: $project version: $RELEASE_VERSION..."
    wheels=($(ls $WHEELHOUSE | grep "^${project}-${RELEASE_VERSION}[.-].*\.whl"))
    if [[ ${#wheels[@]} -eq 0 ]]; then
        echo "No wheels found for project: $project version: $RELEASE_VERSION. Skipping..."
        return
    fi
    echo "Found wheels for $project: ${wheels[*]}"

    echo "Uploading wheels for $project version $RELEASE_VERSION to PyPI..."
    for wheel in "${wheels[@]}"; do
        twine upload --verbose --repository pypi --non-interactive --username "__token__" --password "$PYPI_API_TOKEN" "$WHEELHOUSE/$wheel"
    done
}

# Install twine if not already installed.
python -m pip install --upgrade twine

# Iterate over each project and upload its wheels.
for project in "${PROJECTS[@]}"; do
    upload_and_release_project $project
done
