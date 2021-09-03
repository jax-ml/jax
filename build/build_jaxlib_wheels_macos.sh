#!/bin/bash
set -e

# Script that builds wheels for a JAX release on Mac OS X.
# Builds wheels for multiple Python versions, using pyenv instead of Docker.
# Usage: run from root of JAX source tree as:
# build/build_jaxlib_wheels_macos.sh
# The wheels will end up in dist/
#
# Requires pyenv, pyenv-virtualenv (e.g., from Homebrew). If you have Homebrew
# installed, you can install these with:
# brew install pyenv pyenv-virtualenv
#
# May also need to install XCode command line tools to fix zlib build problem:
# https://github.com/pyenv/pyenv/issues/1219

if ! pyenv --version 2>/dev/null ;then
  echo "Error: You need to install pyenv and pyenv-virtualenv"
  exit 1
fi
eval "$(pyenv init -)"

build_jax () {
  PY_VERSION="$1"
  NUMPY_VERSION="$2"
  echo -e "\nBuilding JAX for Python ${PY_VERSION}"
  echo "NumPy version ${NUMPY_VERSION}"
  pyenv install -s "${PY_VERSION}"
  VENV="jax-build-${PY_VERSION}"
  if pyenv virtualenvs | grep "${VENV}" ;then
    pyenv virtualenv-delete -f "${VENV}"
  fi
  pyenv virtualenv "${PY_VERSION}" "${VENV}"
  pyenv activate "${VENV}"
  # We pin the Numpy wheel to a version < 1.16.0 for Python releases prior to
  # 3.8, because Numpy extensions built at 1.16.0 are not backward compatible to
  # earlier Numpy versions.
  pip install numpy==$NUMPY_VERSION wheel future six
  rm -fr build/build
  python build/build.py --output_path=dist/
  pyenv deactivate
  pyenv virtualenv-delete -f "${VENV}"
}


rm -fr dist
build_jax 3.7.2 1.18.5
build_jax 3.8.0 1.18.5
build_jax 3.9.0 1.19.4
