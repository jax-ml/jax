#!/bin/bash
# Script that builds wheels for a JAX release on Mac OS X.
# Builds wheels for multiple Python versions, using pyenv instead of Docker.
# Usage: run from root of JAX source tree as:
# build/build_wheels_macos.sh
# The wheels will end up in build/dist.
#
# Requires pyenv, pyenv-virtualenv (e.g., from Homebrew). If you have Homebrew
# installed, you can install these with:
# brew install pyenv pyenv-virtualenv
#
# May also need to install XCode command line tools to fix zlib build problem:
# https://github.com/pyenv/pyenv/issues/1219

eval "$(pyenv init -)"

PLATFORM_TAG="macosx_10_9_x86_64"

build_jax () {
  PY_VERSION="$1"
  PY_TAG="$2"
  echo "\nBuilding JAX for Python ${PY_VERSION}, tag ${PY_TAG}"
  pyenv install -s "${PY_VERSION}"
  VENV="jax-build-${PY_VERSION}"
  pyenv virtualenv "${PY_VERSION}" "${VENV}"
  pyenv activate "${VENV}"
  pip install scipy wheel
  rm -fr build/build
  python build/build.py
  cd build
  python setup.py bdist_wheel --python-tag "${PY_TAG}" --plat-name "${PLATFORM_TAG}"
  cd ..
  pyenv deactivate
  pyenv virtualenv-delete -f "${VENV}"
}


rm -fr build/dist
build_jax 2.7.15 cp27
build_jax 3.6.8 cp36
build_jax 3.7.2 cp37