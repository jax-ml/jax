#!/bin/bash
# Script that builds wheels for a JAX release on Mac OS X.
# Builds wheels for multiple Python versions, using pyenv instead of Docker.
# Usage: run from root of JAX source tree as:
# build/build_jaxlib_wheels_macos.sh
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
  NUMPY_VERSION="$3"
  SCIPY_VERSION="$4"
  echo "\nBuilding JAX for Python ${PY_VERSION}, tag ${PY_TAG}"
  echo "NumPy version ${NUMPY_VERSION}, SciPy version ${SCIPY_VERSION}"
  pyenv install -s "${PY_VERSION}"
  VENV="jax-build-${PY_VERSION}"
  pyenv virtualenv-delete -f "${VENV}"
  pyenv virtualenv "${PY_VERSION}" "${VENV}"
  pyenv activate "${VENV}"
  # We pin the Numpy wheel to a version < 1.16.0 for Python releases prior to
  # 3.8, because Numpy extensions built at 1.16.0 are not backward compatible to
  # earlier Numpy versions.
  pip install numpy==$NUMPY_VERSION scipy==$SCIPY_VERSION wheel future six
  rm -fr build/build
  python build/build.py
  cd build
  python setup.py bdist_wheel --python-tag "${PY_TAG}" --plat-name "${PLATFORM_TAG}"
  cd ..
  pyenv deactivate
  pyenv virtualenv-delete -f "${VENV}"
}


rm -fr build/dist
build_jax 3.6.8 cp36 1.15.4 1.2.0
build_jax 3.7.2 cp37 1.15.4 1.2.0
build_jax 3.8.0 cp38 1.17.3 1.3.2
