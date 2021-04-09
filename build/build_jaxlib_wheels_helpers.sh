#!/bin/bash

build_cuda_wheels() {
  local PYTHON_VERSIONS=$1
  local CUDA_VERSIONS=$2
  local CUDA_VARIANTS=$3
  mkdir -p dist
  for CUDA_VERSION in $CUDA_VERSIONS
  do
    docker build -t jaxbuild jax/build/ --build-arg JAX_CUDA_VERSION=$CUDA_VERSION
    for PYTHON_VERSION in $PYTHON_VERSIONS
    do
      for CUDA_VARIANT in $CUDA_VARIANTS
      do
        mkdir -p dist/${CUDA_VARIANT}${CUDA_VERSION//.}
        docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION $CUDA_VARIANT $CUDA_VERSION
        mv -f dist/*.whl dist/${CUDA_VARIANT}${CUDA_VERSION//.}/
      done
    done
  done
}

build_nocuda_wheels() {
  local PYTHON_VERSIONS=$1
  mkdir -p dist
  docker build -t jaxbuild jax/build/
  for PYTHON_VERSION in $PYTHON_VERSIONS
  do
    mkdir -p dist/nocuda/
    docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION nocuda
    mv -f dist/*.whl dist/nocuda/
  done
}
