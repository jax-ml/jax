#!/bin/bash
set -xev

PYTHON_VERSIONS="3.5.6 3.6.8 3.7.2 3.8.0"
CUDA_VERSIONS="9.2 10.0 10.1 10.2"
CUDA_VARIANTS="cuda" # "cuda-included"

mkdir -p dist

# build the pypi linux packages, tagging with manylinux1 for pypi reasons
docker build -t jaxbuild jax/build/
for PYTHON_VERSION in $PYTHON_VERSIONS
do
  mkdir -p dist/nocuda/
  docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION nocuda
  mv -f dist/*.whl dist/nocuda/
done

# build the cuda linux packages, tagging with linux_x86_64
for CUDA_VERSION in $CUDA_VERSIONS
do
  docker build -t jaxbuild jax/build/ --build-arg JAX_CUDA_VERSION=$CUDA_VERSION
  for PYTHON_VERSION in $PYTHON_VERSIONS
  do
    for CUDA_VARIANT in $CUDA_VARIANTS
    do
      mkdir -p dist/${CUDA_VARIANT}${CUDA_VERSION//.}
      docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION $CUDA_VARIANT
      mv -f dist/*.whl dist/${CUDA_VARIANT}${CUDA_VERSION//.}/
    done
  done
done
