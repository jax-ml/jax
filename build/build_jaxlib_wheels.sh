#!/bin/bash
set -xev
JAXLIB_VERSION=$(sed -n "s/^ \+version=[']\(.*\)['],$/\\1/p" jax/build/setup.py)

PYTHON_VERSIONS="py2 py3"
CUDA_VERSIONS="9.0 9.2 10.0"
CUDA_VARIANTS="cuda"  # "cuda cuda-included"

mkdir -p dist

# build the pypi linux packages, tagging with manylinux1 for pypi reasons
docker build -t jaxbuild jax/build/
for PYTHON_VERSION in $PYTHON_VERSIONS
do
  mkdir -p dist/nocuda/
  nvidia-docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION nocuda
  mv dist/*.whl dist/nocuda/jaxlib-${JAXLIB_VERSION}-${PYTHON_VERSION}-none-manylinux1_x86_64.whl
done

# build the cuda linux packages, tagging with linux_x86_64
for CUDA_VERSION in $CUDA_VERSIONS
do
  docker build -t jaxbuild jax/build/ --build-arg CUDA_VERSION=$CUDA_VERSION
  for PYTHON_VERSION in $PYTHON_VERSIONS
  do
    for CUDA_VARIANT in $CUDA_VARIANTS
    do
      mkdir -p dist/${CUDA_VARIANT}${CUDA_VERSION//.}
      nvidia-docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION $CUDA_VARIANT
      mv dist/*.whl dist/${CUDA_VARIANT}${CUDA_VERSION//.}/jaxlib-${JAXLIB_VERSION}-${PYTHON_VERSION}-none-linux_x86_64.whl
    done
  done
done
