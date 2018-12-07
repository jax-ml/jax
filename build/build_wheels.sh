#!/bin/bash -xev
JAXLIB_VERSION=$(sed -n "s/^ \+version=[']\(.*\)['],$/\\1/p" jax/build/setup.py)

PYTHON_VERSIONS="py2 py3"
CUDA_VERSIONS="9.2"  # "9.2 10.0"
CUDA_VARIANTS="cuda"  # "cuda cuda-included"

mkdir -p dist
for CUDA_VERSION in $CUDA_VERSIONS
do
  docker build -t jaxbuild jax/build/ --build-arg CUDA_VERSION=$CUDA_VERSION

  for PYTHON_VERSION in $PYTHON_VERSIONS
  do
    mkdir -p dist/nocuda/
    nvidia-docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION nocuda
    mv dist/*.whl dist/nocuda/jaxlib-${JAXLIB_VERSION}-${PYTHON_VERSION}-none-linux_x86_64.whl

    for CUDA_VARIANT in $CUDA_VARIANTS
    do
      mkdir -p dist/cuda${CUDA_VERSION//.}
      nvidia-docker run -it --tmpfs /build:exec --rm -v $(pwd)/dist:/dist jaxbuild $PYTHON_VERSION $CUDA_VARIANT
      mv dist/*.whl dist/cuda${CUDA_VERSION//.}/jaxlib-${JAXLIB_VERSION}-${PYTHON_VERSION}-none-linux_x86_64.whl
    done
  done
done

echo "now you might want to run something like:"
echo "python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/nocuda/*.whl --verbose"
