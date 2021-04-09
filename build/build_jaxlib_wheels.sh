#!/bin/bash
set -xev

source "$(dirname $(realpath $0))/build_jaxlib_wheels_helpers.sh"

PYTHON_VERSIONS="3.6.8 3.7.2 3.8.0 3.9.0"
CUDA_VERSIONS="10.1 10.2 11.0 11.1 11.2"
CUDA_VARIANTS="cuda" # "cuda-included"

build_cuda_wheels "$PYTHON_VERSIONS" "$CUDA_VERSIONS" "$CUDA_VARIANTS"
build_nocuda_wheels "$PYTHON_VERSIONS"
