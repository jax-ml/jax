#!/bin/bash
set -xe

CUDA_VERSION=$1

LIBCUDNN=libcudnn7
if [ $CUDA_VERSION = "10.2" ]; then
  NCCL_VERSION=2.5.6
  CUDNN_VERSION=7.6.5.32
elif [ $CUDA_VERSION = "11.0" ]; then
  NCCL_VERSION=2.7.3
  CUDNN_VERSION=8.0.0.180
  LIBCUDNN=libcudnn8
fi

echo "Installing cuda version $CUDA_VERSION"
echo "nccl version: $NCCL_VERSION"
echo "cudnn version: $CUDNN_VERSION"

apt-get update
apt-get remove -y --allow-change-held-packages cuda-license-10-0 libcudnn7 libnccl2
apt-get install -y --no-install-recommends \
  cuda-nvml-dev-$CUDA_VERSION \
  cuda-command-line-tools-$CUDA_VERSION \
  cuda-libraries-dev-$CUDA_VERSION \
  cuda-minimal-build-$CUDA_VERSION \
  libnccl2=$NCCL_VERSION-1+cuda$CUDA_VERSION \
  libnccl-dev=$NCCL_VERSION-1+cuda$CUDA_VERSION \
  $LIBCUDNN=$CUDNN_VERSION-1+cuda$CUDA_VERSION \
  $LIBCUDNN-dev=$CUDNN_VERSION-1+cuda$CUDA_VERSION
rm -f /usr/local/cuda
ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda
