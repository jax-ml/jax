#!/bin/bash
set -xe

CUDA_VERSION=$1

LIBCUDNN=libcudnn7
if [ $CUDA_VERSION = "10.0" ]; then
  CUBLAS=libcublas10
  CUBLAS_DEV=libcublas-dev
elif [ $CUDA_VERSION = "10.1" ]; then
  # Have to pin to libcublas10=10.2.1.243-1 due to bug in TF, see
  # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
  CUBLAS=libcublas10=10.2.1.243-1
  CUBLAS_DEV=libcublas-dev=10.2.1.243-1
elif [ $CUDA_VERSION = "10.2" ]; then
  CUBLAS=libcublas10
  CUBLAS_DEV=libcublas-dev
  NCCL_VERSION=2.5.6
  CUDNN_VERSION=7.6.5.32
elif [ $CUDA_VERSION = "11.0" ]; then
  CUBLAS=libcublas-11-0
  CUBLAS_DEV=libcublas-dev-11-0
  NCCL_VERSION=2.7.3
  CUDNN_VERSION=8.0.0.180
  LIBCUDNN=libcudnn8
elif [ $CUDA_VERSION = "11.1" ]; then
  CUBLAS=libcublas-11-1
  CUBLAS_DEV=libcublas-dev-11-1
  NCCL_VERSION=2.7.8
  CUDNN_VERSION=8.0.4.30
  LIBCUDNN=libcudnn8
else
  echo "Unsupported CUDA version: $CUDA_VERSION"
  exit 1
fi

echo "Installing cuda version: $CUDA_VERSION"
echo "nccl version: $NCCL_VERSION"
echo "cudnn version: $CUDNN_VERSION"

apt-get update
apt-get remove -y --allow-change-held-packages cuda-license-10-0 libcudnn7 libnccl2
apt-get install -y --no-install-recommends --allow-downgrades \
  $CUBLAS \
  $CUBLAS_DEV \
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
