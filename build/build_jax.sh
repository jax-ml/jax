#!/bin/bash
set -exv

tmp=/tmp/jax-build  # could mktemp -d but this way we can cache results
mkdir -p ${tmp}

## get bazel
bazel_dir=${tmp}/jax-bazel
if [ ! -d ${bazel_dir}/bin ]
then
    mkdir -p ${bazel_dir}
    curl -OL https://github.com/bazelbuild/bazel/releases/download/0.19.1/bazel-0.19.1-installer-linux-x86_64.sh
    chmod +x bazel-0.19.1-installer-linux-x86_64.sh
    ./bazel-0.19.1-installer-linux-x86_64.sh --prefix=${bazel_dir}
    rm bazel-0.19.1-installer-linux-x86_64.sh
fi
export PATH="${bazel_dir}/bin:$PATH"
# BUG: https://github.com/bazelbuild/bazel/issues/6665
handle_temporary_bazel_0_19_1_bug=1  # TODO(mattjj): remove with bazel 0.19.2

## get and configure tensorflow for building xla with gpu support
git clone https://github.com/tensorflow/tensorflow.git
pushd tensorflow
export PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-$(which python)}
export PYTHON_LIB_PATH=${SP_DIR:-$(python -m site --user-site)}
export USE_DEFAULT_PYTHON_LIB_PATH=1
export CUDA_TOOLKIT_PATH=${CUDA_PATH:-/usr/local/cuda}
export CUDNN_INSTALL_PATH=${CUDA_TOOLKIT_PATH}
export TF_CUDA_VERSION=$(readlink -f ${CUDA_TOOLKIT_PATH}/lib64/libcudart.so | cut -d '.' -f4-5)
export TF_CUDNN_VERSION=$(readlink -f ${CUDNN_INSTALL_PATH}/lib64/libcudnn.so | cut -d '.' -f4-5)
export TF_CUDA_COMPUTE_CAPABILITIES="3.0,3.5,5.2,6.0,6.1,7.0"
export TF_NEED_CUDA=1
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
export TF_ENABLE_XLA=1
export TF_NEED_MKL=0
export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
export TF_NEED_IGNITE=1
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_MPI=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_CUDA_CLANG=0
export TF_NEED_TENSORRT=0
export TF_NCCL_VERSION="2"
./configure
popd

## build xla inside tensorflow
mkdir -p ${PYTHON_LIB_PATH}
bazel_output_user_root=${tmp}/jax-bazel-output-user-root
bazel_output_base=${bazel_output_user_root}/output-base
bazel_opt="--output_user_root=${bazel_output_user_root} --output_base=${bazel_output_base} --bazelrc=tensorflow/tools/bazel.rc"
bazel_build_opt="-c opt --config=cuda"
if [ -n $handle_temporary_bazel_0_19_1_bug ]
then
  set +e
  bazel ${bazel_opt} build ${bazel_build_opt} jax:interactive 2> /dev/null
  sed -i 's/toolchain_identifier = "local"/toolchain_identifier = "local_linux"/' ${bazel_output_base}/external/local_config_cc/BUILD
  set -e
fi
bazel ${bazel_opt} build ${bazel_build_opt} jax:interactive

## extract the pieces we need
runfiles_prefix="execroot/__main__/bazel-out/k8-opt/bin/jax/interactive.runfiles/org_tensorflow/tensorflow"
cp ${bazel_output_base}/${runfiles_prefix}/libtensorflow_framework.so jax/lib/
cp ${bazel_output_base}/${runfiles_prefix}/compiler/xla/xla_data_pb2.py jax/lib/
cp ${bazel_output_base}/${runfiles_prefix}/compiler/xla/python/{xla_client.py,pywrap_xla.py,_pywrap_xla.so} jax/lib/

## rewrite some imports
sed -i 's/from tensorflow.compiler.xla.python import pywrap_xla as c_api/from . import pywrap_xla as c_api/' jax/lib/xla_client.py
sed -i 's/from tensorflow.compiler.xla import xla_data_pb2/from . import xla_data_pb2/' jax/lib/xla_client.py
sed -i '/from tensorflow.compiler.xla.service import hlo_pb2/d' jax/lib/xla_client.py

## clean up
rm -f bazel-*  # symlinks
rm -rf tensorflow
# rm -rf ${tmp}
