#!/bin/bash
set -exv

# For a build with CUDA, from the repo root run:
#   bash build/build_jax.sh
# For building without CUDA (CPU-only), instead run:
#   JAX_BUILD_WITH_CUDA=0 bash build/build_jax.sh
# To clean intermediate results, run
#   rm -rf /tmp/jax-build/jax-bazel-output-user-root
# To clean everything, run
#   rm -rf /tmp/jax-build

JAX_BUILD_WITH_CUDA=${JAX_BUILD_WITH_CUDA:-1}

init_commit=a30e858e59d7184b9e54dc3f3955238221d70439
if [[ ! -d .git || $(git rev-list --parents HEAD | tail -1) != ${init_commit} ]]
then
  (>&2 echo "must be executed from jax repo root")
  exit 1
fi

tmp=/tmp/jax-build  # could mktemp -d but this way we can cache results
mkdir -p ${tmp}

## get bazel
bazel_dir=${tmp}/jax-bazel
if [ ! -d ${bazel_dir}/bin ]
then
    mkdir -p ${bazel_dir}
    case "$(uname -s)" in
      Linux*) installer=bazel-0.19.2-installer-linux-x86_64.sh;;
      Darwin*) installer=bazel-0.19.2-installer-darwin-x86_64.sh;;
      *) exit 1;;
    esac
    curl -OL https://github.com/bazelbuild/bazel/releases/download/0.19.2/${installer}
    chmod +x ${installer}
    bash ${installer} --prefix=${bazel_dir}
    rm ${installer}
fi
export PATH="${bazel_dir}/bin:$PATH"

## get and configure tensorflow for building xla
if [[ ! -d tensorflow ]]
then
  git clone https://github.com/tensorflow/tensorflow.git
fi
pushd tensorflow
export PYTHON_BIN_PATH=${PYTHON_BIN_PATH:-$(which python)}
export PYTHON_LIB_PATH=${SP_DIR:-$(python -m site --user-site)}
export USE_DEFAULT_PYTHON_LIB_PATH=1
if [[ ${JAX_BUILD_WITH_CUDA} != 0 ]]
then
  export CUDA_TOOLKIT_PATH=${CUDA_PATH:-/usr/local/cuda}
  export CUDNN_INSTALL_PATH=${CUDA_TOOLKIT_PATH}
  export TF_CUDA_VERSION=$(readlink -f ${CUDA_TOOLKIT_PATH}/lib64/libcudart.so | cut -d '.' -f4-5)
  export TF_CUDNN_VERSION=$(readlink -f ${CUDNN_INSTALL_PATH}/lib64/libcudnn.so | cut -d '.' -f4-5)
  export TF_CUDA_COMPUTE_CAPABILITIES="3.0,3.5,5.2,6.0,6.1,7.0"
  export TF_NEED_CUDA=1
else
  export TF_NEED_CUDA=0
fi
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
if [[ ${JAX_BUILD_WITH_CUDA} != 0 ]]
then
  bazel_build_opt="-c opt --config=cuda"
else
  bazel_build_opt="-c opt"
fi
bazel ${bazel_opt} build ${bazel_build_opt} jax:build_jax

## extract the pieces we need
runfiles_prefix="execroot/__main__/bazel-out/k8-opt/bin/jax/build_jax.runfiles/org_tensorflow/tensorflow"
cp -f ${bazel_output_base}/${runfiles_prefix}/libtensorflow_framework.so jax/lib/
cp -f ${bazel_output_base}/${runfiles_prefix}/compiler/xla/xla_data_pb2.py jax/lib/
cp -f ${bazel_output_base}/${runfiles_prefix}/compiler/xla/python/{xla_client.py,pywrap_xla.py,_pywrap_xla.so} jax/lib/

## rewrite some imports
sed -i 's/from tensorflow.compiler.xla.python import pywrap_xla as c_api/from . import pywrap_xla as c_api/' jax/lib/xla_client.py
sed -i 's/from tensorflow.compiler.xla import xla_data_pb2/from . import xla_data_pb2/' jax/lib/xla_client.py
sed -i '/from tensorflow.compiler.xla.service import hlo_pb2/d' jax/lib/xla_client.py

## clean up
rm -f bazel-*  # symlinks
rm -rf tensorflow
rm -rf ${bazel_output_user_root}  # clean build results
# rm -rf ${tmp}  # clean everything, including the bazel binary
