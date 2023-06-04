################################################################################
FROM rocm/dev-ubuntu-20.04:5.4-complete as rt_build
MAINTAINER Rahul Batra<rahbatra@amd.com>
################################################################################
ARG ROCM_PATH=/opt/rocm-5.4.0

ARG DEBIAN_FRONTEND=noninteractive
ENV HOME /root/
ENV ROCM_PATH=$ROCM_PATH

RUN apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  software-properties-common \
  clang-6.0 \
  clang-format-6.0 \
  curl \
  g++-multilib \
  git \
  vim \
  libnuma-dev \
  virtualenv \
  python3-pip \
  pciutils \
  wget && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


# Set up paths
ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/hip
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"

# Add target file to help determine which device(s) to build for
RUN bash -c 'echo -e "gfx900\ngfx906\ngfx908\ngfx90a\ngfx1030" >> ${ROCM_PATH}/bin/target.lst'

# Need to explicitly create the $ROCM_PATH/.info/version file to workaround what seems to be a bazel bug
# The env vars being set via --action_env in .bazelrc and .tf_configure.bazelrc files are sometimes
# not getting set in the build command being spawned by bazel (in theory this should not happen)
# As a consequence ROCM_PATH is sometimes not set for the hipcc commands.
# When hipcc incokes hcc, it specifies $ROCM_PATH/.../include dirs via the `-isystem` options
# If ROCM_PATH is not set, it defaults to /opt/rocm, and as a consequence a dependency is generated on the
# header files included within `/opt/rocm`, which then leads to bazel dependency errors
# Explicitly creating the $ROCM_PATH/.info/version allows ROCM path to be set correrctly, even when ROCM_PATH
# is not explicitly set, and thus avoids the eventual bazel dependency error.
# The bazel bug needs to be root-caused and addressed, but that is out of our control and may take a long time
# to come to fruition, so implementing the workaround to make do till then
# Filed https://github.com/bazelbuild/bazel/issues/11163 for tracking this
RUN touch ${ROCM_PATH}/.info/version

ENV PATH="/root/bin:/root/.local/bin:$PATH"


# Install python3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt update && \
  apt install -y python3.9-dev \
    python3-pip \
    python3.9-distutils \
    python-is-python3

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN pip3 install --upgrade --force-reinstall setuptools pip

RUN pip3 install absl-py numpy==1.20.0 scipy wheel six setuptools pytest pytest-rerunfailures matplotlib

# Get jax and build it with ROCm
RUN git clone https://github.com/google/jax.git 

################################################################################
FROM rt_build as ci_build
################################################################################
WORKDIR /jax
RUN ./build/rocm/build_rocm.sh 
RUN ./build/rocm/run_single_gpu.py 
RUN ./build/rocm/run_multi_gpu.sh
