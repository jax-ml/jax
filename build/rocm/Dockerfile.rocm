FROM ubuntu:bionic
MAINTAINER Reza Rahimi <reza.rahimi@amd.com>

ARG ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/5.1/
ARG ROCM_BUILD_NAME=ubuntu
ARG ROCM_BUILD_NUM=main
ARG ROCM_PATH=/opt/rocm-5.1.0

ARG DEBIAN_FRONTEND=noninteractive
ENV HOME /root/
ENV ROCM_PATH=$ROCM_PATH

RUN apt-get --allow-unauthenticated update && apt install -y wget software-properties-common
RUN apt-get clean all
RUN wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -;
RUN bin/bash -c 'if [[ $ROCM_DEB_REPO == http://repo.radeon.com/rocm/*  ]] ; then \
      echo "deb [arch=amd64] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" > /etc/apt/sources.list.d/rocm.list; \
    else \
      echo "deb [arch=amd64 trusted=yes] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" > /etc/apt/sources.list.d/rocm.list ; \
    fi'


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

# Add to get ppa
RUN apt-get update
RUN apt-get install -y software-properties-common
# Install rocm pkgs
RUN apt-get update --allow-insecure-repositories && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    rocm-dev rocm-libs rccl && \
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
    python3.9-distutils

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

RUN pip3 install --upgrade setuptools pip
  
RUN pip3 install absl-py numpy==1.19.5 scipy wheel six setuptools pytest pytest-rerunfailures
  
