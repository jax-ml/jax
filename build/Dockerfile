FROM gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010
LABEL maintainer "Matt Johnson <mattjj@google.com>"

WORKDIR /
RUN apt-get update
RUN apt-get install libffi-dev
RUN git clone --branch v1.2.14 https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
RUN /pyenv/bin/pyenv install 2.7.15
RUN /pyenv/bin/pyenv install 3.5.6
RUN /pyenv/bin/pyenv install 3.6.8
RUN /pyenv/bin/pyenv install 3.7.2
RUN /pyenv/bin/pyenv install 3.8.0

# Install build-dependencies of scipy.
# TODO(phawkins): remove when there are scipy wheels for Python 3.8.
RUN apt-get update && apt-get install -y libopenblas-dev gfortran

# We pin numpy to a version < 1.16 to avoid version compatibility issues.
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 2.7.15 && pip install numpy==1.15.4 scipy cython setuptools wheel future six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.5.6 && pip install numpy==1.15.4 scipy cython setuptools wheel six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.6.8 && pip install numpy==1.15.4 scipy cython setuptools wheel six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.7.2 && pip install numpy==1.15.4 scipy cython setuptools wheel six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.8.0 && pip install numpy==1.17.3 scipy cython setuptools wheel six

# Change the CUDA version if it doesn't match the installed version.
ARG JAX_CUDA_VERSION=10.0
RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
  apt-get update && \
  apt-get remove -y --allow-change-held-packages cuda-license-10-0 libcudnn7 libnccl2 && \
  apt-get install -y --no-install-recommends \
      cuda-nvml-dev-$JAX_CUDA_VERSION \
      cuda-command-line-tools-$JAX_CUDA_VERSION \
      cuda-libraries-dev-$JAX_CUDA_VERSION \
      cuda-minimal-build-$JAX_CUDA_VERSION \
      libnccl2=$NCCL_VERSION-1+cuda$JAX_CUDA_VERSION \
      libnccl-dev=$NCCL_VERSION-1+cuda$JAX_CUDA_VERSION \
      libcudnn7=$CUDNN_VERSION-1+cuda$JAX_CUDA_VERSION \
      libcudnn7-dev=$CUDNN_VERSION-1+cuda$JAX_CUDA_VERSION && \
  rm -f /usr/local/cuda && \
  ln -s /usr/local/cuda-$JAX_CUDA_VERSION /usr/local/cuda; \
  fi'


WORKDIR /
COPY build_wheel_docker_entrypoint.sh /build_wheel_docker_entrypoint.sh
RUN chmod +x /build_wheel_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_wheel_docker_entrypoint.sh"]
