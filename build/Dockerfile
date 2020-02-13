FROM gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010
LABEL maintainer "Matt Johnson <mattjj@google.com>"

WORKDIR /
RUN rm /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
RUN apt-get update
RUN apt-get install libffi-dev
RUN git clone --branch v1.2.14 https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
RUN /pyenv/bin/pyenv install 3.5.6
RUN /pyenv/bin/pyenv install 3.6.8
RUN /pyenv/bin/pyenv install 3.7.2
RUN /pyenv/bin/pyenv install 3.8.0

# We pin numpy to a version < 1.16 to avoid version compatibility issues.
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.5.6 && pip install numpy==1.15.4 scipy cython setuptools wheel six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.6.8 && pip install numpy==1.15.4 scipy cython setuptools wheel six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.7.2 && pip install numpy==1.15.4 scipy cython setuptools wheel six
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.8.0 && pip install numpy==1.17.3 scipy cython setuptools wheel six

# Change the CUDA version if it doesn't match the installed version.
ARG JAX_CUDA_VERSION=10.0
COPY install_cuda.sh /install_cuda.sh
RUN chmod +x /install_cuda.sh
RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
  /install_cuda.sh $JAX_CUDA_VERSION; \
  fi'


WORKDIR /
COPY build_wheel_docker_entrypoint.sh /build_wheel_docker_entrypoint.sh
RUN chmod +x /build_wheel_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_wheel_docker_entrypoint.sh"]
