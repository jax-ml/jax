ARG CUDA_VERSION=9.2
FROM nvidia/cuda:$CUDA_VERSION-cudnn7-devel-ubuntu16.04
LABEL maintainer "Matt Johnson <mattjj@google.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
            dh-autoreconf git curl \
            build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
            libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev \
            libxml2-dev libxmlsec1-dev libffi-dev openjdk-8-jdk curl \
            bash-completion unzip python

RUN wget "https://github.com/bazelbuild/bazel/releases/download/0.24.0/bazel_0.24.0-linux-x86_64.deb" && \
    dpkg -i bazel_0.24.0-linux-x86_64.deb

RUN git clone https://github.com/nixos/patchelf /tmp/patchelf
WORKDIR /tmp/patchelf
RUN bash bootstrap.sh && ./configure && make && make install && rm -r /tmp/patchelf


WORKDIR /
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
RUN /pyenv/bin/pyenv install 2.7.15
RUN /pyenv/bin/pyenv install 3.5.6
RUN /pyenv/bin/pyenv install 3.6.8
RUN /pyenv/bin/pyenv install 3.7.2

# We pin numpy to a version < 1.16 to avoid version compatibility issues.
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 2.7.15 && pip install numpy==1.15.4 scipy cython setuptools wheel future
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.5.6 && pip install numpy==1.15.4 scipy cython setuptools wheel
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.6.8 && pip install numpy==1.15.4 scipy cython setuptools wheel
RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.7.2 && pip install numpy==1.15.4 scipy cython setuptools wheel


WORKDIR /
COPY build_wheel_docker_entrypoint.sh /build_wheel_docker_entrypoint.sh
RUN chmod +x /build_wheel_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_wheel_docker_entrypoint.sh"]
