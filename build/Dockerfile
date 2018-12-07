ARG CUDA_VERSION=9.2
FROM nvidia/cuda:$CUDA_VERSION-cudnn7-devel-ubuntu16.04
LABEL maintainer "Matt Johnson <mattjj@google.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
            dh-autoreconf git curl \
            python python-pip python-dev \
            python3 python3-pip python3-dev
RUN pip install numpy setuptools wheel && pip3 install numpy setuptools wheel

RUN git clone https://github.com/nixos/patchelf /tmp/patchelf
WORKDIR /tmp/patchelf
RUN bash bootstrap.sh && ./configure && make && make install && rm -r /tmp/patchelf

WORKDIR /
RUN curl -O https://raw.githubusercontent.com/google/jax/762abcf29b4a155c3de325c27ecffa5d4a3da28c/build/build_wheel_docker_entrypoint.sh
RUN chmod +x /build_wheel_docker_entrypoint.sh

WORKDIR /build
ENV TEST_TMPDIR /build
ENTRYPOINT ["/build_wheel_docker_entrypoint.sh"]
