# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

chmod 1777 /tmp
DEBIAN_FRONTEND=noninteractive apt-get --allow-unauthenticated update 
DEBIAN_FRONTEND=noninteractive apt install -y wget software-properties-common
DEBIAN_FRONTEND=noninteractive apt-get clean all

apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
        python-is-python3 \
        libffi-dev \
        libssl-dev \
        build-essential \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev curl \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        wget && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*
