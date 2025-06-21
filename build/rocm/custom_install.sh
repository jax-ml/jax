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
