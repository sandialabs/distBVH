FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN apt update \
    && DEBIAN_FRONTEND="noninteractive" apt install -y \
        bzip2 \
        git \
        gpg \
        gpgconf \
        libmpich-dev \
        libncurses5-dev \
        libsigsegv-dev \
        libsigsegv2 \
        m4 \
        perl \
        pkg-config \
        python3 \
        python3-distutils \
        python3-pip \
        software-properties-common \
        wget \
        xz-utils \
        zip \
        zlib1g \
        zlib1g-dev \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt update \
    && DEBIAN_FRONTEND="noninteractive" apt install -y \
        cmake-data=3.26.4-0kitware1ubuntu20.04.1 \
        cmake=3.26.4-0kitware1ubuntu20.04.1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install clingo

# Now we install spack and find compilers/externals
RUN mkdir -p /opt/ && cd /opt/ && git clone --depth 1 --branch "v0.22.0" https://github.com/spack/spack.git

# Add current source dir into the image
COPY . /opt/src/ci-images

# Get the latest version of the darma-vt repo
RUN cd /opt/src/ci-images/spack-repos && git clone --depth 1 --branch "16-external-fmt" https://github.com/DARMA-tasking/spack-package.git vt

# Add our new repos
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack repo add /opt/src/ci-images/spack-repos/p3a \
  && spack repo add /opt/src/ci-images/spack-repos/vt

# Setup our environment
RUN mkdir -p /opt/spack-env && mv /opt/src/ci-images/spack-cuda.yaml /opt/spack-env/spack.yaml
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack --env-dir /opt/spack-env concretize
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack --env-dir /opt/spack-env install --fail-fast \
  && spack --env-dir /opt/spack-env gc -y
