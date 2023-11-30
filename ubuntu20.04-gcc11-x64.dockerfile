FROM ubuntu:20.04 as build_dependencies-stage

RUN apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
      git \
      python3 \
      python3-pip \
      python3-distutils \
      xz-utils \
      bzip2 \
      zip \
      gpg \
      wget \
      gpgconf \
      software-properties-common \
      libsigsegv2 \
      libsigsegv-dev \
      pkg-config \
      zlib1g \
      zlib1g-dev \
      m4 \
  && rm -rf /var/lib/apt/lists/*

# Cmake ppa
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

# gcc ppa
RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-get update \
  && apt-get install -y \
     gcc-11=11.4.0-2ubuntu1~20.04 \
     g++-11=11.4.0-2ubuntu1~20.04 \
     gfortran-11=11.4.0-2ubuntu1~20.04 \
     cmake-data=3.26.4-0kitware1ubuntu20.04.1 \
     cmake=3.26.4-0kitware1ubuntu20.04.1 \
     pkg-config \
     libncurses5-dev \
     m4 \
     perl \
  && rm -rf /var/lib/apt/lists/*
RUN pip install clingo

# Now we install spack and find compilers/externals
RUN mkdir -p /opt/ && cd /opt/ && git clone --depth 1 --branch "v0.20.1" https://github.com/spack/spack.git

# Add current source dir into the image
COPY . /opt/src/ci-images

# Get the latest version of the darma-vt repo
RUN cd /opt/src/ci-images/spack-repos && git clone https://github.com/DARMA-tasking/spack-package.git vt

# Apply our patch to get more up-to-date packages
RUN cd /opt/spack && git apply /opt/src/ci-images/arborx_spack_package.patch

# Add our new repos
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack repo add /opt/src/ci-images/spack-repos/p3a \
  && spack repo add /opt/src/ci-images/spack-repos/vt

# Setup our environment
RUN mkdir -p /opt/spack-env && mv /opt/src/ci-images/spack.yaml /opt/spack-env
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack --env-dir /opt/spack-env concretize
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack --env-dir /opt/spack-env install --fail-fast \
  && spack --env-dir /opt/spack-env gc -y
