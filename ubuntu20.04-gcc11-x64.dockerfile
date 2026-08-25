FROM ubuntu:20.04 AS build_dependencies-stage

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
      gcc-11 \
      g++-11 \
      gfortran-11 \
      cmake-data=3.26.4-0kitware1ubuntu20.04.1 \
      cmake=3.26.4-0kitware1ubuntu20.04.1 \
      pkg-config \
      libncurses5-dev \
      m4 \
      perl \
  && rm -rf /var/lib/apt/lists/*
RUN pip install clingo

# Now we install spack and find compilers/externals
RUN mkdir -p /opt/ && cd /opt/ \
  && git clone --depth 1 --branch "v1.2.2" https://github.com/spack/spack.git

# Pin the builtin package repo (spack/spack-packages) to the commit that merged
# darma-vt and darma-magistrate
RUN mkdir -p /root/.spack
COPY repos.yaml /root/.spack/repos.yaml

# Add current source dir into the image
COPY . /opt/src/ci-images

# Add our remaining repos
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack repo add /opt/src/ci-images/spack-repos/p3a

# Find compilers and system externals.
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack compiler find \
  && spack external find

# Setup our environment
RUN mkdir -p /opt/spack-env && mv /opt/src/ci-images/spack.yaml /opt/spack-env
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack --env-dir /opt/spack-env concretize
RUN . /opt/spack/share/spack/setup-env.sh \
  && spack --env-dir /opt/spack-env install --fail-fast \
  && spack --env-dir /opt/spack-env gc -y
