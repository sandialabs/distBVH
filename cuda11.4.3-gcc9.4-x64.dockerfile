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

RUN git clone --branch develop --depth 1 https://github.com/kokkos/kokkos.git \
    && git clone --branch develop --depth 1 https://github.com/DARMA-tasking/vt.git \
    && git clone --branch develop --depth 1 https://github.com/DARMA-tasking/magistrate.git vt/lib/checkpoint \
    && git config --global --add safe.directory /kokkos \
    && git config --global --add safe.directory /vt

RUN cmake -B /vt/builddir -S /vt \
        -DCMAKE_CXX_COMPILER=/kokkos/bin/nvcc_wrapper \
        -Dvt_build_examples=OFF \
        -Dvt_build_tests=OFF \
        -Dvt_build_tools=OFF \
        -Dvt_trace_enabled=ON \
    && cmake --build /vt/builddir --target install -j 2 \
    && rm -rf /vt/builddir

RUN cmake -B /kokkos/builddir -S /kokkos \
        -DCMAKE_CXX_COMPILER=/kokkos/bin/nvcc_wrapper \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ARCH_PASCAL61=ON \
    && cmake --build /kokkos/builddir --target install -j 2 \
    && rm -rf /kokkos/builddir
