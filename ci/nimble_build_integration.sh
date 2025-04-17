#!/bin/bash

set -e
mkdir -p /opt/src

pushd /opt/src
git clone -b 371-fix-distbvh-integration https://github.com/NimbleSM/NimbleSM.git
mkdir -p /opt/builds/NimbleSM

pushd /opt/builds/NimbleSM
cmake -DCMAKE_C_COMPILER="gcc-11" \
  -DCMAKE_CXX_COMPILER="g++-11" \
  -DCMAKE_PREFIX_PATH="/opt/view" \
  -DNimbleSM_ENABLE_KOKKOS="ON" \
  -DNimbleSM_ENABLE_BVH="ON" \
  -DNimbleSM_ENABLE_MPI="ON" \
  -Dbvh_ROOT="/opt/builds/bvh" \
  /opt/src/NimbleSM
cmake --build . -j $(nproc)

popd  # /opt/builds/NimbleSM
popd  # /opt/src
