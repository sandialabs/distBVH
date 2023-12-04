# distBVH

distBVH is a library for asynchronous parallel distributed collision detection. It provides data
structures and parallel algorithms for use in solid mechanics applications that
require proximity search.

## Building

Building is done via CMake (version 3.15 required). Detailed instructions can be found [here](http://bvh.gitlab.lan/building.html).

### Dependencies

BVH requires a compiler that supports C++17. This includes at least clang 5 or gcc version 7.0.

BVH uses [DARMA/vt](https://github.com/DARMA-tasking/vt) for asynchronous tasking. For more information about DARMA/vt
please consult the [documentation](https://darma-tasking.github.io/docs/html/index.html).
BVH requires [Kokkos](https://github.com/kokkos/kokkos) version `4.0` or later.

VTK can also be used for visualizing the the tree data structure.

## Build example

```{.bash}
cd bvh
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -Dvt_DIR=/path/to/vt/build-debug/install/cmake \
      -DVTK_DIR=/path/to/vtk/8.2.0/lib/cmake/vtk-8.2 \
      -DBVH_DEBUG_LEVEL=5 \
      ..
make -j8
```

In order to build the doxygen documentation:

```{.bash}
cmake -DCMAKE_PROGRAM_PATH=/path/to/doxygen/bin         \
      -DSphinx_ROOT=/path/to/python/install/or/venv     \
      -DCMAKE_BUILD_TYPE=Debug \
      -Dvt_DIR=/path/to/vt/build-debug/install/cmake \
      -DVTK_DIR=/path/to/vtk/8.2.0/lib/cmake/vtk-8.2 \
      -DBVH_DEBUG_LEVEL=5 \
      ..
make doc
```

Additionally all tests can be run:

```{.bash}
make test
```

## Including bvh in applications

BVH is built using cmake, which provides a straightforward way of including it as a dependency.
Either the build tree or install tree can be used when linking to BVH. This makes linking to BVH
simpler when modifying or patching BVH as it does not have to be installed after every change.
However, BVH does not currently support sub-directory builds (i.e. copying the project into
a subdirectory and using cmake add_subdirectory).

### Dependency via CMake

Set `bvh_DIR` to point to the directory containing `bvhConfig.cmake` or point `bvh_ROOT`
(in more recent versions of CMake) to point to the install directory or build tree. These
can also be environment variables.

```{.bash}
cmake -Dbvh_ROOT="<path_to_build_or_install>"
```

Then inside your `CMakeLists.txt` simply call:

```{.cmake}
find_package(bvh)
```

This will import the `bvh::bvh` target. To create a dependency on this target (this will automatically set linker flags and include paths), call:

```{.cmake}
target_link_libraries(<target> bvh::bvh)
```

where `<target>` is your target that is dependent on bvh.

### Manual depedency management

The simplest way of dealing with a manual dependency is to install bvh to a standard install directory via `make install`.
