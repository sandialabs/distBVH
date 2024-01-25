#[[
distBVH 1.0

Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
]]
cmake_minimum_required(VERSION 3.15)

file(READ VERSION _ver)

string(STRIP "${_ver}" _ver)

project(bvh VERSION ${_ver}
    DESCRIPTION "A library for asynchronous parallel distributed collision detection."
    HOMEPAGE_URL "http://bvh.gitlab.lan/")

message(STATUS "BVH Version ${PROJECT_VERSION}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# static analysis
find_program( CLANG_TIDY_EXE
  NAMES "clang-tidy" )

add_library(bvh)
add_library(bvh::bvh ALIAS bvh)

target_include_directories(bvh INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
        )

# Various default compiler options

target_compile_options(bvh PUBLIC $<$<CONFIG:DEBUG>:-O0 -ggdb3 -fno-inline>)
target_compile_options(bvh PUBLIC $<$<CONFIG:RELEASE>:-O3 -ffast-math>)

option(BVH_WARNINGS_AS_ERRORS "Build with warnings as errors" OFF)
if (BVH_WARNINGS_AS_ERRORS)
  target_compile_options(bvh PRIVATE -Werror)
endif()

if (ENABLE_ASAN)
  target_compile_options(bvh PUBLIC -fsanitize=address)
  target_link_libraries(bvh PUBLIC -fsanitize=address)
endif()

if (BVH_DEBUG_LEVEL)
  message(STATUS "Compiling BVH with Debug level ${BVH_DEBUG_LEVEL}")
  target_compile_definitions(bvh PUBLIC BVH_DEBUG_LEVEL=${BVH_DEBUG_LEVEL})
endif()

# Optional dependency on vtk for visualization
find_package(VTK QUIET)
if (VTK_FOUND)
  message(STATUS "Found VTK: ${VTK_DIR}")
  target_link_libraries(bvh PUBLIC ${VTK_LIBRARIES})
  target_include_directories(bvh PUBLIC ${VTK_INCLUDE_DIRS})
  target_compile_definitions(bvh PUBLIC BVH_USE_VTK)
else()
  message(STATUS "NOT using VTK for visualization (Configure VTK_DIR to use VTK)")
endif()

find_package(vt REQUIRED)
target_link_libraries(bvh PUBLIC vt::runtime::vt)

option(BVH_VT_INSERTABLE_COLLECTIONS "Enable insertable collections (experimental VT feature)" ON)
if (BVH_VT_INSERTABLE_COLLECTIONS)
  message(STATUS "Enabling VT insertable collections (experimental feature)")
  target_compile_definitions(bvh PUBLIC BVH_VT_INSERTABLE_COLLECTIONS)
else()
  message(STATUS "Not using VT insertable collections")
endif()

find_package(Kokkos 4.0 REQUIRED)

if (Kokkos_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "distBVH requires Kokkos built with C++17 or later")
endif()
set_property(TARGET bvh PROPERTY CXX_STANDARD ${Kokkos_CXX_STANDARD})
target_link_libraries(bvh PUBLIC Kokkos::kokkos)
target_compile_definitions(bvh PUBLIC BVH_ENABLE_KOKKOS)

option(BVH_ENABLE_TRACING "Enable detailed performance tracing (may have an impact on performance" OFF)
if (BVH_ENABLE_TRACING)
  find_package(perf REQUIRED)
  target_compile_definitions(bvh PUBLIC BVH_ENABLE_TRACING)
  target_link_libraries(bvh PUBLIC perf::perf)
endif()

option(BVH_BROADPHASE_6_DOP "Enable 6-dops for broadphase")
if (BVH_BROADPHASE_6_DOP)
  target_compile_definitions(bvh PUBLIC BVH_BROADPHASE_6_DOP)
endif()

option(BVH_COPY_ALL_NARROWPHASE_PATCHES "Don't use lazy initialization" OFF)
if (BVH_COPY_ALL_NARROWPHASE_PATCHES)
  message(STATUS "Not using lazy narrowphase initialization")
  message(STATUS "Set BVH_COPY_ALL_NARROWPHASE_PATCHES to \"ON\" to use lazy initialization")
  target_compile_definitions(bvh PUBLIC BVH_COPY_ALL_NARROWPHASE_PATCHES)
else()
  message(STATUS "Using lazy narrowphase initialization")
endif()

install(TARGETS bvh EXPORT bvhTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

install(EXPORT bvhTargets
        FILE bvhTargets.cmake
        NAMESPACE bvh::
        DESTINATION cmake
    )

export(TARGETS bvh
        NAMESPACE bvh::
        FILE bvhTargets.cmake
    )

install(DIRECTORY src/bvh DESTINATION include FILES_MATCHING PATTERN "*.hpp")

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/bvhConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/bvhConfig.cmake
        INSTALL_DESTINATION cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/bvhConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
    )

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bvhConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/bvhConfigVersion.cmake
        DESTINATION cmake
    )

add_subdirectory(src)

option(BVH_DISABLE_TESTS "Disable tests" OFF)
if (NOT BVH_DISABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(examples)

add_subdirectory(docs)
