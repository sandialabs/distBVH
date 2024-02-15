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
if (NOT BVH_DISABLE_TESTS)
  add_executable(BVHTests
      TestMain.cpp
      # KDopTest.cpp
      # TreeTest.cpp
      # CollisionTest.cpp
      # IteratorTest.cpp
      #BroadphaseTest.cpp
      #NarrowphaseTest.cpp
      # SerializerTest.cpp
      #PatchTest.cpp
      #PrimitiveTest.cpp
      # bits_test.cpp
      # hash_test.cpp
      # sort_test.cpp
      # SplitTest.cpp
      # UtilTest.cpp
      # math_vec_tests.cpp
      collision_object_test.cpp
      # cluster_test.cpp
      # snapshot_test.cpp
    )

  if (Kokkos_FOUND)
    #target_sources(BVHTests PRIVATE KokkosTest.cpp)
  endif()

  target_compile_definitions(BVHTests PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)
  target_compile_options(BVHTests PRIVATE -Werror)

  target_include_directories(BVHTests PRIVATE ${CMAKE_CURRENT_LIST_DIR})
  target_link_libraries(BVHTests PRIVATE bvh::bvh)

  include(CTest)
  include(Catch)

  # Additionally manually added tests since catch2 doesn't really support custom executables so we can't use MPI
  add_test(
    NAME "mpi_collision_object_multiple_broadphase_np_2"
    COMMAND mpirun -np 2 $<TARGET_FILE:BVHTests> "collision_object multiple broadphase")
  add_test(
    NAME "mpi_collision_object_multiple_broadphase_np_4"
    COMMAND mpirun -np 4 $<TARGET_FILE:BVHTests> "collision_object multiple broadphase")
  add_test(
    NAME "mpi_collision_object_narrowphase_multi_iteration_np_2"
    COMMAND mpirun -np 2 $<TARGET_FILE:BVHTests> "collision_object narrowphase multi-iteration")
  add_test(
    NAME "mpi_collision_object_narrowphase_multi_iteration_np_4"
    COMMAND mpirun -np 4 $<TARGET_FILE:BVHTests> "collision_object narrowphase multi-iteration")
  add_test(
    NAME "mpi_collision_object_narrowphase_no_overlap_multi_iteration_np_2"
    COMMAND mpirun -np 2 $<TARGET_FILE:BVHTests> "collision_object narrowphase no overlap multi-iteration")
  add_test(
    NAME "mpi_collision_object_narrowphase_no_overlap_multi_iteration_np_4"
    COMMAND mpirun -np 4 $<TARGET_FILE:BVHTests> "collision_object narrowphase no overlap multi-iteration")
  catch_discover_tests(BVHTests TEST_SPEC  EXTRA_ARGS --vt_quiet)

  message(STATUS "Building tests")
endif()
