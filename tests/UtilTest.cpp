/*
 * distBVH 1.0
 *
 * Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <catch2/catch.hpp>

#include <bvh/util/prefix_sum.hpp>
#include <bvh/util/sort.hpp>
#include <bvh/util/kokkos.hpp>
#include "TestCommon.hpp"
#include <array>
#include <utility>

#ifdef BVH_ENABLE_KOKKOS

TEST_CASE("kokkos prefix sum yields the correct results", "[utility][kokkos]")
{
  bvh::host_view< int * > nums{ "Numbers", 8 };
  
  gen_array( nums, 3, 7, 2, 1, 4, 9, 1, 3 );
  
  auto device_nums =  Kokkos::create_mirror_view_and_copy( bvh::primary_execution_space{}, nums );
  
  bvh::kokkos::prefix_sum( device_nums );
  
  nums = Kokkos::create_mirror_view_and_copy( bvh::host_execution_space{}, device_nums );
  
  test_array( nums, 0, 3, 10, 12, 13, 17, 26, 27 );
}

TEST_CASE("kokkos radix sort yields the sorted results", "[utility][kokkos]")
{
  bvh::host_view< uint32_t * > nums{ "Numbers", 8 };
  bvh::host_view< int * > indices{ "Indices", 8 };
  
  gen_array( nums, 3, 7, 2, 1, 4, 9, 1, 3 );
  gen_array( indices, 0, 1, 2, 3, 4, 5, 6, 7 );
  
  auto dev_nums = Kokkos::create_mirror_view_and_copy( bvh::primary_execution_space{}, nums );
  auto dev_indices = Kokkos::create_mirror_view_and_copy( bvh::primary_execution_space{}, indices );

  bvh::kokkos::radix_sort( dev_nums, dev_indices );

  nums = Kokkos::create_mirror_view_and_copy( bvh::host_execution_space{}, dev_nums );
  indices = Kokkos::create_mirror_view_and_copy( bvh::host_execution_space{}, dev_indices );
  
  test_array( nums, 1, 1, 2, 3, 3, 4, 7, 9 );
  test_array( indices, 3, 6, 2, 0, 7, 4, 1, 5 );
}

#endif  // BVH_ENABLE_KOKKOS
