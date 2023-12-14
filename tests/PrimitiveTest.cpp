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

#include <bvh/primitive/kokkos_closest_point.hpp>
#include "TestCommon.hpp"
#include <array>

inline void gen_array( bvh::kokkos::host_view< double *[3] > _points )
{
  Kokkos::parallel_for( Kokkos::RangePolicy< bvh::kokkos::host_execution_space >( 0, 4 ), [_points]( int i ){
    static constexpr std::array< bvh::m::constant_vec3d, 4 > varr = {{
      { 0.0, 2.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 }
    }};

    _points( i, 0 ) = varr[i][0];
    _points( i, 1 ) = varr[i][1];
    _points( i, 2 ) = varr[i][2];
  } );
}

inline void check_array( bvh::kokkos::host_view< double *[3] >  _closest )
{
  Kokkos::parallel_for( Kokkos::RangePolicy< bvh::kokkos::host_execution_space >( 0, 4 ), [_closest]( int i ){
    static constexpr std::array< bvh::m::constant_vec3d, 4 > expected = {{
      { 0.0, 1.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }
    }};

    REQUIRE( _closest( i, 0 ) == expected[i][0] );
    REQUIRE( _closest( i, 1 ) == expected[i][1] );
    REQUIRE( _closest( i, 2 ) == expected[i][2] );
  } );
}

TEST_CASE("the closest point algorithm determines the closest points using kokkos", "[primitive][kokkos]")
{
  bvh::triangle< double > t{ {  0.0, 1.0, 0.0 },
                             { -1.0, 0.0, 0.0 },
                             {  1.0, 0.0, 0.0 } };

  bvh::kokkos::host_view< double *[3] > points{ "Points", 4 };

  gen_array( points );

  auto dev_points = bvh::kokkos::transfer_to_device( points );

  bvh::kokkos::view< double *[3] > closest( "Closest", 4 );

  bvh::kokkos::closest_point( dev_points, t, closest );

  auto host_closest = bvh::kokkos::transfer_to_host( closest );

  check_array( host_closest );
}
