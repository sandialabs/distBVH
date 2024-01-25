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
#include "TestCommon.hpp"
#include <bvh/narrowphase.hpp>

#if 0
TEST_CASE("two objects collide with narrowphase", "[narrowphase]")
{
  auto obj1 = buildElementGrid( 1, 1, 1 );
  auto obj2 = buildElementGrid( 2, 3, 2 );

  bvh::collision_query_result< std::size_t > res;

  bvh::narrowphase( obj1, obj2, [&res]( const auto &_e1, const auto &_e2 ){
    res.pairs.emplace_back( _e1.global_id(), _e2.global_id() );
  } );

  // obj1 should collide with all of obj2 (12 elements)
  REQUIRE( res.size() == 12 );
}
#endif

#include <bvh/narrowphase/kokkos.hpp>

TEST_CASE("two objects collide with kokkos narrowphase", "[narrowphase]")
{
  auto obj1 = buildElementGrid( 1, 1, 1 );
  auto obj2 = buildElementGrid( 2, 3, 2 );

  using patch_type = bvh::patch<>;

  patch_type a( 0, bvh::span< Element >( obj1.data(), obj1.size() ) );
  patch_type b( 1, bvh::span< Element >( obj2.data(), obj2.size() ) );

  auto res = bvh::kokkos::narrowphase( a, b );
  // obj1 should collide with all of obj2 (12 elements)
  REQUIRE( res.size() == 12 );
}
