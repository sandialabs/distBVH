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
#include <bvh/narrowphase/kokkos.hpp>
#include "TestCommon.hpp"

TEST_CASE("max variant axis works", "[kokkos][search]")
{
  auto rank = ::vt::theContext()->getNode();

  SECTION( "x-axis" )
  {
    auto vec2 = buildElementGrid(1, 1, 1, rank);
    auto vec = buildElementGrid(12, 1, 1, rank * 12);
    auto xview = Kokkos::View<Element *, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec.data(), vec.size());
    auto compview = Kokkos::View<Element *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec2.data(), vec2.size());

    int axis = bvh::max_variant_axis(xview, compview);
    REQUIRE(axis == 0);
  }

  SECTION( "y-axis" )
  {
    auto vec2 = buildElementGrid(1, 1, 1, rank);
    auto vec = buildElementGrid(1, 12, 1, rank * 12);
    auto xview = Kokkos::View<Element *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec.data(), vec.size());
    auto compview = Kokkos::View<Element *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec2.data(), vec2.size());

    int axis = bvh::max_variant_axis(xview, compview);
    REQUIRE(axis == 1);
  }

  SECTION( "y-axis" )
  {
    auto vec2 = buildElementGrid(1, 1, 1, rank);
    auto vec = buildElementGrid(1, 1, 12, rank * 12);
    auto xview = Kokkos::View<Element *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec.data(), vec.size());
    auto compview = Kokkos::View<Element *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec2.data(), vec2.size());

    int axis = bvh::max_variant_axis(xview, compview);
    REQUIRE(axis == 2);
  }
}

TEST_CASE("sort_and_sweep_local", "[kokkos][search]")
{
  auto rank = ::vt::theContext()->getNode();

  SECTION("x-axis")
  {
    auto vec2 = buildElementGrid(1, 1, 1, rank);
    auto vec = buildElementGrid(12, 1, 1, rank * 12);
    auto xview = Kokkos::View<Element *, Kokkos::HostSpace,
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec.data(), vec.size());
    auto compview = Kokkos::View<Element *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec2.data(), vec2.size());

    bvh::patch<> p1( 0, bvh::span< const Element >( vec.data(), vec.size() ) );
    bvh::patch<> p2( 1, bvh::span< const Element >( vec2.data(), vec2.size() ) );

    std::atomic< int > count{ 0 };
    bvh::sort_and_sweep_local( p1, xview, p2, compview, 0,
                                      [rank, &count]( const Element &_a, const Element &_b ) {
                                        REQUIRE( _b.global_id() == rank );
                                        REQUIRE( _a.global_id() >= rank * 12 );
                                        REQUIRE( _a.global_id() < ( rank + 1 ) * 12 );
                                        ++count;
                                      } );
    REQUIRE( count.load() == 12 );
  }

  SECTION("x-axis flipped")
  {
    auto vec2 = buildElementGrid(1, 1, 1, rank);
    auto vec = buildElementGrid(12, 1, 1, rank * 12);
    auto xview = Kokkos::View<Element *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec.data(), vec.size());
    auto compview = Kokkos::View<Element *, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        vec2.data(), vec2.size());

    bvh::patch<> p1( 0, bvh::span< const Element >( vec.data(), vec.size() ) );
    bvh::patch<> p2( 1, bvh::span< const Element >( vec2.data(), vec2.size() ) );

    std::atomic< int > count{ 0 };
    bvh::sort_and_sweep_local( p2, compview, p1, xview, 0,
                                       [rank, &count]( const Element &_a, const Element &_b ) {
                                         REQUIRE( _a.global_id() == rank );
                                         REQUIRE( _b.global_id() >= rank * 12 );
                                         REQUIRE( _b.global_id() < ( rank + 1 ) * 12 );
                                         ++count;
                                       } );
    REQUIRE( count.load() == 12 );
  }
}
