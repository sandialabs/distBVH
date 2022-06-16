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

#include <vector>
#include <bvh/kdop.hpp>
#include <random>
#include <limits>
#include <bvh/memory.hpp>
#include <bvh/util/container.hpp>


using kd_type = bvh::dop_26< double >;
using namespace Catch::literals;

static_assert( bvh::is_kdop_type< bvh::dop_6< double > >::value, "" );
static_assert( bvh::is_kdop_type< bvh::dop_18< double > >::value, "" );
static_assert( bvh::is_kdop_type< bvh::dop_26< double > >::value, "" );
static_assert( bvh::is_kdop_type< bvh::dop_6< float > >::value, "" );
static_assert( bvh::is_kdop_type< bvh::dop_18< float > >::value, "" );
static_assert( bvh::is_kdop_type< bvh::dop_26< float > >::value, "" );

TEST_CASE("26-dops are default constructable", "[kdop]")
{
  kd_type kd;
  // Degenerate dop
  REQUIRE( 0.0 > kd.extents[kd.longest_axis()].length() );
}


TEST_CASE("26-dops longest axis yields the correct result", "[kdop]")
{
  kd_type kd;
  kd.extents[0].min = 0.0;
  kd.extents[0].max = 5.0;
  REQUIRE( 0 == kd.longest_axis() );

  kd = kd_type();
  kd.extents[0].min = -1.0;
  kd.extents[0].max = 0.0;
  REQUIRE( 0 == kd.longest_axis() );
}

TEST_CASE("26-dop project function yields the correct projection on the x axis", "[kdop]")
{
  REQUIRE( 0.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 0.0 }, 0 ) );
  REQUIRE( 1.0_a == kd_type::project( bvh::m::vec3d{ 1.0, 0.0, 0.0 }, 0 ) );
  REQUIRE( -1.0_a == kd_type::project( bvh::m::vec3d{ -1.0, 0.0, 0.0 }, 0 ) );
  REQUIRE( 5.0_a == kd_type::project( bvh::m::vec3d{ 5.0, 0.0, 0.0 }, 0 ) );
  REQUIRE( 5.0_a == kd_type::project( bvh::m::vec3d{ 5.0, 0.9, 11.0 }, 0 ) );
}

TEST_CASE("26-dop project function yields the correct projection on the y axis", "[kdop]")
{
  REQUIRE( 0.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 0.0 }, 1 ) );
  REQUIRE( 1.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 1.0, 0.0 }, 1 ) );
  REQUIRE( -1.0_a == kd_type::project( bvh::m::vec3d{ 0.0, -1.0, 0.0 }, 1 ) );
  REQUIRE( 5.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 5.0, 0.0 }, 1 ) );
  REQUIRE( 5.0_a == kd_type::project( bvh::m::vec3d{ 0.2, 5.0, 11.0 }, 1 ) );
}

TEST_CASE("26-dop project function yields the correct projection on the z axis", "[kdop]")
{
  REQUIRE( 0.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 0.0 }, 2 ) );
  REQUIRE( 1.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 1.0 }, 2 ) );
  REQUIRE( -1.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, -1.0 }, 2 ) );
  REQUIRE( 5.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 5.0 }, 2 ) );
  REQUIRE( 5.0_a == kd_type::project( bvh::m::vec3d{ 0.2, 13.0, 5.0 }, 2 ) );
}


TEST_CASE("26-dop project function yields the correct projection on the xyz corner", "[kdop]")
{
  REQUIRE( 0.0_a == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 0.0 }, 3 ) );
  REQUIRE( Approx( 3.0 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ 1.0, 1.0, 1.0 }, 3 ) );
  REQUIRE( Approx( -1.0 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ -1.0, 0.0, 0.0 }, 3 ) );
  REQUIRE( Approx( 5.0 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ 5.0, 0.0, 0.0 }, 3 ) );
  REQUIRE( Approx( 16.9 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ 5.0, 0.9, 11.0 }, 3 ) );
}


TEST_CASE("26-dop project function yields the correct projection on the x-yz corner", "[kdop]")
{
  REQUIRE( Approx( 0.0 ) == kd_type::project( bvh::m::vec3d{ 0.0, 0.0, 0.0 }, 17 ) );
  REQUIRE( Approx( -1.0 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ 1.0, 1.0, 1.0 }, 17 ) );
  REQUIRE( Approx( 1.0 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ -1.0, 0.0, 0.0 }, 17 ) );
  REQUIRE( Approx( -5.0 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ 5.0, 0.0, 0.0 }, 17 ) );
  REQUIRE( Approx( -15.1 / std::sqrt( 3.0 ) ) == kd_type::project( bvh::m::vec3d{ 5.0, 0.9, 11.0 }, 17 ) );
}

TEST_CASE("a size 0 26-dop can be constructed from a point", "[kdop]")
{
  std::vector< bvh::m::vec3d, bvh::aligned_allocator< bvh::m::vec3d > > verts = { { 1.0, 1.0, 1.0 } };
  kd_type kd = kd_type::from_vertices( verts.begin(), verts.end() );

  REQUIRE( 0.0_a == kd.extents[kd.longest_axis()].length() );
}

TEST_CASE("26-dop can be constructed from vertices in a line", "[kdop]")
{
  std::vector< bvh::m::vec3d, bvh::aligned_allocator< bvh::m::vec3d > > verts = { { 1.0, 1.0, 1.0 }, { 1.0, 5.0, 1.0} };
  kd_type kd = kd_type::from_vertices( verts.begin(), verts.end() );

  int longest = kd.longest_axis();
  REQUIRE( 1 == longest );
  REQUIRE( 4.0_a == kd.extents[longest].length() );

  verts = { { 1.0, 1.0, 1.0 }, { 4.0, 5.0, 4.0} };
  kd = kd_type::from_vertices( verts.begin(), verts.end() );

  longest = kd.longest_axis();
  REQUIRE( 3 == longest );
  REQUIRE( Approx( 10.0 / std::sqrt( 3.0 ) ) == kd.extents[longest].length() );
}

TEST_CASE("26-dop can be constructed from a diagonal line", "[kdop]")
{
  bvh::dynarray< bvh::m::vec3d > verts = { { -5.0, -1.0, 1.0 }, { 1.0, 5.0, 1.0} };
  kd_type kd = kd_type::from_vertices( verts.begin(), verts.end() );

  int longest = kd.longest_axis();
  REQUIRE( 7 == longest );
  REQUIRE( Approx( 12.0 / std::sqrt( 2.0 ) ) == kd.extents[longest].length() );
}

TEST_CASE("26-dop can be constructed from a line with the correct lengths", "[kdop]")
{
  bvh::dynarray< bvh::m::vec3d > verts = { { 0.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0} };
  kd_type kd = kd_type::from_vertices( verts.begin(), verts.end() );

  int longest = kd.longest_axis();
  REQUIRE( 0 == longest );
  REQUIRE( 1.0_a == kd.extents[longest].length() );
  REQUIRE( 0.0_a == kd.extents[1].length() );
  REQUIRE( 0.0_a == kd.extents[2].length() );
}

namespace
{
  std::array< bvh::m::vec3d, 8 > makeCube( const bvh::m::vec3d &_center )
  {
    return {{ _center + bvh::m::vec3d{ 1., 1., 1. },
              _center + bvh::m::vec3d{ -1., 1., 1. },
              _center + bvh::m::vec3d{ 1., -1., 1. },
              _center + bvh::m::vec3d{ 1., 1., -1. },
              _center + bvh::m::vec3d{ -1., -1., 1. },
              _center + bvh::m::vec3d{ -1., 1., -1. },
              _center + bvh::m::vec3d{ 1., -1., -1. },
              _center + bvh::m::vec3d{ -1., -1., -1. },
            }};
  }
}

TEST_CASE("two 26-dops from cubes collide correctly", "[kdop]")
{
  auto cube1 = makeCube( bvh::m::vec3d( 0.0, 0.0, 0.0 ) );
  auto cube2 = makeCube( bvh::m::vec3d( 3.0, 0.0, 0.0 ) );

  auto kd1 = kd_type::from_vertices( cube1.begin(), cube1.end() );
  auto kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE_FALSE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( 2.0 - std::numeric_limits< double >::epsilon(), 0.0, 0.0 ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( 0.0, 0.0, 0.0 ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( -2.0 + std::numeric_limits< double >::epsilon(), 0.0, 0.0 ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( -3.0, 0.0, 0.0 ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE_FALSE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( 2.0 - std::numeric_limits< double >::epsilon(), 2.0 - std::numeric_limits< double >::epsilon(), 0.0 ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( -2.0 + std::numeric_limits< double >::epsilon(), -2.0 + std::numeric_limits< double >::epsilon(), 0.0 ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( 2.0 - std::numeric_limits< double >::epsilon(), 2.0 - std::numeric_limits< double >::epsilon(), 2.0 - std::numeric_limits< double >::epsilon() ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );

  cube2 = makeCube( bvh::m::vec3d( -2.0 + std::numeric_limits< double >::epsilon(), -2.0 + std::numeric_limits< double >::epsilon(), -2.0 + std::numeric_limits< double >::epsilon() ) );
  kd2 = kd_type::from_vertices( cube2.begin(), cube2.end() );

  REQUIRE( overlap( kd1, kd2 ) );
}

TEST_CASE("two 26-dops that are identical overlap", "[kdop]")
{
  auto cube1 = makeCube( bvh::m::vec3d( 0.0, 0.0, 0.0 ) );

  auto kd1 = kd_type::from_vertices( cube1.begin(), cube1.end() );
  auto kd2 = kd_type::from_vertices( cube1.begin(), cube1.end() );

  REQUIRE( overlap( kd1, kd2 ) );
}

TEST_CASE("two 26-dops from spheres collide correctly", "[kdop]")
{
  auto kd1 = kd_type::from_sphere( bvh::m::vec3d( 0.0, 0.0, 0.0 ), 1.0 );
  auto kd2 = kd_type::from_sphere( bvh::m::vec3d( 3.0, 0.0, 0.0 ), 1.0 );

  REQUIRE_FALSE( overlap( kd1, kd2 ) );

  kd2 = kd_type::from_sphere( bvh::m::vec3d( 2.0 - std::numeric_limits< double >::epsilon(), 0.0, 0.0 ), 1.0 );

  REQUIRE( overlap( kd1, kd2 ) );
}

TEST_CASE("from 100 random 26-dops from spheres, for any colliding sphere, the 26-dops also collide", "[kdop]")
{
  static std::mt19937 gen( 0 );
  static std::uniform_real_distribution<> locgen( 0.0, 4.0 );

  for ( int i = 0; i < 100; ++i )
  {
    auto c1 = bvh::m::vec3d( locgen( gen ), locgen( gen ), locgen( gen ) );
    auto c2 = bvh::m::vec3d( locgen( gen ), locgen( gen ), locgen( gen ) );

    auto kd1 = kd_type::from_sphere( c1, 1.0 );
    auto kd2 = kd_type::from_sphere( c2, 1.0 );

    // If the spheres are close enough to collide
    if ( distance2( c1, c2 ) < 2.0 )
    {
      // The dop should also collide
      REQUIRE( overlap( kd1, kd2 ) );
    }
  }
}

TEST_CASE("from 100 random 26-dops from spheres with random radii, for any colliding sphere, the 26-dops also collide", "[kdop]")
{
  static std::mt19937 gen( 0 );
  static std::uniform_real_distribution<> locgen( 0.0, 4.0 );
  static std::uniform_real_distribution<> radiusgen( 0.5, 1.5 );

  for ( int i = 0; i < 100; ++i )
  {
    auto c1 = bvh::m::vec3d( locgen( gen ), locgen( gen ), locgen( gen ) );
    auto c2 = bvh::m::vec3d( locgen( gen ), locgen( gen ), locgen( gen ) );

    auto r1 = radiusgen( gen );
    auto r2 = radiusgen( gen );

    auto kd1 = kd_type::from_sphere( c1, r1 );
    auto kd2 = kd_type::from_sphere( c2, r2 );

    // If the spheres are close enough to collide
    if ( distance2( c1, c2 ) <= ( r1 + r2 ) * ( r1 + r2 ) )
    {
      // The dop should also collide
      REQUIRE( overlap( kd1, kd2 ) );
    }
  }
}


