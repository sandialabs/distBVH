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

#include <bvh/hash.hpp>
#include "TestCommon.hpp"
#include <array>
#include <random>
#include <chrono>

TEST_CASE("morton hashing", "[hash]")
{
  // Note we can only encode the lower 10 bits in each component
  // since the final hash is only 32 bits. The upper 22 bits are lost
  SECTION("32 bit")
  {
    SECTION("all bits set")
    {
      std::uint32_t x = 0x000003ff;
      std::uint32_t y = 0x000003ff;
      std::uint32_t z = 0x000003ff;

      auto hash = bvh::morton( x, y, z );

      // Top 2 bits never set
      REQUIRE( hash == 0x3fffffff );
    }

    SECTION("only x")
    {
      std::uint32_t x = 0x000003ff;
      std::uint32_t y = 0x00000000;
      std::uint32_t z = 0x00000000;

      auto hash = bvh::morton( x, y, z );

      // Top 2 bits never set
      // Pattern is zyxzyxzyx so here we expect (m is mask)
      // mm00 1001 0010 0100 1001 0010 0100 1001
      REQUIRE( hash == 0x09249249 );
    }

    SECTION("only y")
    {
      std::uint32_t x = 0x00000000;
      std::uint32_t y = 0x000003ff;
      std::uint32_t z = 0x00000000;

      auto hash = bvh::morton( x, y, z );

      // Top 2 bits never set
      // Pattern is zyxzyxzyx so here we expect (m is mask)
      // mm01 0010 0100 1001 0010 0100 1001 0010
      REQUIRE( hash == 0x12492492 );
    }

    SECTION("only z")
    {
      std::uint32_t x = 0x00000000;
      std::uint32_t y = 0x00000000;
      std::uint32_t z = 0x000003ff;

      auto hash = bvh::morton( x, y, z );

      // Top 2 bits never set
      // Pattern is zyxzyxzyx so here we expect (m is mask)
      // mm10 0100 1001 0010 0100 1001 0010 0100
      REQUIRE( hash == 0x24924924 );
    }
  }

  // Note we can only encode the lower 21 bits in each component
  // since the final hash is only 64 bits. The upper 43 bits are lost
  SECTION("64 bit")
  {
    SECTION("all bits set")
    {
      std::uint64_t x = 0x1fffff;
      std::uint64_t y = 0x1fffff;
      std::uint64_t z = 0x1fffff;

      auto hash = bvh::morton( x, y, z );

      // Top bit never set
      REQUIRE( hash == 0x7fffffffffffffff );
    }

    SECTION("only x")
    {
      std::uint64_t x = 0x1fffff;
      std::uint64_t y = 0x0;
      std::uint64_t z = 0x0;

      auto hash = bvh::morton( x, y, z );

      // Top bit never set
      // Pattern is zyxzyxzyx so here we expect
      // 0010 0100 1001 (0x249) repeating
      REQUIRE( hash == 0x1249249249249249 );
    }

    SECTION("only y")
    {
      std::uint64_t x = 0x0;
      std::uint64_t y = 0x1fffff;
      std::uint64_t z = 0x0;

      auto hash = bvh::morton( x, y, z );

      // Top bit never set
      // Pattern is zyxzyxzyx so here we expect
      // 0100 1001 0010 (0x492) repeating
      REQUIRE( hash == 0x2492492492492492 );
    }

    SECTION("only z")
    {
      std::uint64_t x = 0x0;
      std::uint64_t y = 0x0;
      std::uint64_t z = 0x1fffff;

      auto hash = bvh::morton( x, y, z );

      // Top bit never set
      // Pattern is zyxzyxzyx so here we expect
      // 1001 0010 0100 (0x924) repeating
      REQUIRE( hash == 0x4924924924924924 );
    }
  }
}

TEST_CASE("quantization", "[hash]")
{
  bvh::m::vec3d min{ -1.0, 0.0, 3.0 };
  bvh::m::vec3d max{ -0.5, 2.0, 4.0 };

  SECTION("32 bit")
  {
    using vec_type = bvh::m::vec3< std::uint32_t >;
    SECTION("bounds")
    {
      auto almost_max = max - bvh::m::vec3d::set1( 0.00001 );
      REQUIRE( bvh::quantize32( min, min, max ) == vec_type{ 0, 0, 0 } );
      REQUIRE( bvh::quantize32( almost_max, min, max ) == vec_type{ 0x3ff, 0x3ff, 0x3ff } );
    }
  }
}

TEST_CASE("benchmark morton", "[hash][!benchmark]")
{
#ifdef __BMI2__
  REQUIRE( bvh::detail::morton64( 5, 7, 9 ) == bvh::detail::morton64_intrin( 5, 7, 9 ) );
#endif

  std::default_random_engine e;
  std::uniform_int_distribution< std::uint64_t > dist( 0x0u, 0x1fffff );

  std::array< std::uint64_t, 3000 > vals;
  for ( int i = 0; i < 3000; ++i )
    vals[i] = dist( e );


#ifdef __BMI2__
  for ( int i = 0; i < 3000; i += 3 )
  {
    REQUIRE( bvh::detail::morton64( vals[i], vals[i + 1], vals[i + 2] )
             == bvh::detail::morton64_intrin( vals[i], vals[i + 1], vals[i + 2] ) );
  }
#endif

  BENCHMARK("morton64")
  {
    volatile std::uint64_t ret;

    for ( int i = 0; i < 3000; i += 3 )
    {
      ret = bvh::detail::morton64( vals[i], vals[i + 1], vals[i + 2] );
    }
  };

#ifdef __BMI2__
  BENCHMARK("morton64 intrin")
  {
    volatile std::uint64_t ret;

    for ( int i = 0; i < 3000; i += 3 )
    {
      ret = bvh::detail::morton64_intrin( vals[i], vals[i + 1], vals[i + 2] );
    }
  };
#endif
}
