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
#include <bvh/math/vec.hpp>
#include <bvh/math/constant_vec.hpp>

TEST_CASE( "vec init", "[math][vec]" )
{
  SECTION( "zero" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 4 >::zeros();

      REQUIRE( a[0] == 0.0 );
      REQUIRE( a[1] == 0.0 );
      REQUIRE( a[2] == 0.0 );
      REQUIRE( a[3] == 0.0 );
    }

    SECTION( "constant_vec" )
    {
      auto a = bvh::m::constant_vec< double, 4 >::zeros();

      REQUIRE( a[0] == 0.0 );
      REQUIRE( a[1] == 0.0 );
      REQUIRE( a[2] == 0.0 );
      REQUIRE( a[3] == 0.0 );
    }
  }

  SECTION( "undefined" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 4 >::undefined();

      a[0] = 0.0;

      REQUIRE( a[0] == 0.0 );
    }

    SECTION( "constant_vec" )
    {
      auto a = bvh::m::constant_vec< double, 4 >::undefined();

      a[0] = 0.0;

      REQUIRE( a[0] == 0.0 );
    }
  }

  SECTION( "set1" )
  {
    SECTION( "init4" )
    {
      SECTION( "vec" )
      {
        auto a = bvh::m::vec< double, 4 >::set1( 4.0 );

        REQUIRE( a[0] == 4.0 );
        REQUIRE( a[1] == 4.0 );
        REQUIRE( a[2] == 4.0 );
        REQUIRE( a[3] == 4.0 );
      }

      SECTION( "constant_vec" )
      {
        auto a = bvh::m::constant_vec< double, 4 >::set1( 4.0 );

        REQUIRE( a[0] == 4.0 );
        REQUIRE( a[1] == 4.0 );
        REQUIRE( a[2] == 4.0 );
        REQUIRE( a[3] == 4.0 );
      }
    }

    SECTION( "init5" )
    {
      SECTION( "vec" )
      {
        auto a = bvh::m::vec< double, 5 >::set1( 4.0 );

        REQUIRE( a[0] == 4.0 );
        REQUIRE( a[1] == 4.0 );
        REQUIRE( a[2] == 4.0 );
        REQUIRE( a[3] == 4.0 );
        REQUIRE( a[4] == 4.0 );
        REQUIRE( a[5] == 0.0 );
        REQUIRE( a[6] == 0.0 );
        REQUIRE( a[7] == 0.0 );
      }

      SECTION( "constant_vec" )
      {
        auto a = bvh::m::constant_vec< double, 5 >::set1( 4.0 );

        REQUIRE( a[0] == 4.0 );
        REQUIRE( a[1] == 4.0 );
        REQUIRE( a[2] == 4.0 );
        REQUIRE( a[3] == 4.0 );
        REQUIRE( a[4] == 4.0 );
      }
    }
  }

  SECTION( "init4" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 4 >( 1.0, 2.0, 3.0, 4.0 );

      REQUIRE( a[0] == 1.0 );
      REQUIRE( a[1] == 2.0 );
      REQUIRE( a[2] == 3.0 );
      REQUIRE( a[3] == 4.0 );
    }

    SECTION( "constant_vec" )
    {
      auto a = bvh::m::constant_vec< double, 4 >( 1.0, 2.0, 3.0, 4.0 );

      REQUIRE( a[0] == 1.0 );
      REQUIRE( a[1] == 2.0 );
      REQUIRE( a[2] == 3.0 );
      REQUIRE( a[3] == 4.0 );
    }
  }

  SECTION( "init5" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 5 >( 1.0, 2.0, 3.0, 4.0, 5.0 );

      REQUIRE( a[0] == 1.0 );
      REQUIRE( a[1] == 2.0 );
      REQUIRE( a[2] == 3.0 );
      REQUIRE( a[3] == 4.0 );
      REQUIRE( a[4] == 5.0 );
      REQUIRE( a[5] == 0.0 );
      REQUIRE( a[6] == 0.0 );
      REQUIRE( a[7] == 0.0 );
    }

    SECTION( "constant_vec" )
    {
      auto a = bvh::m::constant_vec< double, 5 >( 1.0, 2.0, 3.0, 4.0, 5.0 );

      REQUIRE( a[0] == 1.0 );
      REQUIRE( a[1] == 2.0 );
      REQUIRE( a[2] == 3.0 );
      REQUIRE( a[3] == 4.0 );
      REQUIRE( a[4] == 5.0 );
    }
  }
}

TEST_CASE( "vec add", "[math][vec]" )
{
  SECTION( "1" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 1 > a{ 1.0 };
      bvh::m::vec< double, 1 > b{ 1.0 };

      auto ret = a + b;

      const auto expected = bvh::m::vec< double, 1 >{ 2.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
      bvh::m::constant_vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
      bvh::m::constant_vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

      auto ret = a + b;

      constexpr auto expected = bvh::m::constant_vec< double, 4 >{ 5.0, 5.0, 5.0, 5.0 };
      REQUIRE( ret == expected );
    }
#endif
  }

  SECTION( "2" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 2 > a{ 1.0, 2.0 };
      bvh::m::vec< double, 2 > b{ 2.0, 1.0 };

      auto ret = a + b;

      const auto expected = bvh::m::vec< double, 2 >{ 3.0, 3.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "3" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 3 > a{ 1.0, 2.0, 3.0 };
      bvh::m::vec< double, 3 > b{ 3.0, 2.0, 1.0 };

      auto ret = a + b;

      const auto expected = bvh::m::vec< double, 3 >{ 4.0, 4.0, 4.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
      bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

      auto ret = a + b;

      auto expected = bvh::m::vec< double, 4 >{ 5.0, 5.0, 5.0, 5.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 5 > a{ 1.0, 2.0, 3.0, 4.0, 5.0 };
      bvh::m::vec< double, 5 > b{ 5.0, 4.0, 3.0, 2.0, 1.0 };

      auto ret = a + b;

      const auto expected = bvh::m::vec< double, 5 >{ 6.0, 6.0, 6.0, 6.0, 6.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "6" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 6 > a{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
      bvh::m::vec< double, 6 > b{ 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

      auto ret = a + b;

      auto expected = bvh::m::vec< double, 6 >{ 7.0, 7.0, 7.0, 7.0, 7.0, 7.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "7" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 7 > a{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
      bvh::m::vec< double, 7 > b{ 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

      auto ret = a + b;

      auto expected = bvh::m::vec< double, 7 >{ 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "8" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 8 > a{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
      bvh::m::vec< double, 8 > b{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };

      auto ret = a + b;

      auto expected = bvh::m::vec< double, 8 >{ 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec sub", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
      bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

      auto ret = a - b;

      auto expected = bvh::m::vec< double, 4 >{ -3.0, -1.0, 1.0, 3.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec mul", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
      bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

      auto ret = a * b;

      auto expected = bvh::m::vec< double, 4 >{ 4.0, 6.0, 6.0, 4.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec div", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
      bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

      auto ret = a / b;

      auto expected = bvh::m::vec< double, 4 >{ 0.25, 2.0 / 3.0, 1.5, 4.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec scalar mul", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "pre" )
    {
      SECTION( "vec" )
      {
        bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
        double b = 5.0;

        auto ret = a * b;

        auto expected = bvh::m::vec< double, 4 >{ 5.0, 10.0, 15.0, 20.0 };
        REQUIRE( ret == expected );
      }

#if 0
      SECTION( "constant_vec" )
      {
      }
#endif
    }

    SECTION( "post" )
    {
      SECTION( "vec" )
      {
        double a = 2.0;
        bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

        auto ret = a * b;

        auto expected = bvh::m::vec< double, 4 >{ 8.0, 6.0, 4.0, 2.0 };
        REQUIRE( ret == expected );
      }

#if 0
      SECTION( "constant_vec" )
      {
      }
#endif
    }
  }
}


TEST_CASE( "vec scalar div", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "pre" )
    {
      SECTION( "vec" )
      {
        bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
        double b = 5.0;

        auto ret = a / b;

        auto expected = bvh::m::vec< double, 4 >{ 0.2, 0.4, 0.6, 0.8 };
        REQUIRE( bvh::m::approx_equals( ret, expected ) );
      }

#if 0
      SECTION( "constant_vec" )
      {
      }
#endif
    }

    SECTION( "post" )
    {
      SECTION( "vec" )
      {
        double a = 2.0;
        bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

        auto ret = a / b;

        auto expected = bvh::m::vec< double, 4 >{ 0.5, 2.0 / 3.0, 1.0, 2.0 };
        REQUIRE( ret == expected );
      }

#if 0
      SECTION( "constant_vec" )
      {
      }
#endif
    }
  }


  SECTION( "5" )
  {
    SECTION( "post" )
    {
      SECTION( "vec" )
      {
        double a = 2.0;
        bvh::m::vec< double, 5 > b{ 5.0, 4.0, 3.0, 2.0, 1.0 };

        auto ret = a / b;

        auto expected = bvh::m::vec< double, 5 >{ 2.0 / 5.0, .5, 2.0 / 3.0, 1.0, 2.0 };
        REQUIRE( ret == expected );
      }

#if 0
      SECTION( "constant_vec" )
      {
      }
#endif
    }
  }
}


TEST_CASE( "vec hadd", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };

      auto ret = hadd( a );

      auto expected = 10.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 5 > a{ 1.0, 2.0, 3.0, 4.0, 5.0 };

      auto ret = hadd( a );

      auto expected = 15.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec dot", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
      bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };

      auto ret = dot( a, b );

      auto expected = 20.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 5 > a{ 1.0, 2.0, 3.0, 4.0, 5.0 };
      bvh::m::vec< double, 5 > b{ 5.0, 4.0, 3.0, 2.0, 1.0 };

      auto ret = dot( a, b );

      auto expected = 35.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec length2", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };

      auto ret = length2( a );

      auto expected = 30.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 5 > a{ 1.0, 2.0, 3.0, 4.0, 5.0 };

      auto ret = length2( a );

      auto expected = 55.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec length", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 2.0, 24.0, 16.0, 8.0 };

      auto ret = length( a );

      auto expected = 30.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 5 > a{ 6.0, 48.0, 36.0, 24.0, 12.0 };

      auto ret = length( a );

      auto expected = 66.0;
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec sqrt", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 9.0, 16.0, 1.0, 25.0 };

      auto ret = sqrt( a );

      auto expected = bvh::m::vec< double, 4 >{ 3.0, 4.0, 1.0, 5.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec normal", "[math][vec]" )
{
  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      bvh::m::vec< double, 4 > a{ 2.0, 24.0, 16.0, 8.0 };

      auto ret = normal( a );

      auto expected = bvh::m::vec< double, 4 >{ 2.0 / 30.0, 24.0 / 30.0, 16.0 / 30.0, 8.0 / 30.0 };
      REQUIRE( ret == expected );
    }

#if 0
    SECTION( "constant_vec" )
    {
      static constexpr bvh::m::constant_vec< double, 4 > a{ 5.0 };

      constexpr auto ret = a.normal();
    }
#endif
  }
}


TEST_CASE( "vec ceil", "[math][vec]" )
{
  SECTION( "3" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 3 >( 1.1, 2.2, 3.3 );

      auto b = ceil( a );

      REQUIRE( b[0] == 2.0 );
      REQUIRE( b[1] == 3.0 );
      REQUIRE( b[2] == 4.0 );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 4 >( 1.1, 2.2, 3.3, 4.4 );

      auto b = ceil( a );

      REQUIRE( b[0] == 2.0 );
      REQUIRE( b[1] == 3.0 );
      REQUIRE( b[2] == 4.0 );
      REQUIRE( b[3] == 5.0 );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 5 >( 1.1, 2.2, 3.3, 4.4, 5.5 );

      auto b = ceil( a );

      REQUIRE( b[0] == 2.0 );
      REQUIRE( b[1] == 3.0 );
      REQUIRE( b[2] == 4.0 );
      REQUIRE( b[3] == 5.0 );
      REQUIRE( b[4] == 6.0 );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec floor", "[math][vec]" )
{
  SECTION( "3" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 3 >( 1.1, 2.2, 3.3 );

      auto b = floor( a );

      REQUIRE( b[0] == 1.0 );
      REQUIRE( b[1] == 2.0 );
      REQUIRE( b[2] == 3.0 );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "4" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 4 >( 1.1, 2.2, 3.3, 4.4 );

      auto b = floor( a );

      REQUIRE( b[0] == 1.0 );
      REQUIRE( b[1] == 2.0 );
      REQUIRE( b[2] == 3.0 );
      REQUIRE( b[3] == 4.0 );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }

  SECTION( "5" )
  {
    SECTION( "vec" )
    {
      auto a = bvh::m::vec< double, 5 >( 1.1, 2.2, 3.3, 4.4, 5.5 );

      auto b = floor( a );

      REQUIRE( b[0] == 1.0 );
      REQUIRE( b[1] == 2.0 );
      REQUIRE( b[2] == 3.0 );
      REQUIRE( b[3] == 4.0 );
      REQUIRE( b[4] == 5.0 );
    }

#if 0
    SECTION( "constant_vec" )
    {
    }
#endif
  }
}


TEST_CASE( "vec access", "[math][vec]" )
{
  SECTION( "vec" )
  {
    auto a = bvh::m::vec< double, 3 >( 1.0, 2.0, 3.0 );

    REQUIRE( a.x() == 1.0 );
    REQUIRE( a.y() == 2.0 );
    REQUIRE( a.z() == 3.0 );
  }

  SECTION( "constant_vec" )
  {
    auto a = bvh::m::constant_vec< double, 3 >( 1.0, 2.0, 3.0 );

    REQUIRE( a.x() == 1.0 );
    REQUIRE( a.y() == 2.0 );
    REQUIRE( a.z() == 3.0 );
  }
}

TEST_CASE( "vec const convert", "[math][vec]" )
{
  SECTION( "4" )
  {
    auto a = bvh::m::constant_vec< double, 4 >( 1.0, 2.0, 3.0, 4.0 );

    bvh::m::vec< double, 4 > b = a;

    REQUIRE( b[0] == 1.0 );
    REQUIRE( b[1] == 2.0 );
    REQUIRE( b[2] == 3.0 );
    REQUIRE( b[3] == 4.0 );
  }

  SECTION( "5" )
  {
    auto a = bvh::m::constant_vec< double, 5 >( 1.0, 2.0, 3.0, 4.0, 5.0 );

    bvh::m::vec< double, 5 > b = a;

    REQUIRE( b[0] == 1.0 );
    REQUIRE( b[1] == 2.0 );
    REQUIRE( b[2] == 3.0 );
    REQUIRE( b[3] == 4.0 );
    REQUIRE( b[4] == 5.0 );
    REQUIRE( b[5] == 0.0 );
    REQUIRE( b[6] == 0.0 );
    REQUIRE( b[7] == 0.0 );
  }
}

#if 0
TEST(VecTest, plus_equals4)
{
  bvh::m::vec< double, 4 > a{ 1.0, 2.0, 3.0, 4.0 };
  bvh::m::vec< double, 4 > b{ 4.0, 3.0, 2.0, 1.0 };
  
  a += b;
  
  auto expected = bvh::m::vec< double, 4 >{ 5.0, 5.0, 5.0, 5.0 };
  EXPECT_EQ( a, expected );
}

TEST(VecTest, constexpr_add4)
{
}

#endif
