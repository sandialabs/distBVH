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
#ifndef INC_BVH_MATH_COMMON_HPP
#define INC_BVH_MATH_COMMON_HPP

#ifdef __CUDA_ARCH__
#include <float.h>
#include <math.h>
#else
#include <cmath>
#include <limits>
#include <type_traits>
#endif

#include <cstdint>

#include "compiler.hpp"

namespace bvh
{
  namespace m
  {
    constexpr double pi = 3.141592653589793238462643383279502884197169399375;

#ifdef __CUDA_ARCH__
    constexpr double epsilon = DBL_EPSILON;
    constexpr double epsilonf = FLT_EPSILON;
#else
    constexpr double epsilon = std::numeric_limits< double >::epsilon();
    constexpr double epsilonf = std::numeric_limits< float >::epsilon();

    template< typename T >
    constexpr T epsilon_value = std::numeric_limits< T >::epsilon();
#endif

    namespace detail
    {
      template< typename T >
      struct clamp_impl
      {
        static BVH_INLINE T
        clamp( T v, T min, T max ) noexcept
        {
          return ( v < min ) ? min : ( ( v > max ) ? max : v );
        }
      };

      template< typename T >
      struct minmax_impl
      {
        static BVH_INLINE T
        min( T a, T b ) noexcept
        {
          return ( a < b ) ? a : b;
        }

        static BVH_INLINE T
        max( T a, T b ) noexcept
        {
          return ( a > b ) ? a : b;
        }
      };
    } // namespace detail

    template< typename T >
    constexpr BVH_INLINE T
    abs( T v ) noexcept
    {
      return std::abs( v );
    }

    template< typename T >
    BVH_INLINE T
    clamp( T v, T min, T max ) noexcept
    {
      return detail::clamp_impl< T >::clamp( v, min, max );
    }

    template< typename T >
    BVH_INLINE T
    min( T a, T b ) noexcept
    {
      return detail::minmax_impl< T >::min( a, b );
    }

    template< typename T >
    BVH_INLINE T
    max( T a, T b ) noexcept
    {
      return detail::minmax_impl< T >::max( a, b );
    }

    template< typename T >
    BVH_INLINE T
    wrap( T v, T min, T max )
    {
      // TODO: work for integral types

      T scale = max - min;
      // normalize
      v = ( v - min ) / scale;

      v -= std::floor( v );
      return v * scale;
    }

    template< typename T >
    BVH_INLINE T
    rad( T v )
    {
      return v * pi / T( 180 );
    }

    template< typename T >
    BVH_INLINE bool
    sign( T _v )
    {
      return std::signbit( _v );
    }

    template< typename T >
    BVH_INLINE decltype( std::sqrt( T() ) )
    sqrt( T _x )
    {
      return std::sqrt( _x );
    }

    constexpr bool
    BVH_INLINE approx_equals( int _a, int _b, int )
    {
      return _a == _b;
    }

    constexpr bool
    BVH_INLINE approx_equals( float _a, float _b, float _eps = epsilonf )
    {
      return abs( _a - _b ) < _eps;
    }

    constexpr bool
    BVH_INLINE approx_equals( double _a, double _b, double _eps = epsilon )
    {
      return abs( _a - _b ) < _eps;
    }

#ifdef BVH_EXPERIMENTAL
#if defined( __GNUC__ ) && !defined( __clang__ )
    template<>
    constexpr double
    sqrt< double >( double _x )
    {
      return __builtin_sqrt( _x );
    }

    template<>
    constexpr float
    sqrt< float >( float _x )
    {
      return __builtin_sqrtf( _x );
    }

    template<>
    constexpr long double
    sqrt< long double >( long double _x )
    {
      return __builtin_sqrtl( _x );
    }
#endif
#endif
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_COMMON_HPP
