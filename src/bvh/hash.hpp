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
#ifndef INC_BVH_HASH_HPP
#define INC_BVH_HASH_HPP

#include <cstdint>
#include "math/vec.hpp"

#if !defined(BVH_ENABLE_KOKKOS) || defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
#include <immintrin.h>
#endif

#ifdef BVH_ENABLE_KOKKOS
#include "util/kokkos.hpp"
#endif

namespace bvh
{
  namespace detail
  {
    inline std::uint32_t expand32( std::uint32_t _num )
    {
      _num = ( _num * 0x00010001u ) & 0xff0000ffu;
      _num = ( _num * 0x00000101u ) & 0x0f00f00fu;
      _num = ( _num * 0x00000011u ) & 0xc30c30c3u;
      _num = ( _num * 0x00000005u ) & 0x49249249u;

      return _num;
    }

    inline std::uint32_t expand32_alt( std::uint32_t _num )
    {
      _num = ( _num ^ ( _num << 16 ) ) & 0xff0000ff;
      _num = ( _num ^ ( _num << 8  ) ) & 0x0300f00f;
      _num = ( _num ^ ( _num << 4  ) ) & 0x030c30c3;
      _num = ( _num ^ ( _num << 2  ) ) & 0x09249249;

      return _num;
    }



    inline std::uint64_t expand64( std::uint64_t _21bit )
    {
      _21bit = ( _21bit * 0x1000001u ) & 0xfff000000fffu;
      _21bit = ( _21bit * 0x1001u ) & 0xfc003f000fc003fu;
      _21bit = ( _21bit * 0x41u ) & 0x01c0e070381c0e07u;
      _21bit = ( _21bit * 0x11u ) & 0x10c86432190c8643u;
      _21bit = ( _21bit * 0x5u ) & 0x1249249249249249u;

      return _21bit;
    }

#ifdef __BMI2__
    inline std::uint64_t expand64intrin( std::uint64_t _21bit )
    {
      return _pdep_u64( _21bit, 0x1249249249249249u );
    }

    inline std::uint64_t morton64_intrin( std::uint64_t _x21, std::uint64_t _y21, std::uint64_t _z21 )
    {
      return ( expand64intrin( _z21 ) << 2u ) | ( expand64intrin( _y21 ) << 1u ) | expand64intrin( _x21 );
    }
#endif


    inline std::uint64_t morton64( std::uint64_t _x21, std::uint64_t _y21, std::uint64_t _z21 )
    {
      return ( expand64( _z21 ) << 2u ) | ( expand64( _y21 ) << 1u ) | expand64( _x21 );
    }
  }

  using morton32_t = std::uint32_t;
  using morton64_t = std::uint64_t;

  inline std::uint32_t morton( std::uint32_t _x, std::uint32_t _y, std::uint32_t _z )
  {
    return ( detail::expand32( _z ) << 2 ) + ( detail::expand32( _y ) << 1 ) + detail::expand32( _x );
  }

  inline std::uint64_t morton( std::uint64_t _x, std::uint64_t _y, std::uint64_t _z )
  {
#ifdef __BMI2__
    // If it's available, BMI2 intrinsics are around 4x faster
    return detail::morton64_intrin( _x, _y, _z );
#else
    return detail::morton64( _x, _y, _z );
#endif
  }

#ifdef BVH_ENABLE_KOKKOS
  template< typename T >
  void morton( const view< T *[3] > _points, const m::vec3< T > &_min, const m::vec3< T > &_max,
    view< std::uint32_t * > _codes )
  {
    Kokkos::parallel_for( _points.extent( 0 ), [_points, _min, _max, _codes] KOKKOS_FUNCTION ( int i ){
      auto p = m::vec3< T >( _points( i, 0 ), _points( i, 1 ), _points( i, 2 ) );
      auto norm = ( p - _min ) / ( _max - _min );
      auto clamped = m::clamp( norm * T{ 1024 }, m::vec3< T >{ T{ 0 } }, m::vec3< T >{ T{ 1023 } } );

      _codes( i ) = ::bvh::morton( static_cast< std::uint32_t >( clamped.x() ),
        static_cast< std::uint32_t >( clamped.y() ),
        static_cast< std::uint32_t >( clamped.z() ) );
    } );
  }
#endif
}

#endif  // INC_BVH_HASH_HPP
