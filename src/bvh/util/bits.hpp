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
#ifndef INC_BVH_UTIL_BITS_HPP
#define INC_BVH_UTIL_BITS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if !defined(KOKKOS_COMPILER_NVCC)
#include <immintrin.h>
#endif

namespace bvh
{
  template< typename To, typename From, typename = std::enable_if_t<
      ( sizeof( To ) <= sizeof( From ) )
      && std::is_trivially_copyable< To >::value
      && std::is_trivially_copyable< From >::value > >
  To bit_cast( const From &_from )
  {
#ifdef __GNUC__
    To t = t;
    __builtin_memcpy( &t, &_from, sizeof( To ) );
#else
#error "unsupported compiler for find first set"
#endif
    return t;
  }
  template< typename T, unsigned char V >
  T fill()
  {
#ifdef __GNUC__
    T t = t;
    __builtin_memset( &t, V, sizeof( T ) );
#else
#error "unsupported compiler for find first set"
#endif
    return t;
  }

  inline unsigned ffs( unsigned _val )
  {
#ifdef __GNUC__
    return static_cast< unsigned >( __builtin_ffs( _val ) );
#else
#error "unsupported compiler for find first set"
#endif
  }

  inline std::uint64_t ffs( std::uint64_t _val )
  {
#ifdef __GNUC__
    return static_cast< std::uint64_t >( __builtin_ffsl( _val ) );
#else
#error "unsupported compiler for find first set"
#endif
  }

  inline std::uint32_t clz( std::uint32_t _val )
  {
#if !defined(KOKKOS_COMPILER_NVCC)
    return _lzcnt_u32( _val );
#else
    return 0u;
#endif
  }

  inline std::uint64_t clz( std::uint64_t _val )
  {
#if !defined(KOKKOS_COMPILER_NVCC)
    return _lzcnt_u64( _val );
#else
    return 0u;
#endif
  }

  inline int bsr( unsigned long _val )
  {
#ifdef __GNUC__
    return __builtin_clzl( _val ) ^ 63;
#else
#error "unsupported compiler for find first set"
#endif
  }

  template< typename T >
  inline T bit_log2( T _val )
  {
    return bsr( _val );
  }

  inline bool is_pow2( unsigned _val )
  {
    return _val & ( _val - 1 );
  }

  template< typename T >
  T next_pow2( T _val )
  {
      return ( _val == 1 ) ? 1 : 0x1 << ( bit_log2( _val - 1 ) + 1 );
  }
}

#endif  // INC_BVH_UTIL_BITS_HPP
