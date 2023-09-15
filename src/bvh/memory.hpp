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
#ifndef INC_BVH_MEMORY_HPP
#define INC_BVH_MEMORY_HPP

#include <cstddef>

#include <xmmintrin.h>

namespace bvh
{
  template< typename T, unsigned Align = alignof( T ) >
  struct aligned_allocator
  {
    using pointer = T *;
    using const_pointer = const T *;
    using void_pointer = void *;
    using const_void_pointer = const void *;

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template< typename U >
    struct rebind
    {
      using other = aligned_allocator< U, Align >;
    };

    pointer
    allocate( size_type _n )
    {
      return static_cast< pointer >( _mm_malloc( sizeof( T ) * _n, Align ) );
    }

    void
    deallocate( pointer _p, size_type )
    {
      _mm_free( _p );
    }
  };

  template< typename T, unsigned Align >
  bool
  operator==( const aligned_allocator< T, Align > &, const aligned_allocator< T, Align > & )
  {
    return true;
  }

  template< typename T, unsigned Align >
  bool
  operator!=( const aligned_allocator< T, Align > &, const aligned_allocator< T, Align > & )
  {
    return false;
  }
} // namespace bvh

#endif // INC_BVH_MEMORY_HPP
