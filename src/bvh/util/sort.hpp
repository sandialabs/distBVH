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
#ifndef INC_BVH_UTIL_SORT_HPP
#define INC_BVH_UTIL_SORT_HPP

#include <cstdint>

#ifdef BVH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#endif  // BVH_ENABLE_KOKKOS

#include "prefix_sum.hpp"
#include "kokkos.hpp"

#include <iostream>

namespace bvh
{
  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#ifdef BVH_ENABLE_KOKKOS
  namespace kokkos
  {
    namespace detail
    {
      inline void
      radix_sort_indices_iter( view< uint32_t * > _hashes, view< unsigned long * > _indices,
        uint32_t _shift )
      {
        const auto n = _hashes.extent( 0 );
  
        view< uint32_t * > f( "FalseIndices", n );
        view< uint32_t * > split( "Split", n );
  
        Kokkos::parallel_for( n, [_hashes, f, split, _shift] KOKKOS_FUNCTION ( int i ){
          uint32_t h = _hashes( i ) >> _shift;
          split( i ) = h & 0x1;
          f( i ) = ~h & 0x1;
        } );
        
        prefix_sum( f );
        Kokkos::parallel_for( n, [f, n, _indices, split] KOKKOS_FUNCTION ( int i ){
          const auto total = f( n - 1 ) + static_cast< uint32_t >( ~split( n - 1 ) & 0x1 );
          
          auto t = i - f( i ) + total;
          
          _indices( i ) = split( i ) ? t : f( i );
        } );
      }
    }
    
    template< typename T >
    void
    radix_sort( view< uint32_t * > _hashes, view< T > _objects )
    {
      const auto n = _hashes.extent( 0 );
  
      view< unsigned long * > m( "Indices", n );
      
      view< uint32_t * > a( "a", n );
      view< uint32_t * > b( "b", n );
      view< T > a_objects( "a_objects", n );
      view< T > b_objects( "b_objects", n );
      
      auto *front = &a;
      auto *back = &b;
      Kokkos::deep_copy( *front, _hashes );
      
      auto *front_objects = &a_objects;
      auto *back_objects = &b_objects;
      Kokkos::deep_copy( *front_objects, _objects );
  
      for ( std::uint32_t i = 0; i < 32; ++i )
      {
        detail::radix_sort_indices_iter( *front, m, i );
        
        // Init captures not allowed in cuda
        auto &fbuff = *front;
        auto &bbuff = *back;
        auto &fobjs = *front_objects;
        auto &bobjs = *back_objects;

        Kokkos::parallel_for( n, [m, fbuff, bbuff, fobjs, bobjs] KOKKOS_FUNCTION ( int i ) {
          bbuff( m( i ) ) = fbuff( i );
          bobjs( m( i ) ) = fobjs( i );
        } );
        
        std::swap( front, back );
        std::swap( front_objects, back_objects );
      }
      
      Kokkos::deep_copy( _hashes, *front );
      Kokkos::deep_copy( _objects, *front_objects );
    }
  }
#endif  // BVH_ENABLE_KOKKOS
}

#endif  // INC_BVH_UTIL_SORT_HPP
