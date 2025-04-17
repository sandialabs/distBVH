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
#include <cstdlib>

#include "prefix_sum.hpp"
#include "kokkos.hpp"

#include <iostream>

namespace bvh
{
  // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
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

  template< typename T, typename IndexType = ::std::uint32_t >
  class radix_sorter
  {
  public:

    static constexpr std::uint32_t num_bits = sizeof( T ) * 8;

    radix_sorter() = default;
    explicit radix_sorter( std::size_t _n )
      : m_scratch( "radix_sort_scratch", _n ),
        m_index_scratch( "radix_sort_index_scratch", _n ),
        m_scan( "radix_sort_scan", _n ),
        m_bits( "radix_sort_bits", _n )
    {

    }

    void resize_scratch( std::size_t _n )
    {
      Kokkos::resize( m_scratch, _n );
      Kokkos::resize( m_index_scratch, _n );
      Kokkos::resize( m_scan, _n );
      Kokkos::resize( m_bits, _n );
    }

    void operator()( view< T * > _hashes, view< IndexType * > _indices )
    {
      assert( _hashes.extent( 0 ) == _indices.extent( 0 )
        && _hashes.extent( 0 ) == m_scratch.extent( 0 ) );

      for ( std::uint32_t i = 0; i < num_bits; ++i )
      {
        step( _hashes, _indices, i );

        std::swap( m_scratch, _hashes );
        std::swap( m_index_scratch, _indices );
        // Number of bits is always even, and we know on odd numbered
        // iterations we are reading from m_scratch and writing to _hashes
        // So when this loop ends, _hashes will contain the results
      }
    }

    void step( view< T * > _hashes, view< IndexType * > _indices, std::uint32_t _shift )
    {
      const auto n = _hashes.extent( 0 );
      Kokkos::parallel_for( n, KOKKOS_CLASS_LAMBDA ( int i ){
        auto h = _hashes( i ) >> _shift;
        m_bits( i ) = ~h & 0x1;
        m_scan( i ) = m_bits( i );
      } );

      prefix_sum( m_scan );
      Kokkos::parallel_for( n, KOKKOS_CLASS_LAMBDA ( int i ){
        const auto total = m_scan( n - 1 ) + m_bits( n - 1 );

        auto t = i - m_scan( i ) + total;
        auto new_idx = m_bits( i ) ? m_scan( i ) : t;
        m_index_scratch( new_idx ) = _indices( i );
        m_scratch( new_idx ) = _hashes( i );
      } );
    }

  private:

    view< T * > m_scratch;
    view< IndexType * > m_index_scratch;
    view< T * > m_scan;
    view< T * > m_bits;
  };
}

#endif  // INC_BVH_UTIL_SORT_HPP
