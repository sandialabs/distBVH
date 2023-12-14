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
#ifndef INC_BVH_PRIMITIVE_KOKKOS_CLOSEST_POINT_HPP
#define INC_BVH_PRIMITIVE_KOKKOS_CLOSEST_POINT_HPP

#include "../util/kokkos.hpp"
#include "../util/attributes.hpp"
#include "triangle.hpp"

namespace bvh
{
  namespace kokkos
  {
    namespace detail
    {
      template< typename T >
      BVH_INLINE void store_vec( const m::vec3< T > &_v, view< T *[3] > _dest, int _index )
      {
        _dest( _index, 0 ) = _v[0];
        _dest( _index, 1 ) = _v[1];
        _dest( _index, 2 ) = _v[2];
      }
    }

    template< typename T >
    void
    closest_point( const view< T *[3] > _points,
      const triangle< T > &_tri, view< T *[3] > _closest )
    {
      using vec = m::vec3< T >;
      auto ab = _tri.b - _tri.a;
      auto ac = _tri.c - _tri.a;
      auto bc = _tri.c - _tri.b;

      Kokkos::parallel_for( _points.extent( 0 ), [_points, _closest, _tri, ab, ac, bc] KOKKOS_FUNCTION ( int i ){
        auto p = vec{ _points( i, 0 ), _points( i, 1 ), _points( i, 2 ) };

        auto ap = p - _tri.a;
        auto bp = p - _tri.b;
        auto cp = p - _tri.c;

        // Vertex region a
        auto d1 = m::dot( ab, ap );
        auto d2 = m::dot( ac, ap );
        if ( d1 <= T{ 0 } && d2 <= T{ 0 } )
        {
          detail::store_vec( _tri.a, _closest, i );
          return; // next iteration
        }

        // Vertex region b
        auto d3 = m::dot( ab, bp );
        auto d4 = m::dot( ac, bp );
        if ( d3 >= T{ 0 } && d4 <= d3 )
        {
          detail::store_vec( _tri.b, _closest, i );
          return; // next iteration
        }

        // Edge region ab
        auto vc = d1 * d4 - d3 * d2;
        if ( vc <= T{ 0 } && d1 >= T{ 0 } && d3 <= T{ 0 } )
        {
          auto v = d1 / ( d1 - d3 );
          detail::store_vec( _tri.a + v * ab, _closest, i );
          return; // next iteration
        }

        // Vertex region c
        auto d5 = m::dot( ab, cp );
        auto d6 = m::dot( ac, cp );

        if ( d6 >= T{ 0 } && d5 <= d6 )
        {
          detail::store_vec( _tri.c, _closest, i );
          return; // next iteration
        }

        // Edge region ac
        auto vb = d5 * d2 - d1 * d6;
        if ( vb <= T{ 0 } && d2 >= T{ 0 } && d6 <= T{ 0 } )
        {
          auto w = d2 / ( d2 - d6 );
          detail::store_vec( _tri.a + w * ac, _closest, i );
          return; // next iteration
        }

        // Edge region bc
        auto va = d3 * d6 - d5 * d4;
        if ( va <= T{ 0 } && ( d4 - d3 ) >= T{ 0 } && ( d5 - d6 ) >= T{ 0 } )
        {
          auto w = ( d4 - d3 ) / ( ( d4 - d3 ) + ( d5 - d6 ) );
          detail::store_vec( _tri.b + w * bc, _closest, i );
          return; // next iteration
        }

        // Triangle interior
        auto denom = T{ 1 } / ( va + vb + vc );
        auto v = vb * denom;
        auto w = vc * denom;
        detail::store_vec( _tri.a + ab * v + ac * w, _closest, i );
      } );
    }
  }
}

#endif  // INC_BVH_PRIMITIVE_KOKKOS_CLOSEST_POINT_HPP
