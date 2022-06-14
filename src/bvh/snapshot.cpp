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
#include "snapshot.hpp"

namespace bvh
{
  namespace
  {
    struct bounds_union
    {
      using value_type = bphase_kdop;
      using size_type = typename view< const entity_snapshot * >::size_type;

      explicit bounds_union( const view< const entity_snapshot * > &_v )
        : entities( _v )
      {}

      KOKKOS_INLINE_FUNCTION
      void
      operator()( size_type _i, bphase_kdop &_update ) const noexcept
      {
        _update.union_with( entities( _i ).kdop() );
      }

      KOKKOS_INLINE_FUNCTION
      void
      join( value_type &_dst, const value_type &_src ) const noexcept
      {
        _dst.union_with( _src );
      }

      KOKKOS_INLINE_FUNCTION
      void
      init( value_type &_dst ) const noexcept
      {
        _dst = value_type{};
      }

      view< const entity_snapshot * > entities;
    };
  }

  KOKKOS_FUNCTION
  void
  compute_bounds( view< const entity_snapshot * > _elements,
                  single_view< bphase_kdop > _bounds )
  {
    Kokkos::parallel_reduce( "compute_bounds", _elements.extent( 0 ),
                              bounds_union{ _elements }, _bounds );
  }

  namespace
  {
    template< typename T >
    KOKKOS_INLINE_FUNCTION
    void
    morton_impl( view< const entity_snapshot * > _elements,
          single_view< const bphase_kdop > _bounds,
          view< T * > _out_codes )
    {
      Kokkos::parallel_for( _elements.extent( 0 ), KOKKOS_LAMBDA( int _i ){
        const auto p = _elements( _i ).centroid();
        m::vec3< T > discretized = ( ( p - _bounds( 0 ).cardinal_min() ) / ( _bounds( 0 ).cardinal_max() - _bounds( 0 ).cardinal_min() ) ) * std::numeric_limits< T >::max();

        _out_codes( _i ) = morton( discretized.x(), discretized.y(), discretized.z() );
      } );
    }
  }

  KOKKOS_FUNCTION
  void
  morton( view< const entity_snapshot * > _elements,
          single_view< const bphase_kdop > _bounds,
          view< morton32_t * > _out_codes )
  {
    morton_impl( _elements, _bounds, _out_codes );
  }
}