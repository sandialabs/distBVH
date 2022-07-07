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
#ifndef INC_BVH_NARROWPHASE_KOKKOS_HPP
#define INC_BVH_NARROWPHASE_KOKKOS_HPP

#ifdef BVH_ENABLE_KOKKOS

#include <limits>
#include <array>
#include <Kokkos_Sort.hpp>
#include "../util/kokkos.hpp"
#include "../util/sort.hpp"
#include "../patch.hpp"
#include "../vt/print.hpp"

namespace bvh
{
  namespace kokkos
  {
    namespace detail
    {
      template< typename Element >
      struct centroid_sum
      {
        using arithmetic_type = float_type;
        using view_type = view< arithmetic_type *[3] >;
        using value_type = arithmetic_type[];
        using size_type = typename view_type::size_type;
        static constexpr size_type value_count = 3;

        explicit centroid_sum( const view_type &_view )
          : in_view( _view )
        {}

        KOKKOS_INLINE_FUNCTION void operator()( const unsigned long i, value_type sum ) const
        {
          for ( int j = 0; j < 3; ++j )
            sum[j] += in_view( i, j );
        }

        KOKKOS_INLINE_FUNCTION void join( volatile value_type _dst, const volatile value_type _src ) const
        {
          for ( int i = 0; i < 3; ++i )
            _dst[i] += _src[i];
        }

        KOKKOS_INLINE_FUNCTION void init( value_type _sum )
        {
          for ( int i = 0; i < 3; ++i )
            _sum[i] = arithmetic_type{ 0 };
        }

        view_type in_view;

      };

      template< typename Element >
      struct centroid_sum2
      {
        using arithmetic_type = float_type;
        using view_type = view< arithmetic_type *[3] >;
        using value_type = arithmetic_type[];
        using size_type = typename view_type::size_type;
        static constexpr size_type value_count = 3;

        explicit centroid_sum2( const view_type &_view )
          : in_view( _view )
        {}

        KOKKOS_INLINE_FUNCTION void operator()( const unsigned long i, value_type sum ) const
        {
          for ( int j = 0; j < 3; ++j )
            sum[j] += in_view( i, j ) * in_view( i, j );
        }

        KOKKOS_INLINE_FUNCTION void join( volatile value_type _dst, const volatile value_type _src ) const
        {
          for ( int i = 0; i < 3; ++i )
            _dst[i] += _src[i];
        }

        KOKKOS_INLINE_FUNCTION void init( value_type _sum )
        {
          for ( int i = 0; i < 3; ++i )
            _sum[i] = arithmetic_type{ 0 };
        }

        view_type in_view;

      };
    }

    template< typename View >
    int max_variant_axis( const View &_a, const View &_b )
    {
      using entity_type = typename View::value_type;
      view< float_type *[3] > c( "Centroids", _a.size() + _b.size() );

      // Get the centroids
      Kokkos::parallel_for( _a.size(), [_a, c] KOKKOS_FUNCTION ( int i ){
        for ( int j = 0; j < 3; ++j )
          c( i, j ) = _a( i ).centroid()[j];
      } );

      Kokkos::parallel_for( _b.size(), [_a, _b, c] KOKKOS_FUNCTION ( int i ){
        for ( int j = 0; j < 3; ++j )
          c( i + _a.size(), j ) = _b( i ).centroid()[j];
      } );


      float_type s[3];
      float_type s2[3];

      auto sum1f = detail::centroid_sum< entity_type >( c );
      auto sum2f = detail::centroid_sum2< entity_type >( c );

      // Get the sum and the squared sum for computing the variance
      Kokkos::parallel_reduce( _a.size() + _b.size(), sum1f, s );
      Kokkos::parallel_reduce( _a.size() + _b.size(), sum2f, s2 );

      // Compute the variance of the centroid
      // Smaller variance can indicate clustering, we want to avoid that as much as
      // possible or the running time of sort and sweep is O(n^2)
      m::vec3< float_type > var;
      for ( int i = 0; i < 3; ++i )
      {
        var[i] = s2[i] - s[i] * s[i] / ( _a.size() + _b.size() );
      }

      int axis = 0;
      if ( var[1] > var[0] ) axis = 1;
      if ( var[2] > var[axis] ) axis = 2;

      return axis;
    }


    template< typename View, typename F >
    void
    sort_and_sweep_local( const patch<> &_pa, const View &_a,
                          const patch<> &_pb, const View &_b, int _axis,
                    F &&_fun )
    {
      using element_type = typename View::value_type;

      view< std::uint32_t * > extents_b_min( "ExtentsBMin", _b.size() );
      view< std::size_t * > indices_b( "IndicesB", _b.size() );

      float_type global_min = std::min(  _pa.kdop().extents[_axis].min, _pb.kdop().extents[_axis].min );
      float_type global_max = std::max(  _pa.kdop().extents[_axis].max, _pb.kdop().extents[_axis].max );

      float_type conversion_fac = static_cast< float_type >( std::numeric_limits< uint32_t >::max() ) / ( global_max - global_min );

      // Reset indices
      Kokkos::parallel_for( _b.size(), [indices_b] KOKKOS_FUNCTION ( int i ) {
        indices_b( i ) = i;
      } );

      // Radix sort for floating point types requires some adjusting,
      // so for each axis, we will divide the range of values by our bitset width
      Kokkos::parallel_for( _b.size(), [_axis, _b, extents_b_min, global_min, conversion_fac] KOKKOS_FUNCTION ( int i ) {
        const auto &bk = element_traits< element_type >::get_kdop( _b( i ) );
        extents_b_min( i ) = ( bk.extents[_axis].min - global_min ) * conversion_fac;
      } );

      // Sort extents
      radix_sorter< std::uint32_t, std::size_t > sorter{ extents_b_min.extent( 0 ) };
      sorter( extents_b_min, indices_b );

      // Sweep along a on the axis
      //view< unsigned long * > num_collisions( "NumCollisions", _a.size() );
      //view< unsigned long ** > collisions( "Collisions", _a.size(), _b.size() );

      Kokkos::parallel_for( _a.size(), [_a, _b, extents_b_min, indices_b, global_min, conversion_fac, _axis, &_fun] KOKKOS_FUNCTION ( int i )
      {
        const auto &ak = element_traits< element_type >::get_kdop( _a( i ) );

        for ( unsigned long j = 0; j < _b.size(); ++j )
        {
          std::uint32_t amin = ( ak.extents[_axis].min - global_min ) * conversion_fac;
          std::uint32_t amax = ( ak.extents[_axis].max - global_min ) * conversion_fac;

          const auto &bk = element_traits< element_type >::get_kdop( _b( indices_b( j ) ) );
          auto bmin = extents_b_min( j );
          auto bmax = ( bk.extents[_axis].max - global_min ) * conversion_fac;

          // Before the range of A, no collision yet
          if ( bmax < amin )
            continue;

          // After the range of A, there will be no more collisions
          if ( bmin > amax )
            return;

          if ( overlap( ak, bk ) )
          {
            //vt::print( "overlap a_{} with b_{}\n", i, indices_b( j ) );
            auto &&a = _a( i );
            auto &&b = _b( indices_b( j ) );
            std::forward< F >( _fun )( a, b );
          }
            //collisions( i, count++ ) = indices_b( j );
        }

        //num_collisions( i ) = count;
      } );

      Kokkos::fence();

      /*
      for ( std::size_t i = 0; i < _a.size(); ++i )
      {
        auto &&a = _a[i];

        for ( unsigned long j = 0; j < num_collisions( i ); ++j )
        {
          auto &&b = _b[collisions( i, j )];
          std::forward< F >( _fun )( a.local_id(), b.local_id() );
        }
      }
       */
    }
  }
}
#endif  // BVH_ENABLE_KOKKOS

#endif  // INC_BVH_NARROWPHASE_KOKKOS_HPP
