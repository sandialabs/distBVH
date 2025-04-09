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
#ifndef INC_BVH_PATCH_HPP
#define INC_BVH_PATCH_HPP

#include <numeric>
#include "util/span.hpp"
#include "range.hpp"
#include "iterators/transform_iterator.hpp"
#include "math/vec.hpp"
#include <vector>
#include "snapshot.hpp"

#include <Kokkos_Core.hpp>

namespace bvh
{
  /**
   * Utility class for representing a patch, or collection of elements. Does not store the
   * elements themselves, but rather stores information that is useful for constructing a
   * bounding volume hierarchy, such as the centroid, bounds, and reference id.
   *
   * \tparam Element  The type of element to store in the patch.
   */
  template< typename KDop = bphase_kdop >
  class patch
  {
  public:

    using index_type = std::size_t;
    using kdop_type = KDop;
    using size_type = std::size_t;

    patch() = default;

    template< typename E >
    patch( index_type _patch_id, span< const E > _span )
      : m_global_id( _patch_id )
    {
      set_elements( _span );
    }

    patch( index_type _patch_id, std::size_t _size, const kdop_type &_kdop,
          const m::vec3< float_type > &_centroid )
        : m_global_id( _patch_id ), m_size( _size ), m_kdop( _kdop ),
          m_centroid( _centroid )
    {}

    template< typename Element >
    void set_elements( span< const Element > _span ) noexcept
    {
      if ( _span.size() > 0 )
        compute_kdops_and_centroid( _span );
      m_size = _span.size();
    }

    KOKKOS_INLINE_FUNCTION index_type global_id() const noexcept { return m_global_id; }

    KOKKOS_INLINE_FUNCTION kdop_type kdop() const noexcept { return m_kdop; }

    KOKKOS_INLINE_FUNCTION m::vec3< float_type > centroid() const noexcept { return m_centroid; }

    bool empty() const noexcept { return m_size == 0; }
    size_type size() const noexcept { return m_size; }

    template< typename Serializer >
    friend void serialize( Serializer &_s, const patch &_patch )
    {
      _s | _patch.m_global_id | _patch.m_size | _patch.m_kdop | _patch.m_centroid;
    }

  private:

    template< typename Element >
    void compute_kdops_and_centroid( span< const Element > _span )
    {
      static auto get_kdop = []( const auto &_a ) { return element_traits< Element >::get_kdop( _a ); };
      auto kdop_range = make_range( make_transform_iterator( _span.begin(), get_kdop ),
                                    make_transform_iterator( _span.end(), get_kdop ) );

      m_kdop = KDop::from_kdops( kdop_range.begin(), kdop_range.end() );

      static auto get_centroid = []( const auto &_a ) { return element_traits< Element >::get_centroid( _a ); };
      auto centroid_range = make_range( make_transform_iterator( _span.begin(), get_centroid ),
                                        make_transform_iterator( _span.end(), get_centroid ) );

      m_centroid = std::accumulate( centroid_range.begin(), centroid_range.end(), m::vec3< float_type >() )
                  / static_cast< float_type >( centroid_range.size() );
    }

    index_type  m_global_id = static_cast< index_type >( -1 );
    std::size_t m_size = 0;
    kdop_type   m_kdop;
    m::vec3< float_type > m_centroid = m::vec3< float_type >::zeros();
  };

  template< typename KDop > KOKKOS_INLINE_FUNCTION decltype( auto ) get_entity_kdop( const patch< KDop > &_patch )
  {
    return _patch.kdop();
  }

  template< typename KDop > KOKKOS_INLINE_FUNCTION decltype( auto ) get_entity_global_id( const patch< KDop > &_patch )
  {
    return _patch.global_id();
  }

  template< typename KDop > KOKKOS_INLINE_FUNCTION decltype( auto ) get_entity_centroid( const patch< KDop > &_patch )
  {
    return _patch.centroid();
  }

}

#endif  // INC_BVH_PATCH_HPP
