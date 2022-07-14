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
#ifndef INC_BVH_SNAPSHOT_HPP
#define INC_BVH_SNAPSHOT_HPP

#include "math/vec.hpp"
#include "util/attributes.hpp"
#include "util/kokkos.hpp"
#include "traits.hpp"
#include "kdop.hpp"
#include "types.hpp"
#include "contact_entity.hpp"

namespace bvh
{
  namespace detail
  {
    template< typename Entity >
    auto convert_centroid( const Entity &_ent )
    {
      using kdop_type = typename element_traits< Entity >::kdop_type;
      using arithmetic_type = typename kdop_type::arithmetic_type;

      auto &&centroid = element_traits< Entity >::get_centroid( _ent );

      return m::vec3< arithmetic_type >( centroid[0], centroid[1], centroid[2] );
    }
  }

  /**
   * A class for a minimal representation of a contact entity in a bounding volume
   * hierarchy. It contains a reference back to the originating entity, a bounding
   * volume, and a centroid.
   *
   * @tparam IndexType  The type of global index
   * @tparam KdopType   The type of bounding volume
   */
  class entity_snapshot
  {
  public:

    using index_type = std::size_t;
    using kdop_type = bphase_kdop;
    using centroid_type = m::vec3< float_type >;

    KOKKOS_INLINE_FUNCTION
    entity_snapshot( index_type _gid, kdop_type _bounds, centroid_type _centroid, index_type _local_index )
        : m_global_id( _gid ), m_kdop( _bounds ), m_centroid( _centroid ), m_local_index( _local_index )
    {}

    KOKKOS_INLINE_FUNCTION entity_snapshot() = default;
    KOKKOS_INLINE_FUNCTION ~entity_snapshot() = default;

    KOKKOS_INLINE_FUNCTION
    entity_snapshot( const entity_snapshot & ) = default;

    KOKKOS_INLINE_FUNCTION
    entity_snapshot( entity_snapshot && ) = default;

    KOKKOS_INLINE_FUNCTION
    entity_snapshot &operator=( const entity_snapshot & ) = default;
    KOKKOS_INLINE_FUNCTION
    entity_snapshot &operator=( entity_snapshot && ) = default;

    KOKKOS_INLINE_FUNCTION index_type global_id() const noexcept { return m_global_id; }
    KOKKOS_INLINE_FUNCTION kdop_type kdop() const noexcept { return m_kdop; }
    KOKKOS_INLINE_FUNCTION centroid_type centroid() const noexcept { return m_centroid; }

    KOKKOS_INLINE_FUNCTION index_type local_index() const noexcept { return m_local_index; }

  private:

    index_type m_global_id;
    kdop_type m_kdop;
    centroid_type m_centroid;
    index_type m_local_index;

    friend KOKKOS_INLINE_FUNCTION bool operator==( const entity_snapshot &_lhs, const entity_snapshot &_rhs )
    {
      return ( _lhs.m_global_id == _rhs.m_global_id )
             && ( _lhs.m_kdop == _rhs.m_kdop )
             && ( _lhs.m_centroid == _rhs.m_centroid );
    }

    friend KOKKOS_INLINE_FUNCTION bool operator!=( const entity_snapshot &_lhs, const entity_snapshot &_rhs )
    {
      return !( _lhs == _rhs );
    }

    template< typename Serializer >
    friend void serialize( Serializer &_s, const entity_snapshot &_snapshot )
    {
      _s | _snapshot.m_global_id | _snapshot.m_kdop | _snapshot.m_centroid;
    }
  };

  /**
   * Utility function for creating a snapshot from a contact entity. Overload
   * the following functions for a custom entity:
   *
   * get_entity_global_id( const your_entity_type & )
   * get_entity_kdop( const your_entity_type & )
   * get_entity_centroid( const your_entity_type & )
   *
   * @tparam Entity     The contact entity type
   * @param _entity     The contact entity to snapshot
   * @return            The snapshot of the contact entity
   */
  template< typename Entity >
  KOKKOS_INLINE_FUNCTION auto
  make_snapshot( const Entity &_entity, std::size_t _local_index = static_cast< std::size_t >( -1 ) )
  {
    using traits_type = element_traits< Entity >;
    return entity_snapshot{ traits_type::get_global_id( _entity ),
                            traits_type::get_kdop( _entity ),
                            detail::convert_centroid( _entity ),
                            _local_index
    };
  }
}

#endif // INC_BVH_SNAPSHOT_HPP
