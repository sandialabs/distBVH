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
#ifndef INC_BVH_BVH_SERIALIZE_HPP
#define INC_BVH_BVH_SERIALIZE_HPP

#include "../snapshot.hpp"
#include "../iterators/offset_iterator.hpp"
#include "../vt/print.hpp"

namespace bvh
{
  template< typename Serializer,
            typename T,
            typename KDop,
            typename NodeData >
  void serialize( Serializer &_s, const bvh_node< T, KDop, NodeData > &_node )
  {
    _s | _node.m_parent_offset | _node.m_child_offsets | _node.m_kdop | _node.m_entity_offsets;
  }

  template< typename Serializer, typename T, typename KDop, typename NodeData >
  void serialize( Serializer &_s, const bvh_tree< T, KDop, NodeData > &_tree )
  {
    _s | _tree.m_leafs | _tree.m_nodes;
  }

  template< typename Serializer, typename IndexType >
  void serialize( Serializer &_s, const collision_query_result< IndexType > &_query_results )
  {
    _s | _query_results.pairs;
  }

}


namespace checkpoint
{
  template< typename T >
  struct isByteCopyable< bvh::dop_6< T > > : std::true_type {};

  template< typename T >
  struct isByteCopyable< bvh::dop_18< T > > : std::true_type {};

  template< typename T >
  struct isByteCopyable< bvh::dop_26< T > > : std::true_type {};

  template< typename T, unsigned N >
  struct isByteCopyable< bvh::m::vec< T, N > > : std::true_type {};
}

#endif  // INC_BVH_BVH_SERIALIZE_HPP
