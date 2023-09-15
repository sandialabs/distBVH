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
#ifndef INC_BVH_TREELET_HPP
#define INC_BVH_TREELET_HPP

#include <cstdint>
#include <utility>
#include "node.hpp"

namespace bvh
{
  template< typename T, typename KDop, typename NodeData >
  struct treelet
  {
    treelet() = default;

    treelet( std::uint64_t _m64, KDop &&_kdop, std::ptrdiff_t _parent, T _leaf )
        : m64( _m64 ), next_merge_height( 64 )
    {
      nodes.emplace_back( std::move( _kdop ), _parent );
      leafs.emplace_back( std::move( _leaf ) );
      nodes.back().set_patch( 0, 1 );
    }

    ~treelet() = default;

    treelet( const treelet & ) = default;
    treelet( treelet &&_other ) noexcept = default;

    treelet &operator=( const treelet & ) = default;
    treelet &operator=( treelet &&_other ) noexcept = default;

    bool empty() const noexcept { return nodes.empty(); }

    void merge_left( const treelet &_left )
    {
      auto right = nodes.size() - 1;
      nodes.insert( nodes.end(), _left.nodes.begin(), _left.nodes.end() );
      auto left = nodes.size() - 1;

      for ( std::size_t i = right + 1; i < nodes.size(); ++i )
      {
        nodes[i].get_patch()[0] += leafs.size();
        nodes[i].get_patch()[1] += leafs.size();
      }

      leafs.insert( leafs.end(), _left.leafs.begin(), _left.leafs.end() );

      nodes.emplace_back( merge( nodes[left].kdop(), nodes[right].kdop() ), 0 );
      nodes.back().set_child_offset( 0, 1 );
      nodes.back().set_child_offset( 1, nodes.size() - 1 - right );
      nodes.back().set_patch( 0, leafs.size() );
      nodes[right].set_parent_offset( static_cast< std::ptrdiff_t >( right ) - static_cast< std::ptrdiff_t >( nodes.size() - 1 ) );
      nodes[left].set_parent_offset( -1 );
    }

    template< typename Serializer >
    friend void serialize( Serializer &_s, const treelet &_treelet )
    {
      _s | _treelet.m64 | _treelet.nodes | _treelet.leafs | _treelet.next_merge_height;
    }

    std::uint64_t m64 = 0UL;
    dynarray< bvh_node< T, KDop, NodeData > > nodes;
    dynarray< T > leafs;
    std::size_t next_merge_height = 0UL;
  };
}

#endif  // INC_BVH_TREELET_HPP
