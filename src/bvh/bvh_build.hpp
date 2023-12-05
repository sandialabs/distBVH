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
#ifndef INC_BVH_BVH_BUILD_HPP
#define INC_BVH_BVH_BUILD_HPP

#include <memory>
#include <algorithm>
#include "iterators/transform_iterator.hpp"
#include "iterators/zip_iterator.hpp"
#include "range.hpp"
#include "hash.hpp"
#include "split/split.hpp"
#include "util/bits.hpp"
#include "treelet.hpp"
#include "split/mean.hpp"

namespace bvh
{
  /**
   *  A top-down build policy for bvh_tree.
   *
   *  \tparam KDop              The k-DOP type.
   *  \tparam SplittingPolicy   A policy for how a node should be split into children.
   */
  template< typename SplittingMethod >
  struct top_down_builder
  {

    template< typename T, typename KDop, typename NodeData >
    class builder
    {
    public:

      builder( span< const T > _elements,
               dynarray< bvh_node< T, KDop, NodeData > > &_nodes )
               : m_nodes( _nodes )
      {
        auto kd = KDop( make_transform_iterator( _elements.begin(), element_traits< T >::get_kdop ),
                        make_transform_iterator( _elements.end(), element_traits< T >::get_kdop ) );

        // Build the root (always index 0)
        _nodes.emplace_back( std::move( kd ), 0 );
      }

      /**
       *  Construct the tree using the bvh_top_down_builder build policy
       *
       *  \tparam InputIterator   The iterator type for collision entities.
       */
      void build( std::size_t _node_index,
                         T *_entities,
                         std::size_t _entity_begin, std::size_t _entity_end,
                         std::size_t _entities_per_leaf )
      {
        using kdop_type = KDop;

        m_nodes[_node_index].set_patch( _entity_begin, _entity_end );

        if (( _entity_end - _entity_begin ) > _entities_per_leaf )
        {
          int split_axis = m_nodes[_node_index].kdop().longest_axis();

          const auto begin = _entities + _entity_begin;
          const auto end   = _entities + _entity_end;

          auto split        = split_in_place< SplittingMethod >( make_range( begin, end ), split_axis );
          auto split_offset = std::distance( begin, split );

          auto get_bounds = []( const auto &_a ) { return _a.kdop(); };

          const bool have_left  = ( begin != split );
          const bool have_right = ( split != end );

          // Create children first
          if ( have_left )
          {
            std::ptrdiff_t left          = m_nodes.size() - _node_index;
            std::ptrdiff_t parent_offset = -left;
            auto           left_kdop     = kdop_type{ make_transform_iterator( begin, get_bounds ),
                                                      make_transform_iterator( split, get_bounds ) };
            m_nodes.emplace_back( std::move( left_kdop ), parent_offset );
            m_nodes[_node_index].set_child_offset( 0, left );

            // Recurse down the left tree for efficient pre-order traversal
            build( _node_index + left, _entities, _entity_begin, _entity_begin + split_offset,
                   _entities_per_leaf );
          }

          if ( have_right )
          {
            std::ptrdiff_t right         = m_nodes.size() - _node_index;
            std::ptrdiff_t parent_offset = -right;
            auto           right_kdop    = kdop_type{ make_transform_iterator( split, get_bounds ),
                                                      make_transform_iterator( end, get_bounds ) };
            m_nodes.emplace_back( std::move( right_kdop ), parent_offset );
            m_nodes[_node_index].set_child_offset( 1, right );

            // Recurse down the right tree for efficient pre-order traversal
            build( _node_index + right, _entities, _entity_begin + split_offset, _entity_end,
                   _entities_per_leaf );
          }
        }
      }

    private:

      dynarray< bvh_node< T, KDop, NodeData > > &m_nodes;
    };
  };


  /**
   *  A bottom-up build policy
   */
  struct bottom_up_serial_builder
  {

    template< typename T, typename KDop, typename NodeData >
    class builder
    {
    public:

      using treelet_type = treelet< T, KDop, NodeData >;

      builder( span< const T > _elements,
               dynarray< bvh_node< T, KDop, NodeData > > &_nodes )
        : m_nodes( _nodes )
      {
        m_global_bounds = KDop( make_transform_iterator( _elements.begin(), element_traits< T >::get_kdop ),
                        make_transform_iterator( _elements.end(), element_traits< T >::get_kdop ) );
        
        // A single element will cause divide-by-zeros if we don't do this
        m_global_bounds.inflate( m::epsilon );
      }
  
      template< typename Vec >
      m::vec3< std::uint64_t > discretize_coord( Vec &&_coord )
      {
        m::vec3< std::uint64_t > ret;
    
        for ( int i = 0; i < 3; ++i )
        {
          auto inv_fac = typename KDop::arithmetic_type{ 1 } / m_global_bounds.extents[i].length();
          auto diff = _coord[i] - m_global_bounds.extents[i].min;
          // Max in each dimension is the maximum in a 21-bit number (1/3rd of 63 bit morton code)
          ret[i] = static_cast< std::uint64_t >( std::floor( diff * inv_fac * 0x1fffff ) );
        }

        return ret;
      }

      static bool can_merge_adjacent( std::uint64_t _m64l, std::uint64_t _m64r )
      {
        return ( _m64l == 0UL ) || ( clz( _m64l ) <= clz( _m64r ));
      }

      static std::size_t merge_height( std::uint64_t _m1, std::uint64_t _m2 )
      {
        // This is undefined iff _m1 == _m2, so duplicates must be removed before this point
        return static_cast< std::size_t >( bsr( _m1 ^ _m2 ) );
      }

      static void merge_duplicates( dynarray< treelet_type > &_treelets )
      {
        dynarray< treelet_type > new_treelets;
        auto start = _treelets.begin();
        for ( auto iter = _treelets.begin(); iter != _treelets.end(); /* increment done in while loop */ )
        {
          new_treelets.emplace_back();
          auto &new_treelet = new_treelets.back();

          new_treelet.m64 = start->m64;
          new_treelet.next_merge_height = start->next_merge_height;

          do
          {
            new_treelet.leafs.insert( new_treelet.leafs.end(),
                std::make_move_iterator( iter->leafs.begin() ),
                std::make_move_iterator( iter->leafs.end() ) );
            iter->leafs.clear();
            ++iter;
          } while ( ( iter->m64 == start->m64 ) && ( iter != _treelets.end() ) );

          start = iter;

          auto b = top_down_builder< split::mean >::builder< T, KDop, NodeData >( span< const T >( new_treelet.leafs.data(), new_treelet.leafs.size() ),
              new_treelet.nodes );
          b.build( 0, new_treelet.leafs.data(), 0, new_treelet.leafs.size(), 1 );
        }

        std::swap( _treelets, new_treelets );
      }

      /**
       *  Construct the tree using the bottom_up_serial_builder build policy
       *
       *  \tparam InputIterator   The iterator type for collision entities.
       */
      static void build_treelets( dynarray< treelet_type > &_treelets, std::size_t _min_height, std::size_t _max_height = 64UL )
      {
        dynarray< treelet_type > new_treelets;
        for ( std::size_t h = _min_height; h < _max_height; ++h )
        {
          if ( _treelets.size() == 1 )
            return;

          new_treelets.clear();

          for ( std::size_t i = 0; i < _treelets.size(); ++i )
          {
            if ( _treelets[i].next_merge_height == h )
            {
              _treelets[i + 1].merge_left( _treelets[i] );
              new_treelets.emplace_back( std::move( _treelets[i + 1] ) );
            } else if ( ( i == 0 ) || ( _treelets[i - 1].next_merge_height != h ) ) {
              new_treelets.emplace_back( std::move( _treelets[i] ) );
            }
          }

          std::swap( _treelets, new_treelets );
        }
      }

      /**
       *  Construct the tree using the bottom_up_serial_builder build policy
       *
       *  \tparam InputIterator   The iterator type for collision entities.
       */
      void build( std::size_t _node_index,
                  T *_entities,
                  std::size_t _entity_begin, std::size_t _entity_end,
                  std::size_t _entities_per_leaf )
      {

        dynarray< treelet_type > treelets;
        treelets.reserve( _entity_end - _entity_begin );

        std::transform( _entities + _entity_begin, _entities + _entity_end, std::back_inserter( treelets ),
                        [this]( auto &_x ) {
                          auto coord = this->discretize_coord( element_traits< T >::get_centroid( _x ) );
                          auto m64 = morton( coord.x(), coord.y(), coord.z() );
                          return treelet_type{ m64, element_traits< T >::get_kdop( _x ), 0, _x };
                        } );

        std::sort( treelets.begin(), treelets.end(), []( const auto &_left, const auto &_right ){ return _left.m64 < _right.m64; } );

        std::size_t min_height = 0;
        for ( std::size_t i = 0; i < treelets.size() - 1; ++i )
        {
          treelets[i].next_merge_height = merge_height( treelets[i].m64, treelets[i + 1].m64 );
          min_height = std::min( min_height, treelets[i].next_merge_height );
        }

        build_treelets( treelets, min_height );

        m_nodes = std::move( treelets[0].nodes );

        std::move( treelets[0].leafs.begin(), treelets[0].leafs.end(), _entities + _entity_begin );

        std::reverse( m_nodes.begin(), m_nodes.end() );
      }

    private:

      dynarray< bvh_node< T, KDop, NodeData > > &m_nodes;
      KDop m_global_bounds;
    };
  };
}

#endif // INC_BVH_BVH_BUILD_HPP
