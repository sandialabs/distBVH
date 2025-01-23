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
#ifndef INC_BVH_COLLISION_QUERY_HPP
#define INC_BVH_COLLISION_QUERY_HPP

#include <vector>
#include <functional>
#include <cstring>
#include "traits.hpp"
#include "util/span.hpp"
#include "patch.hpp"

namespace bvh
{
  template< typename LeftIndex, typename RightIndex >
  using collision_query_func = std::function< void( LeftIndex, RightIndex ) >;

  class collision_object;

  template< typename T >
  struct broadphase_collision
  {
    broadphase_collision( collision_object &_object, const patch<> &_meta, std::size_t _patch_id,
                          bvh::host_view< T * > _elements )
      : object( _object ),
        meta( _meta ),
        patch_id( _patch_id ),
        elements( _elements )
    {}

    collision_object &object;
    patch<> meta;
    std::size_t patch_id;
    bvh::host_view< T * > elements;
  };

  class narrowphase_result
  {
  public:

    narrowphase_result()
        : m_stride( 0 ), m_num_elements( 0 )
    {}

    explicit narrowphase_result( std::size_t _stride )
        : m_stride( _stride ), m_num_elements( 0 )
    {

    }

    void append_data( void *_data, std::size_t _num_elements )
    {
      m_num_elements += _num_elements;
      m_data.insert( m_data.end(),
                    static_cast< std::byte * >( _data ),
                    static_cast< std::byte * >( _data ) + _num_elements * m_stride );
    }

    void set_data( void *_data, std::size_t _num_elements )
    {
      m_num_elements = _num_elements;
      m_data.resize( m_num_elements * m_stride );
      std::memcpy( m_data.data(), _data, m_data.size() );
    }

    void *allocate( std::size_t _n )
    {
      m_num_elements += _n;
      auto iter = m_data.insert( m_data.end(), _n * m_stride, static_cast< std::byte >( 0x00 ) );
      return &( *iter );
    }

    void *at( std::size_t _i )
    {
      return &m_data[_i * m_stride];
    }

    const void *at( std::size_t _i ) const
    {
      return &m_data[_i * m_stride];
    }

    void *data() { return m_data.data(); }

    void reserve( std::size_t _n )
    {
      m_data.reserve( _n * m_stride );
    }

    std::size_t stride() const noexcept { return m_stride; }
    const std::vector< std::byte > &byte_buffer() const noexcept { return m_data; }
    std::size_t size() const noexcept { return m_num_elements; }

  private:

    std::vector< std::byte > m_data;
    std::size_t m_stride;
    std::size_t m_num_elements;
  };

  template< typename T >
  class typed_narrowphase_result : public narrowphase_result
  {
  public:

    explicit typed_narrowphase_result()
        : narrowphase_result( sizeof( T ) )
    {
    }

    template< typename... Args >
    T &emplace_back( Args &&... _args )
    {
      return *( new ( allocate( 1 ) ) T( std::forward< Args >( _args )... ) );
    }

    T &operator[]( std::size_t _i )
    {
      return *static_cast< T * >( at( _i ) );
    }

    const T &operator[]( std::size_t _i ) const
    {
      return *static_cast< const T * >( at( _i ) );
    }
  };

  struct narrowphase_result_pair
  {
    narrowphase_result a;
    narrowphase_result b;
  };


  namespace detail
  {
    template< typename TreeType, typename ContactEntity, typename F >
    void query_node_impl( const typename TreeType::node_type *_node, const ContactEntity &_ent, F &&_fun,
        span< const typename TreeType::value_type > _leafs )
    {
      auto &&kdop = element_traits< ContactEntity >::get_kdop( _ent );
      auto &&idx = element_traits< ContactEntity >::get_global_id( _ent );

      if ( !_node || !overlap( kdop, _node->kdop() ) )
      {
        return;
      }

      if ( _node->is_leaf() )
      {
        for ( std::size_t i = _node->get_patch()[0]; i < _node->get_patch()[1]; ++i )
          std::forward< F >( _fun )( idx, _leafs[i].global_id() );
      } else {
        query_node_impl< TreeType >( _node->left(), _ent, std::forward< F >( _fun ), _leafs );
        query_node_impl< TreeType >( _node->right(), _ent, std::forward< F >( _fun ), _leafs );
      }
    }

    template< typename TreeType, typename ContactEntity, typename F >
    void query_node_local_impl( const typename TreeType::node_type *_node, const ContactEntity &_ent, F &&_fun,
                          span< const typename TreeType::value_type > _leafs )
    {
      auto &&kdop = element_traits< ContactEntity >::get_kdop( _ent );
      auto &&idx = element_traits< ContactEntity >::get_global_id( _ent );

      if ( !_node || !overlap( kdop, _node->kdop() ) )
        return;

      if ( _node->is_leaf() )
      {
        for ( std::size_t i = _node->get_patch()[0]; i < _node->get_patch()[1]; ++i )
          std::forward< F >( _fun )( _leafs[i].local_index() );
      } else {
        query_node_local_impl< TreeType >( _node->left(), _ent, std::forward< F >( _fun ), _leafs );
        query_node_local_impl< TreeType >( _node->right(), _ent, std::forward< F >( _fun ), _leafs );
      }
    }
  };

  template< typename TreeType, typename ContactEntity = typename TreeType::value_type, typename F >
  void query_tree( const TreeType &_tree, const ContactEntity &_ent, F &&_fun )
  {
    detail::query_node_impl< TreeType >( _tree.root(), _ent, std::forward< F >( _fun ), _tree.leafs() );
  }

  template< typename TreeType, typename ContactEntity = typename TreeType::value_type, typename F >
  void query_tree_local( const TreeType &_tree, const ContactEntity &_ent, F &&_fun )
  {
    detail::query_node_local_impl< TreeType >( _tree.root(), _ent, std::forward< F >( _fun ), _tree.leafs() );
  }

  template< typename IndexType >
  struct collision_query_result
  {
    using index_type = IndexType;
    using value_type = std::pair< IndexType, IndexType >;
    using container_type = std::vector< value_type >;

    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    collision_query_result() = default;

    collision_query_result( iterator _b, iterator _e )
      : pairs{ _b, _e }
    {}

    typename container_type::size_type size() const noexcept
    { return pairs.size(); }

    iterator begin() { return pairs.begin(); }
    const_iterator begin() const { return pairs.begin(); }
    const_iterator cbegin() const { return pairs.cbegin(); }

    iterator end() { return pairs.end(); }
    const_iterator end() const { return pairs.end(); }
    const_iterator cend() const { return pairs.cend(); }

    container_type pairs;
  };

  template< typename TreeType >
  struct query_list_functor
  {
    using index_type = typename TreeType::index_type;

    collision_query_result< index_type > results;

    void operator()( index_type _left, index_type _right )
    {
      results.pairs.emplace_back( _left, _right );
    }
  };

  namespace detail
  {
    template< typename TreeType, typename ResultsType >
    typename TreeType::collision_query_result_type
    unroll_results( const ResultsType &_results )
    {
      typename TreeType::collision_query_result_type ret;
      std::size_t                                    reserve_size = 0;
      for ( auto                                     &&pc : _results )
      {
        reserve_size += pc.first.entities.size() * pc.second.entities.size();
      }

      ret.pairs.reserve( reserve_size );

      for ( auto &&pc : _results )
      {
        for ( auto &&entity_i : pc.first.entities )
        {
          for ( auto &&entity_j : pc.second.entities )
          {
            ret.pairs.emplace_back( entity_i.index, entity_j.index );
          }
        }
      }

      return ret;
    }

    template< typename NodeType, typename LeftLeafs, typename RightLeafs, typename OutputIterator >
    void
    get_overlapping_indices( const NodeType *_left,
                             const NodeType *_right,
                             const LeftLeafs &_left_leafs,
                             const RightLeafs &_right_leafs,
                             OutputIterator _iter )
    {
      // Both nodes need to exist and overlap in order to continue traversal
      if ( !_left || !_right || !overlap( _left->kdop(), _right->kdop() ) )
        return;

      if ( _left->is_leaf() )
      {
        // Both are leaves, add all patch indices
        if ( _right->is_leaf() )
        {
          for ( std::size_t i = _left->get_patch()[0]; i < _left->get_patch()[1]; ++i )
            for ( std::size_t j = _right->get_patch()[0]; j < _right->get_patch()[1]; ++j )
              *_iter++ = std::make_pair( _right_leafs[i].global_id(), _left_leafs[j].global_id() );
        } else {
          // Only left is leaf, recurse on right
          get_overlapping_indices< NodeType >( _left, _right->left(), _left_leafs, _right_leafs, _iter );
          get_overlapping_indices< NodeType >( _left, _right->right(), _left_leafs, _right_leafs, _iter );
        }
      } else if ( _right->is_leaf() ) {
        // Only right is leaf, recurse on left
        get_overlapping_indices< NodeType >( _left->left(), _right, _left_leafs, _right_leafs, _iter );
        get_overlapping_indices< NodeType >( _left->right(), _right, _left_leafs, _right_leafs, _iter );
      } else {
        // Neither are leaves, recurse on both sides
        get_overlapping_indices< NodeType >( _left->left(), _right->left(), _left_leafs, _right_leafs, _iter );
        get_overlapping_indices< NodeType >( _left->left(), _right->right(), _left_leafs, _right_leafs, _iter );
        get_overlapping_indices< NodeType >( _left->right(), _right->left(), _left_leafs, _right_leafs, _iter );
        get_overlapping_indices< NodeType >( _left->right(), _right->right(), _left_leafs, _right_leafs, _iter );
      }
    };

    template< typename NodeType, typename Leafs, typename OutputIterator >
    void
    get_self_colliding_indices( const NodeType *_node,
                                const Leafs &_leafs,
                                OutputIterator _iter )
    {
      if ( !_node )
        return;

      if ( _node->is_leaf() )
      {
        // Add all combinations of pairs
        if ( !_node->empty() )
        {
          for ( std::size_t i = _node->get_patch()[0]; i < _node->get_patch()[1] - 1; ++i )
            for ( std::size_t j = i; j < _node->get_patch()[1]; ++j )
              *_iter++ = std::make_pair( _leafs[i].global_id(), _leafs[j].global_id() );
        }
      } else {
        if ( _node->has_left() )
          get_self_colliding_indices< NodeType >( _node->left(), _leafs, _iter );
        if ( _node->has_right() )
          get_self_colliding_indices< NodeType >( _node->right(), _leafs, _iter );
        if ( _node->has_left() && _node->has_right() )
          get_overlapping_indices< NodeType >( _node->left(), _node->right(), _leafs, _leafs, _iter );
      }
    };
  }

  template< typename TreeType >
  typename TreeType::collision_query_result_type
  self_collision_set( const TreeType &_tree )
  {
    typename TreeType::collision_query_result_type ret;
    detail::get_self_colliding_indices< typename TreeType::node_type >( _tree.root(), _tree.leafs(), std::back_inserter( ret.pairs ) );

    return ret;
  }

  template< typename TreeType >
  typename TreeType::collision_query_result_type
  potential_collision_set( const TreeType &_lhs, const TreeType &_rhs )
  {
    typename TreeType::collision_query_result_type ret;
    detail::get_overlapping_indices< typename TreeType::node_type >( _lhs.root(), _rhs.root(), _lhs.leafs(), _rhs.leafs(),
      std::back_inserter( ret.pairs ) );

    return ret;
  }

#if 0
  namespace detail
  {

    template< typename TreeType, typename OutputIterator >
    void
    get_node_overlapping_indices( const typename TreeType::kdop_type &_kdop,
                             const typename TreeType::index_type _local_index,
                             const typename TreeType::node_type *_node,
                             const dynarray< typename TreeType::value_type > &_leafs,
                             OutputIterator _iter )
    {
      if ( !_node || !overlap( _kdop, _node->kdop() ) )
        return;

      if ( _node->is_leaf() )
      {
        for ( std::size_t i = _node->get_patch()[0]; i < _node->get_patch()[1]; ++i )
          *_iter++ = std::make_pair( _local_index, _leafs[i].global_id );
      }
      else
      {
        get_node_overlapping_indices< TreeType, OutputIterator >( _kdop, _local_index, _node->left(), _leafs, _iter );
        get_node_overlapping_indices< TreeType, OutputIterator >( _kdop, _local_index, _node->right(), _leafs, _iter );
      }
    }

    template< typename TreeType, typename OutputIterator >
    void
    get_tree_overlapping_indices( const typename TreeType::kdop_type &_kdop,
                                  const typename TreeType::index_type _local_index,
                                  const TreeType &_tree,
                                  OutputIterator _iter)
    {
      if ( _tree.root() )
        get_node_overlapping_indices< TreeType, OutputIterator >( _kdop, _local_index, _tree.root(), _tree.leafs(), _iter );
    }
  }
#endif
}

#endif  // INC_BVH_COLLISION_QUERY_HPP
