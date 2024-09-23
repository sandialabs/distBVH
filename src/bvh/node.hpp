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
#ifndef INC_BVH_NODE_HPP
#define INC_BVH_NODE_HPP

#include <iostream>
#include "kdop.hpp"
#include "range.hpp"
#include "util/container.hpp"
#include "util/span.hpp"

namespace bvh
{

  template< typename T, typename KDop, typename NodeData >
  class bvh_node;

  template< typename S,
            typename U,
            typename K,
            typename N >
  void serialize( S &_s, const bvh_node< U, K, N > &_node );

  /**
   *  Node in a bvh_tree. This class is byte-serializable (i.e. can be byte-copied over the network or onto disk as
   *  long as the types it stores are byte-serializable).
   *  The type of element, the \f$k\f$-DOP, and additional user data can all be configured by template parameters.
   *
   *  \tparam T         the type of element in the node conforming to `ContactEntity`
   *  \tparam KDop      the type of \f$k\f$-DOP to use in the hierarchy
   *  \tparam NodeData  any additional data to be stored in the node (can be `void` for no additional data)
   */
  template< typename T, typename KDop, typename NodeData >
  class bvh_node
  {
  public:

    using value_type = T;       ///< The `ContactEntity`-conforming type stored in the node
    using kdop_type = KDop;     ///< The type of \f$k\f$-DOP used as bounding volumes.
    using data_type = NodeData; ///< The type of additional user data, `void` if no data specified
  
    using size_type = std::size_t;  ///< The size used for representing distances and depths in the node
    
    // TODO: remove default constructor
    bvh_node() : m_parent_offset( 0 ), m_child_offsets{}, m_entity_offsets{} {}
  
    /**
     *  Get the parent of the node.
     *
     *  \return the parent of the node or `this` if this is the root node
     */
    bvh_node *parent() noexcept { return this + m_parent_offset; }
  
    /**
     *  Get the parent of the node.
     *
     *  \return the parent of the node or `this` if this is the root node
     */
    const bvh_node *parent() const noexcept { return this + m_parent_offset; }
  
    /**
     *  Get the left child of the node.
     *
     *  \return the left child of the node or `this` if there is no left child
     */
    bvh_node *left() noexcept { return this + m_child_offsets[0]; }
  
    /**
     *  Get the left child of the node.
     *
     *  \return the left child of the node or `this` if there is no left child
     */
    const bvh_node *left() const noexcept { return this + m_child_offsets[0]; }
  
    /**
     *  Get the right child of the node.
     *
     *  \return the right child of the node or `this` if there is no right child
     */
    bvh_node *right() noexcept { return this + m_child_offsets[1]; }
  
    /**
     *  Get the right child of the node.
     *
     *  \return the right child of the node or `this` if there is no right child
     */
    const bvh_node *right() const noexcept { return this + m_child_offsets[1]; }

    /**
     * Get the child that has the specified index. This function does not check that `_child < 2`. Invalid numbers
     * will cause array overruns and undefined behavior.
     *
     * \param _child    the index of the child
     * \return          the node at that index or `this` if there is no child at that index
     */
    const bvh_node *get_child( int _child ) const noexcept { return this + m_child_offsets[_child]; }

    /**
     * Set the offset amount for a child. This function does not check that `_child < 2`. Invalid numbers
     * will cause array overruns and undefined behavior.
     *
     * \param _child    the index of the child
     * \param _offset   the offset, i.e. where in the array the child is
     */
    void set_child_offset( int _child, std::ptrdiff_t _offset ) noexcept { m_child_offsets[_child] = _offset; }

    /**
     * Returns the offset of the specified child. Does not convert into a pointer unlike get_child().
     *
     * \param _child    the index of the child
     * \return          the offset of the child or 0 if there is no child at that index
     */
    std::ptrdiff_t get_child_offset( int _child ) const noexcept { return m_child_offsets[_child]; }
    
    /**
     *  Get whether or not the node is a leaf node.
     *
     *  \return whether or not the node is a leaf according to the build policy.
     */
    bool is_leaf() const noexcept
    {
      return !has_left() && !has_right();
    }
  
    /**
     *  Get the depth of the tree at the node.
     *
     *  \return the node depth
     */
    size_type depth() const noexcept
    {
      return has_parent() ? parent()->depth() : 1;
    }

    /**
     * Get whether the node has a left child.
     *
     * \return  whether the node has a left child
     */
    bool has_left() const noexcept { return m_child_offsets[0] != 0; }

    /**
     * Get whether the node has a right child.
     *
     * \return  whether the node has a right child
     */
    bool has_right() const noexcept { return m_child_offsets[1] != 0; }

    /**
     * Get whether the node has a parent.
     *
     * \return  whether the node has a parent
     */
    bool has_parent() const noexcept { return m_parent_offset != 0; }

    /**
     * Get the maximum path length that can be traversed on either side of the node.
     *
     * \return  the maximum path length of the subtree
     */
    size_type max_depth() const noexcept
    {
      return is_leaf() ? 1 : std::max( left()->depth(), right()->depth() ) + 1;
    }
  
    /**
     *  Get the level of the tree at the node.
     *
     *  \return the node level
     */
    size_type level() const noexcept
    {
      size_type level = 0;
      auto p = this;
      while ( p->has_parent() )
      {
        ++level;
        p = p->parent();
      }
    
      return level;
    }

    /**
     * Set the parent offset. A value of `0` orphans the node and makes it a root.
     *
     * \param _offset   the offset of the parent
     */
    void set_parent_offset( std::ptrdiff_t _offset )
    {
      m_parent_offset = _offset;
    }
    
    /**
     *  Construct a node by moving in a k_DOP and giving a parent node.
     *
     *  \param _kdop    temporary or moved-from k-DOP to use for the tree
     *  \param _parent  the parent node. Can be `nullptr` for root nodes.
     */
    bvh_node( KDop &&_kdop, std::ptrdiff_t _parent_offset )
      : m_parent_offset( _parent_offset ),
        m_child_offsets{},
        m_kdop( std::move( _kdop ) ),
        m_entity_offsets{}
    {
    }
    
    /**
     *  Get the number of elements in this node.
     *
     *  \return the number of elements in the bounding volume of the node.
     */
    size_type count() const noexcept
    {
      if ( this->is_leaf() )
        return m_entity_offsets[1] - m_entity_offsets[0];
  
      return this->left()->count() + this->right()->count();
    }
    
    /**
     *  Get the bounding volume of the node
     *
     *  \return the k-DOP bounding the node
     */
    const kdop_type &kdop() const noexcept { return m_kdop; }

    /**
     * Get the offsets of the entities into the entity array that this node represents. These offsets are returned
     * as an array of two values; the first is the start of the range of entities and the second is the end of the range
     *
     * \return  the offsets
     */
    array< std::size_t, 2 > &get_patch() noexcept { return m_entity_offsets; }

    const array< std::size_t, 2 > &get_patch() const noexcept { return m_entity_offsets; }

    /**
     * Set the offsets into the entity array that this node represents. The first offset is treated as a beginning
     * of the range of entitires and the second offset is the end.
     *
     * \param _first_offset     the beginning of the entity range
     * \param _second_offset    the end of the entity range
     */
    void set_patch( std::size_t _first_offset, std::size_t _second_offset ) noexcept
    {
      m_entity_offsets[0] = _first_offset;
      m_entity_offsets[1] = _second_offset;
    }

    /**
     * Get the number of entities represented by this node.
     *
     * \return  the number of entities that the node represents
     */
    std::size_t num_patch_elements() const noexcept { return m_entity_offsets[1] - m_entity_offsets[0]; }

    /**
     * Get whether or not the node represents any entities.
     *
     * \return  whether the node is empty
     */
    bool empty() const noexcept { return num_patch_elements() == 0; }

    /**
     * Tests whether the node is equivalent to another node. Nodes are equal iff their \f$k\f$-DOPs are equal,
     * iff they share the same parent, iff they share the same children, and iff they share the same entities.
     *
     * \param _lhs  the first node to compare
     * \param _rhs  the second node to compare
     * \return      whether the nodes are equivalent
     */
    friend bool operator==( const bvh_node &_lhs, const bvh_node &_rhs )
    {
      return ( _lhs.m_kdop == _rhs.m_kdop )
        && ( _lhs.m_parent_offset == _rhs.m_parent_offset )
        && ( _lhs.m_child_offsets == _rhs.m_child_offsets )
        && ( _lhs.m_entity_offsets == _rhs.m_entity_offsets );
    }
    
  private:

    template< typename S,
              typename U,
              typename K,
              typename N >
    friend void serialize( S &_s, const bvh_node< U, K, N > &_node );
  
    std::ptrdiff_t m_parent_offset;
    array< std::ptrdiff_t, 2 > m_child_offsets;

    KDop m_kdop;
    array< std::size_t, 2 > m_entity_offsets;
  };
  
  namespace detail
  {
    template< typename T, typename KDop, typename NodeData >
    std::ostream &
    dump_node_impl( std::ostream &_strm, const bvh_node< T, KDop, NodeData > &_node, std::vector< bool > &_last_stack,
                    const dynarray< T > &_leafs)
    {
      bool last = _last_stack.back();
      
      for ( std::size_t i = 0; i < _last_stack.size() - 1; ++i )
      {
        _strm << ( ( !_last_stack[i] ) ? "\u2502" : " " ) << ' ';
      }
  
      _strm << ( last ? "\u2514 " : "\u251c " );
      
      auto level = _node.level();
      if ( _node.is_leaf() )
      {
        _strm << "level: " << level << " leaf id:";
        for ( std::size_t i = _node.get_patch()[0]; i < _node.get_patch()[1]; ++i )
          _strm << " " << _leafs[i].global_id();
        _strm << ' ' << _node.kdop() << '\n';
      } else
      {
        _strm << "level: " << level << ' ' << _node.kdop() << '\n';
      }
      if ( _node.has_left() )
      {
        _last_stack.push_back( !_node.right() );
        dump_node_impl( _strm, *_node.left(), _last_stack, _leafs );
      }
    
      if ( _node.has_right() )
      {
        _last_stack.push_back( true );
        dump_node_impl( _strm, *_node.right(), _last_stack, _leafs );
      }
      
      _last_stack.pop_back();
    
      return _strm;
    }
  }

  /**
   * Write a node to an output stream, using the specified leafs as the array of contact entities.
   *
   * \tparam T          the type stored in the tree that conforms to `ContactEntity`
   * \param _strm       the output stream
   * \param _node       the node to output
   * \param _leafs      the array of contact entities
   * \return            the modified stream
   */
  template< typename T, typename KDop, typename NodeData >
  std::ostream &
  dump_node( std::ostream &_strm, const bvh_node< T, KDop, NodeData > &_node,
    const dynarray< T > &_leafs )
  {
    std::vector< bool > last_stack;
    last_stack.push_back( true );
    
    return detail::dump_node_impl( _strm, _node, last_stack, _leafs );
  }
}

#endif  // INC_BVH_NODE_HPP
