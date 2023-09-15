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
#ifndef INC_BVH_BTREE_HPP
#define INC_BVH_BTREE_HPP

#include <array>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include "node.hpp"
#include "bvh_build.hpp"
#include "splitting.hpp"
#include "tree_iterator.hpp"
#include "iterators/transform_iterator.hpp"
#include "iterators/zip_iterator.hpp"
#include "iterators/level_iterator.hpp"
#include "collision_query.hpp"
#include "exceptions/invalid_tree_exception.hpp"
#include "patch.hpp"
#include "snapshot.hpp"
#include "treelet.hpp"

namespace bvh
{
  template< typename T, typename KDop, typename NodeData >
  class bvh_tree;

  template< typename Serializer,
            typename U,
            typename K,
            typename N >
  void serialize( Serializer &_s, const bvh_tree< U, K, N > &_tree );

  /**
   *  Binary tree representing a bounding volume hierarchy using \f$k\f$-DOP volumes.
   *
   *  The tree is potentially unbalanced. \ref bvh_tree provides
   *  methods for fast \f$ O\left(\log n\right) \f$ determination of collision with a \f$k\f$-DOP.
   *
   *  \tparam T                 the type of data stored in the tree; must conform to `ContactEntity`
   *  \tparam KDop              the type of \f$k\f$-DOP used (see \ref bvh::kdop_base)
   *  \tparam NodeData          the type of user data stored in the node alongside T, can be `void` for no data
   */
  template< typename T, typename KDop, typename NodeData >
  class bvh_tree
  {
  public:
    
    using node_type = bvh_node< T, KDop, NodeData >;  ///< The type of node used in the tree.
    
    using size_type = std::size_t;                    ///< The size used for representing distances and depths in the tree.
    using index_type = size_type;                     ///< The type used for referencing elements in the tree.
    using kdop_type = typename node_type::kdop_type;  ///< The type of \f$k\f$-DOP used for bounding volume hierarchies.
    using arithmetic_type = typename kdop_type::arithmetic_type;  ///< The arithmetic type (e.g. `float`, `double` used in the \f$k\f$-DOP extemts.
    using collision_query_result_type = collision_query_result< index_type >; ///< The result type of collision queries
    using value_type = T;                             ///< The `ContactEntity`-conforming type stored in the tree

    /**
     * Construct an empty tree.
     */
    bvh_tree() = default;

    /**
     * Explicitly convert a \ref treelet to a \ref bvh_tree.
     *
     * This is an \f$O(n)\f$ operation, since in the internal representation a \ref treelet has reversed node storage.
     *
     * \param _treelet  the treelet to construct the tree from
     */
    explicit bvh_tree( treelet< T, KDop, NodeData > &&_treelet )
    {
      m_nodes = std::move( _treelet.nodes );
      m_leafs = std::move( _treelet.leafs );

      std::reverse( m_nodes.begin(), m_nodes.end() );
    }

    /**
     * Copy constructor. \f$O(n + m)\f$.
     *
     * \param _other    the tree to copy from
     */
    bvh_tree( const bvh_tree &_other )
      : m_leafs( _other.m_leafs ),
        m_nodes( _other.m_nodes )
    {
    
    }

    /**
     * Copy assignment. \f$O(n + m)\f$.
     *
     * \param _other    the tree to copy from
     * \return          the modified tree
     */
    bvh_tree &operator=( const bvh_tree &_other )
    {
      m_leafs = _other.m_leafs;
      m_nodes = _other.m_nodes;
      
      return *this;
    }

    /**
     * Move constructor. \f$O(1)\f$.
     *
     * \param _other    the tree to move from
     */
    bvh_tree( bvh_tree &&_other ) noexcept
      : m_leafs( std::move( _other.m_leafs ) ),
        m_nodes( std::move( _other.m_nodes ) )
    {
    }

    /**
     * Move assignment. \f$O(1)\f$.
     *
     * \param _other    the tree to move from
     * \return          the modified tree
     */
    bvh_tree &operator=( bvh_tree &&_other ) noexcept
    {
      m_leafs = std::move( _other.m_leafs );
      m_nodes = std::move( _other.m_nodes );
      
      return *this;
    }
  
    /**
     *  The depth of the tree.
     *
     *  An empty tree would be depth 0, and a single root node with no
     *  children would be a depth of 1. \f$O\left(\log n\right)\f$ time.
     *
     *  \return the tree depth
     */
    size_type depth() const noexcept
    {
      return !m_nodes.empty() ? m_nodes[0].max_depth() + 1 : size_type( 0 );
    }

    /**
     * Get the root of the tree.
     *
     * Returns `nullptr` for empty trees.
     *
     * \return  the root of the tree or `nullptr` if the tree is empty
     */
    node_type *root() noexcept { return m_nodes.empty() ? nullptr : &m_nodes[0]; }
    const node_type *root() const noexcept { return m_nodes.empty() ? nullptr : &m_nodes[0]; }
    
    /**
     *  Gets the number of entities in the tree. \f$O\left(1\right)\f$ time.
     *
     *  \return the number of entities in the tree
     */
    size_type count() const noexcept
    {
      return m_leafs.size();
    }

    /**
     * Returns whether the tree is empty. \f$O\left(1\right)\f$ time.
     *
     * \return whether the tree is empty
     */
    bool empty() const noexcept
    {
      return m_nodes.empty();
    }

    /**
     * Gets the bounds of the tree. If the tree is empty, this function will return a bounds object that has a
     * volume of 0. \f$O\left(1\right)\f$ time.
     *
     * \return the bounds of the tree
     */
    kdop_type bounds() const
    {
      if ( !m_nodes.empty() )
        return m_nodes[0]->kdop();
      else
        return kdop_type();
    }

    /**
     * Return the array of `ContactEntity`s the tree represents. \f$O\left(1\right)\f$ time.
     *
     * \return the entities in the tree
     */
    const dynarray< T > &leafs() const noexcept { return m_leafs; }

    /**
     * Return the array of nodes in the tree. The nodes are stored in pre-order traversal order. \f$O\left(1\right)\f$ time.
     *
     * \return the nodes in the tree
     */
    const dynarray< node_type > &nodes() const noexcept { return m_nodes; }

    /**
     * Test equality of two trees. Two trees are equal iff every node in the first tree is equivalent to every node
     * in the second tree and iff every `ContactEntity` in the first tree is equivalent to the corresponding
     * `ContactEntity` in the second tree. \f$O\left(n + m\right)\f$ time.
     *
     * \param _lhs  the first tree
     * \param _rhs  the second tree
     * \return      whether the trees are equal
     */
    friend bool operator==( const bvh_tree &_lhs, const bvh_tree &_rhs )
    {
      return ( _lhs.m_nodes == _rhs.m_nodes ) && ( _lhs.m_leafs == _rhs.m_leafs );
    }

    /**
     * Test whether two trees are not equivalent. Two trees are not equivalent if a node in the first tree does
     * not equal its corresponding node in the second tree, or if a `ContactEntity` in the first tree is not
     * equivalent to its corresponding `ContactEntity` in the second tree. \f$O\left(n + m\right)\f$ time.
     *
     * \param _lhs  the first tree
     * \param _rhs  the second tree
     * \return      whether the trees are not equivalent
     */
    friend bool operator!=( const bvh_tree &_lhs, const bvh_tree &_rhs )
    {
      return !( _lhs == _rhs );
    }

    /**
     * Utility function to ensure that a tree is self-consistent. This checks whether every child's parent is the
     * current node. \f$O\left(n\right)\f$ time.
     *
     * \return   whether the tree is valid
     */
    bool debug_validate() const
    {
      for ( std::size_t i = 0; i < m_nodes.size(); ++i )
      {
        auto &node = m_nodes[i];
        
        // Check child's parent is self
        // child offsets always exist, they just loop back on self
        if ( ( i + node.get_child_offset( 0 ) < m_nodes.size() )
        && ( node.left() != &node ) && ( node.left()->parent() != &node )  )
          return false;
  
        if ( ( i + node.get_child_offset( 1 ) < m_nodes.size() )
        && ( node.right() != &node ) && ( node.right()->parent() != &node )  )
          return false;
      }
      
      return true;
    }
  
    using preorder_iterator = typename dynarray< node_type >::iterator;               ///< An iterator for preorder traversal of a tree
    using const_preorder_iterator = typename dynarray< node_type >::const_iterator;   ///< A const iterator for preorder traversal of a tree
    using leaf_iterator = leaf_iter< node_type * >;                                   ///< An iterator for iterating over the leaves of a tree
    using const_leaf_iterator = leaf_iter< const node_type * >;                       ///< A const iterator for iterating over the leaves of a tree
    using level_iterator = level_iter< node_type * >;                                 ///< An iterator for iterating over a specific level of a tree
    using const_level_iterator = level_iter< const node_type * >;                     ///< A const iterator for iterating over a specific level of a tree
    using max_level_iterator = max_level_iter< node_type * >;                         ///< An iterator for iterating over leafs below a specific level
    using const_max_level_iterator = max_level_iter< const node_type * >;             ///< A const iterator for iterating over leafs below a specific level
  
    /**
     *  Gets the iterator at the beginning of a preorder traversal of the tree.
     *  \return iterator marking the beginning of a preorder traversal.
     */
    preorder_iterator preorder_begin() noexcept { return m_nodes.begin(); }
  
    /**
     *  Gets the iterator at the end of a preorder traversal of the tree.
     *  \return iterator marking the end of a preorder traversal.
     */
    preorder_iterator preorder_end() noexcept { return m_nodes.end(); }
  
    /**
     *  Gets the iterator at the beginning of a preorder traversal of the tree.
     *  \return iterator marking the beginning of a preorder traversal.
     */
    const_preorder_iterator preorder_begin() const noexcept { return m_nodes.begin(); }
  
    /**
     *  Gets the iterator at the end of a preorder traversal of the tree.
     *  \return iterator marking the end of a preorder traversal.
     */
    const_preorder_iterator preorder_end() const noexcept { return m_nodes.end(); }
  
    /**
     *  Gets the const iterator at the beginning of a preorder traversal of the tree.
     *  \return const iterator marking the beginning of a preorder traversal.
     */
    const_preorder_iterator preorder_cbegin() const noexcept { return m_nodes.cbegin(); }
  
    /**
     *  Gets the const iterator at the end of a preorder traversal of the tree.
     *  \return const iterator marking the end of a preorder traversal.
     */
    const_preorder_iterator preorder_cend() const noexcept { return m_nodes.cend(); }
  
    /**
     *  Gets the iterator at the beginning of a leaf traversal of the tree.
     *  \return iterator marking the beginning of a leaf traversal.
     */
    leaf_iterator leaf_begin() noexcept { return m_nodes.empty() ? leaf_iterator{} : leaf_iterator{ &m_nodes[0] }; }
  
    /**
     *  Gets the iterator at the end of a leaf traversal of the tree.
     *  \return iterator marking the end of a leaf traversal.
     */
    leaf_iterator leaf_end() noexcept { return leaf_iterator{}; }
  
    /**
     *  Gets the iterator at the beginning of a leaf traversal of the tree.
     *  \return iterator marking the beginning of a leaf traversal.
     */
    const_leaf_iterator leaf_begin() const noexcept { return m_nodes.empty() ? const_leaf_iterator{} : const_leaf_iterator{ &m_nodes[0] }; }
  
    /**
     *  Gets the iterator at the end of a leaf traversal of the tree.
     *  \return iterator marking the end of a leaf traversal.
     */
    const_leaf_iterator leaf_end() const noexcept { return const_leaf_iterator{}; }
  
    /**
     *  Gets the const iterator at the beginning of a leaf traversal of the tree.
     *  \return const iterator marking the beginning of a leaf traversal.
     */
    const_leaf_iterator leaf_cbegin() const noexcept { return m_nodes.empty() ? const_leaf_iterator{} : const_leaf_iterator{ &m_nodes[0] }; }
  
    /**
     *  Gets the const iterator at the end of a leaf traversal of the tree.
     *  \return const iterator marking the end of a leaf traversal.
     */
    const_leaf_iterator leaf_cend() const noexcept { return const_leaf_iterator{}; }
  
    /**
     *  Gets the iterator at the beginning of a level traversal of the tree.
     *  \return iterator marking the beginning of a level traversal.
     */
    level_iterator level_begin( int _level ) noexcept { return m_nodes.empty() ? level_iterator{ _level } : level_iterator{ _level, &m_nodes[0] }; }
  
    /**
     *  Gets the iterator at the end of a level traversal of the tree.
     *  \return iterator marking the end of a level traversal.
     */
    level_iterator level_end( int _level ) noexcept { return level_iterator{ _level }; }
  
    /**
     *  Gets the iterator at the beginning of a level traversal of the tree.
     *  \return iterator marking the beginning of a level traversal.
     */
    const_level_iterator level_begin( int _level ) const noexcept { return m_nodes.empty() ? const_level_iterator{ _level } : const_level_iterator{ _level, &m_nodes[0] }; }
  
    /**
     *  Gets the iterator at the end of a level traversal of the tree.
     *  \return iterator marking the end of a level traversal.
     */
    const_level_iterator level_end( int _level ) const noexcept { return const_level_iterator{ _level }; }
  
    /**
     *  Gets the const iterator at the beginning of a level traversal of the tree.
     *  \return const iterator marking the beginning of a level traversal.
     */
    const_level_iterator level_cbegin( int _level ) const noexcept { return m_nodes.empty() ? const_level_iterator{ _level } : const_level_iterator{ _level, &m_nodes[0] }; }
  
    /**
     *  Gets the const iterator at the end of a level traversal of the tree.
     *  \return const iterator marking the end of a level traversal.
     */
    const_level_iterator level_cend( int _level ) const noexcept { return const_level_iterator{ _level }; }
  
    /**
     *  Gets the iterator at the beginning of a max level traversal of the tree.
     *  \return iterator marking the beginning of a max level traversal.
     */
    max_level_iterator max_level_begin( int _level ) noexcept { return m_nodes.empty() ? max_level_iterator{ _level } : max_level_iterator{ _level, &m_nodes[0] }; }
  
    /**
     *  Gets the iterator at the end of a max level traversal of the tree.
     *  \return iterator marking the end of a max level traversal.
     */
    max_level_iterator max_level_end( int _level ) noexcept { return max_level_iterator{ _level }; }
  
    /**
     *  Gets the iterator at the beginning of a max level traversal of the tree.
     *  \return iterator marking the beginning of a max level traversal.
     */
    const_max_level_iterator max_level_begin( int _level ) const noexcept { return m_nodes.empty() ? const_max_level_iterator{ _level } : const_max_level_iterator{ _level, &m_nodes[0] }; }
  
    /**
     *  Gets the iterator at the end of a max level traversal of the tree.
     *  \return iterator marking the end of a max level traversal.
     */
    const_max_level_iterator max_level_end( int _level ) const noexcept { return const_max_level_iterator{ _level }; }
  
    /**
     *  Gets the const iterator at the beginning of a max level traversal of the tree.
     *  \return const iterator marking the beginning of a max level traversal.
     */
    const_max_level_iterator max_level_cbegin( int _level ) const noexcept { return m_nodes.empty() ? const_max_level_iterator{ _level } : const_max_level_iterator{ _level, &m_nodes[0] }; }
  
    /**
     *  Gets the const iterator at the end of a max level traversal of the tree.
     *  \return const iterator marking the end of a max level traversal.
     */
    const_max_level_iterator max_level_cend( int _level ) const noexcept { return const_max_level_iterator{ _level }; }


  private:
  
    template< typename TreeBuildPolicy, typename U, typename K, typename N, typename Element >
    friend void rebuild_tree( bvh_tree< U, K, N > &, span< const Element > _elements );
    
    template< typename Serializer,
              typename U,
              typename K,
              typename N >
    friend void serialize( Serializer &_s, const bvh_tree< U, K, N > &_tree );
    
    dynarray< T > m_leafs;
    dynarray< node_type >  m_nodes;
  };

  /**
   * A tree type that uses \f$26\f$-DOPs.
   *
   * \tparam T  the type of `ContactEntity`
   * \tparam N  user-defined data stored in the tree, `void` means no data
   */
  template< typename T, typename N = void >
  using bvh_tree_26d = bvh_tree< T, dop_26d, N >;

  /**
   * Write a tree to a stream.
   *
   * \param _strm   the stream to output to
   * \param _tree   the tree to output
   * \return        the modified stream
   */
  template< typename T, typename KDop, typename NodeData >
  std::ostream &
  dump_tree( std::ostream &_strm, const bvh_tree< T, KDop, NodeData > &_tree )
  {
    _strm << "bvh_tree (" << KDop::k << "-dop):\n";
    if ( !_tree.empty() )
      dump_node( _strm, *_tree.root(), _tree.leafs() );
    
    return _strm;
  }
  
}

#endif  // INC_BVH_BTREE_HPP
