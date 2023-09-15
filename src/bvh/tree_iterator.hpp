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
#ifndef INC_BVH_TREE_ITERATOR_HPP
#define INC_BVH_TREE_ITERATOR_HPP

#include <iterator>
#include "range.hpp"

namespace bvh
{
  namespace detail
  {
    template< typename UnderlyingIterator >
    UnderlyingIterator get_next_preorder( UnderlyingIterator _node )
    {
      if ( _node->has_left() )
        return _node->left();
      
      if ( _node->has_right() )
        return _node->right();
      
      // Leaf node
      while ( _node->has_parent() )
      {
        // If we're on the left child and there is a right sibling
        if ( _node == _node->parent()->left() && _node->parent()->has_right() )
          return _node->parent()->right();
        
        _node = _node->parent();
      }
      
      return UnderlyingIterator{};
    }
  }
  
  /**
   *  Iterator for preorder traversal of a binary_tree.
   *
   *  \tparam UnderlyingIterator  the node iterator to use
   */
  template< typename UnderlyingIterator >
  class preorder_iter
  {
  public:
  
    using value_type = typename std::iterator_traits< UnderlyingIterator >::value_type;
    using pointer = typename std::iterator_traits< UnderlyingIterator >::pointer;
    using reference = typename std::iterator_traits< UnderlyingIterator >::reference;
  
    using difference_type = typename std::iterator_traits< UnderlyingIterator >::difference_type;
    using iterator_category = std::forward_iterator_tag;
  
    explicit preorder_iter(  const UnderlyingIterator &_iter = UnderlyingIterator{} )
      : m_node( _iter )
    {
    
    }
    
    preorder_iter( const preorder_iter &_other ) = default;
    preorder_iter &operator=( const preorder_iter &_other ) = default;
    
    ~preorder_iter() = default;
    
    reference operator*()
    {
      return *m_node;
    }
    
    preorder_iter &operator++()
    {
      m_node = detail::get_next_preorder( m_node );
      return *this;
    }
    
    preorder_iter operator++( int )
    {
      auto ret = *this;
      ++ret;
      
      return ret;
    }
    
    friend bool operator==( const preorder_iter &_lhs, const preorder_iter &_rhs )
    {
      return _lhs.m_node == _rhs.m_node;
    }
    
    friend bool operator!=( const preorder_iter &_lhs, const preorder_iter &_rhs )
    {
      return !( _lhs == _rhs );
    }
    
    
  private:
  
    UnderlyingIterator m_node;
  };
  
  
  /**
   *  Iterator for leaf traversal of a binary_tree.
   *
   *  Incrementing this iterator can taking logarithmic time.
   *
   *  \tparam UnderlyingIterator  the node iterator to use
   */
  template< typename UnderlyingIterator >
  class leaf_iter
  {
  public:
  
    using value_type = typename std::iterator_traits< UnderlyingIterator >::value_type;
    using pointer = typename std::iterator_traits< UnderlyingIterator >::pointer;
    using reference = typename std::iterator_traits< UnderlyingIterator >::reference;
  
    using difference_type = typename std::iterator_traits< UnderlyingIterator >::difference_type;
    using iterator_category = std::forward_iterator_tag;
    
    explicit leaf_iter( const UnderlyingIterator &_iter = UnderlyingIterator{} )
      : m_node( _iter )
    {
      if ( m_node != UnderlyingIterator{} )
        while ( m_node && !m_node->is_leaf() )
          m_node = detail::get_next_preorder( m_node );
    }
    
    leaf_iter( const leaf_iter &_other ) = default;
    leaf_iter &operator=( const leaf_iter &_other ) = default;
    
    ~leaf_iter() = default;
    
    reference operator*()
    {
      return *m_node;
    }
    
    leaf_iter &operator++()
    {
      do {
        m_node = detail::get_next_preorder( m_node );
      } while ( m_node && !m_node->is_leaf() );
      
      return *this;
    }
    
    leaf_iter operator++( int )
    {
      auto ret = *this;
      ++ret;
      
      return ret;
    }
    
    friend bool operator==( const leaf_iter &_lhs, const leaf_iter &_rhs )
    {
      return _lhs.m_node == _rhs.m_node;
    }
    
    friend bool operator!=( const leaf_iter &_lhs, const leaf_iter &_rhs )
    {
      return !( _lhs == _rhs );
    }
    
  private:
  
    UnderlyingIterator m_node;
  };
  
  template< typename TreeType >
  range< typename TreeType::preorder_iterator > make_preorder_traverse( TreeType &_tree )
  {
    return make_range( _tree.preorder_begin(), _tree.preorder_end() );
  }
  
  template< typename TreeType >
  range< typename TreeType::const_preorder_iterator > make_preorder_traverse( const TreeType &_tree )
  {
    return make_range( _tree.preorder_begin(), _tree.preorder_end() );
  }
  
  template< typename TreeType >
  range< typename TreeType::leaf_iterator > leaf_traverse( TreeType &_tree )
  {
    return make_range( _tree.leaf_begin(), _tree.leaf_end() );
  }
  
  template< typename TreeType >
  range< typename TreeType::const_leaf_iterator > leaf_traverse( const TreeType &_tree )
  {
    return make_range( _tree.leaf_begin(), _tree.leaf_end() );
  }
  
}

#endif  // INC_BVH_TREE_ITERATOR_HPP
