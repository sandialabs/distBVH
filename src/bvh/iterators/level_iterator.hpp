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
#ifndef INC_BVH_LEVEL_ITERATOR_HPP
#define INC_BVH_LEVEL_ITERATOR_HPP

#include <iterator>
#include "../tree_iterator.hpp"
#include "../range.hpp"

namespace bvh
{
  namespace detail
  {
    template< typename UnderlyingIterator >
    UnderlyingIterator get_next_towards_level( UnderlyingIterator _node, int &_curr_depth, const int _level )
    {
      if ( _node->has_left() && ( _curr_depth < _level) )
      {
        ++_curr_depth;
        return _node->left();
      }
  
      if ( _node->has_right() && ( _curr_depth < _level ) )
      {
        ++_curr_depth;
        return _node->right();
      }
  
      // Leaf node
      while ( _node->has_parent() )
      {
        // If we're on the left child and there is a right sibling
        if ( _node == _node->parent()->left()
             && _node->parent()->has_right() && ( _curr_depth <= _level ) )
          return _node->parent()->right();
    
        _node = _node->parent();
        --_curr_depth;
      }
  
      return UnderlyingIterator{};
    }
  }
  
  template< typename UnderlyingIterator >
  class level_iter
  {
  public:
  
    using value_type = typename std::iterator_traits< UnderlyingIterator >::value_type;
    using pointer = typename std::iterator_traits< UnderlyingIterator >::pointer;
    using reference = typename std::iterator_traits< UnderlyingIterator >::reference;
  
    using difference_type = typename std::iterator_traits< UnderlyingIterator >::difference_type;
    using iterator_category = std::forward_iterator_tag;
  
    explicit level_iter( const int _level, const UnderlyingIterator &_iter = UnderlyingIterator{} )
      : m_node( _iter ), m_level( _level ), m_current_depth( 0 )
    {
      if ( m_node != UnderlyingIterator{} )
      {
        m_current_depth = _iter->level();
        while ( m_node && ( m_current_depth != _level) )
          m_node = detail::get_next_towards_level( m_node, m_current_depth, m_level );
      }
    }
  
    level_iter( const level_iter &_other ) = default;
    level_iter &operator=( const level_iter &_other ) = default;
  
    ~level_iter() = default;
  
    reference operator*()
    {
      return *m_node;
    }
  
    level_iter &operator++()
    {
      do {
        m_node = detail::get_next_towards_level( m_node, m_current_depth, m_level );
      } while ( m_node && ( m_current_depth != m_level ) );
    
      return *this;
    }
  
    level_iter operator++( int )
    {
      auto ret = *this;
      ++ret;
    
      return ret;
    }
  
    friend bool operator==( const level_iter &_lhs, const level_iter &_rhs )
    {
      return ( _lhs.m_node == _rhs.m_node && ( _lhs.m_level == _rhs.m_level ) );
    }
  
    friend bool operator!=( const level_iter &_lhs, const level_iter &_rhs )
    {
      return !( _lhs == _rhs );
    }

  private:
  
    UnderlyingIterator m_node;
    const int m_level;
    int m_current_depth;
  };
  
  template< typename TreeType >
  range< typename TreeType::level_iterator > level_traverse( TreeType &_tree, const int _level )
  {
    return make_range( _tree.level_begin( _level ), _tree.level_end( _level ) );
  }
  
  template< typename UnderlyingIterator >
  class max_level_iter
  {
  public:
  
    using value_type = typename std::iterator_traits< UnderlyingIterator >::value_type;
    using pointer = typename std::iterator_traits< UnderlyingIterator >::pointer;
    using reference = typename std::iterator_traits< UnderlyingIterator >::reference;
  
    using difference_type = typename std::iterator_traits< UnderlyingIterator >::difference_type;
    using iterator_category = std::forward_iterator_tag;
    
    explicit max_level_iter( const int _level, const UnderlyingIterator &_iter = UnderlyingIterator{} )
      : m_node( _iter ), m_level( _level ), m_current_depth( 0 )
    {
      if ( m_node != UnderlyingIterator{} )
      {
        m_current_depth = _iter->level();
        while ( m_node && ( m_current_depth != _level) && !m_node->is_leaf() )
          m_node = detail::get_next_towards_level( m_node, m_current_depth, m_level );
      }
    }
  
    max_level_iter( const max_level_iter &_other ) = default;
    max_level_iter &operator=( const max_level_iter &_other ) = default;
    
    ~max_level_iter() = default;
    
    reference operator*()
    {
      return *m_node;
    }
  
    max_level_iter &operator++()
    {
      do {
        m_node = detail::get_next_towards_level( m_node, m_current_depth, m_level );
      } while ( m_node && ( m_current_depth != m_level ) && !m_node->is_leaf() );
      
      return *this;
    }
  
    max_level_iter operator++( int )
    {
      auto ret = *this;
      ++ret;
      
      return ret;
    }
    
    friend bool operator==( const max_level_iter &_lhs, const max_level_iter &_rhs )
    {
      return ( _lhs.m_node == _rhs.m_node && ( _lhs.m_level == _rhs.m_level ) );
    }
    
    friend bool operator!=( const max_level_iter &_lhs, const max_level_iter &_rhs )
    {
      return !( _lhs == _rhs );
    }
  
  private:
  
    UnderlyingIterator m_node;
    const int m_level;
    int m_current_depth;
  };
  
  template< typename TreeType >
  range< typename TreeType::max_level_iterator > max_level_traverse( TreeType &_tree, const int _level )
  {
    return make_range( _tree.max_level_begin( _level ), _tree.max_level_end( _level ) );
  }
  
  template< typename TreeType >
  range< typename TreeType::const_max_level_iterator > max_level_traverse( const TreeType &_tree, const int _level )
  {
    return make_range( _tree.max_level_begin( _level ), _tree.max_level_end( _level ) );
  }
}

#endif  // INC_BVH_LEVEL_ITERATOR_HPP
