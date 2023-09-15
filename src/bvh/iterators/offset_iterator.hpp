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
#ifndef INC_BVH_OFFSET_ITERATOR_HPP
#define INC_BVH_OFFSET_ITERATOR_HPP

#include <iterator>
#include <type_traits>

namespace bvh
{
  template< typename RandomAccessContainer >
  class offset_iter
  {
  public:
    
    using container_type = RandomAccessContainer;
    
    using value_type = typename RandomAccessContainer::value_type;
    using pointer = std::conditional_t< std::is_const< RandomAccessContainer >::value,
      const value_type *, value_type * >;
    using reference = std::conditional_t< std::is_const< RandomAccessContainer >::value,
      const value_type &, value_type & >;
    
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;
    
    offset_iter()
      : m_offset( static_cast< difference_type >( -1 ) ), m_container( nullptr )
    {
    
    }
    
    explicit offset_iter( RandomAccessContainer &_container )
      : m_offset( static_cast< difference_type >( _container.size() ) ), m_container( &_container )
    {}
    
    
    offset_iter( RandomAccessContainer &_container, difference_type _offset )
      : m_offset( _offset ), m_container( &_container )
    {}
    
    offset_iter( const offset_iter &_other )
      : m_offset( _other.m_offset ), m_container( _other.m_container )
    {
    }
    
    offset_iter( offset_iter &&_other )
      : m_offset( static_cast< difference_type >( -1 ) ), m_container( nullptr )
    {
      std::swap( m_offset, _other.m_offset );
      std::swap( m_container, _other.m_container );
    }
    
    // Implicit conversion to const version of iterator
    template< typename = std::enable_if_t< std::is_const< RandomAccessContainer >::value > >
    offset_iter( const offset_iter< std::remove_const_t< RandomAccessContainer > > &_other )
      : m_offset( _other.offset() ), m_container( _other.container() )
    {
    
    }
    
    offset_iter &
    operator=( const offset_iter &_other )
    {
      m_offset = _other.m_offset;
      m_container = _other.m_container;
      
      return *this;
    }
    
    offset_iter &
    operator=( offset_iter &&_other ) noexcept
    {
      std::swap( m_offset, _other.m_offset );
      std::swap( m_container, _other.m_container );
      
      return *this;
    }
    
    reference operator*() const
    {
      return ( *m_container ).at( m_offset );
    }
    
    pointer operator->() const
    {
      return &( *m_container ).at( m_offset );
    }
    
    offset_iter &operator++()
    {
      ++m_offset;
      return *this;
    }
    
    offset_iter operator++( int )
    {
      auto ret = *this;
      ++( *this );
      
      return ret;
    }
    
    offset_iter &operator--()
    {
      --m_offset;
      return *this;
    }
    
    offset_iter operator--( int )
    {
      auto ret = *this;
      --( *this );
      
      return ret;
    }
    
    offset_iter &
    operator+=( difference_type _n )
    {
      m_offset += _n;
      return *this;
    }
    
    offset_iter &
    operator-=( difference_type _n )
    {
      m_offset -= _n;
      return *this;
    }
    
    friend offset_iter operator+( offset_iter _lhs, difference_type _n )
    {
      return _lhs += _n;
    }
    
    friend offset_iter operator+( difference_type _n, offset_iter _rhs )
    {
      return _rhs += _n;
    }
    
    friend offset_iter operator-( offset_iter _lhs, difference_type _n )
    {
      return _lhs -= _n;
    }
    
    friend difference_type operator-( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      return _lhs.m_offset - _rhs.m_offset;
    }
    
    reference operator[]( difference_type _n )
    {
      return *( *this + _n );
    }
    
    pointer get() const
    {
      return &( *m_container ).at( m_offset );
    }
    
    friend bool operator==( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      return ( _lhs.m_offset == _rhs.m_offset ) && ( _lhs.m_container == _rhs.m_container );
    }
    
    friend bool operator!=( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      return !( _lhs == _rhs );
    }
    
    friend bool operator<( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      if ( _lhs.m_container == _rhs.m_container )
        return _lhs.m_offset < _rhs.m_offset;
      return _lhs.m_container < _rhs.m_container;
    }
    
    friend bool operator<=( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      if ( _lhs.m_container == _rhs.m_container )
        return _lhs.m_offset <= _rhs.m_offset;
      return _lhs.m_container <= _rhs.m_container;
    }
    
    friend bool operator>( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      if ( _lhs.m_container == _rhs.m_container )
        return _lhs.m_offset > _rhs.m_offset;
      return _lhs.m_container > _rhs.m_container;
    }
    
    friend bool operator>=( const offset_iter &_lhs, const offset_iter &_rhs )
    {
      if ( _lhs.m_container == _rhs.m_container )
        return _lhs.m_offset >= _rhs.m_offset;
      return _lhs.m_container >= _rhs.m_container;
    }
    
    bool valid() const
    {
      return m_container && ( m_offset >= 0 ) && ( static_cast< typename RandomAccessContainer::size_type >( m_offset ) < m_container->size() );
    }
    
    
    operator bool() const
    {
      return valid();
    }
    
    bool operator!() const
    {
      return !valid();
    }
    
    operator pointer() const
    {
      return get();
    }
    
    difference_type offset() const
    {
      return m_offset;
    }
  
    RandomAccessContainer *container() const
    {
      return m_container;
    }
    
    typename RandomAccessContainer::size_type container_size() const noexcept
    {
      return m_container ? m_container->size() : 0;
    }
    
    void rebase( RandomAccessContainer &_container ) noexcept
    {
      m_container = &_container;
    }
    
  private:
    
    difference_type m_offset;
    RandomAccessContainer *m_container;
  };
  
  template< typename RandomAccessContainer, typename... Args >
  offset_iter< RandomAccessContainer >
  emplace_back_offset( RandomAccessContainer &_container, Args &&... _args )
  {
    offset_iter< RandomAccessContainer > ret( _container, _container.size() );
    
    _container.emplace_back( std::forward< Args >( _args )... );
    
    return ret;
  }
  
}

#endif  // INC_BVH_OFFSET_ITERATOR_HPP
