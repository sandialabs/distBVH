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
#ifndef INC_BVH_TRANSFORM_ITERATOR_HPP
#define INC_BVH_TRANSFORM_ITERATOR_HPP

#include <type_traits>
#include <iterator>
#include <functional>

namespace bvh
{
  namespace detail
  {
    template< typename UnaryFunction, typename Iterator >
    class transform_proxy
    {
    public:
    
      using reference = std::result_of_t< const UnaryFunction( typename std::iterator_traits< std::remove_reference_t< Iterator > >::reference ) >;
      using value_type = std::remove_cv_t< std::remove_reference_t< reference > >;
      using pointer = value_type *;
      
      template< typename T >
      transform_proxy( T &&_param, UnaryFunction _f )
#ifdef BVH_CXX_STANDARD_17
        : m_value( std::invoke( _f, std::forward< T >( _param ) ) )
#else
        : m_value( _f( std::forward< T >( _param ) ) )
#endif
      {}
      
      transform_proxy( const transform_proxy & ) = default;
      transform_proxy &operator=( const transform_proxy & ) = default;
      
      pointer operator->()
      {
        return &m_value;
      }
      
    private:
      
      value_type m_value;
    };
  }
  
  template< typename UnaryFunction, typename Iterator >
  class transform_iterator
  {
  public:
    
    using reference = std::result_of_t< const UnaryFunction( typename std::iterator_traits< std::remove_reference_t< Iterator > >::reference ) >;
    using value_type = std::remove_cv_t< std::remove_reference_t< reference > >;
    using pointer = detail::transform_proxy< UnaryFunction, Iterator >;
    
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;
    
    transform_iterator() = default;
    transform_iterator( const Iterator &_iter, UnaryFunction _f )
      : m_f( _f ), m_iter( _iter )
    {}
    
    reference operator*()
    {
#ifdef BVH_CXX_STANDARD_17
      return std::invoke( m_f, *m_iter );
#else
      return m_f( *m_iter );
#endif
    }
    
    pointer operator->() const
    {
      return detail::transform_proxy< UnaryFunction, Iterator >( *m_iter, m_f );
    }
    
    transform_iterator &operator++()
    {
      ++m_iter;
      return *this;
    }
    
    transform_iterator operator++( int )
    {
      auto ret = *this;
      ++ret;
      
      return ret;
    }
    
    transform_iterator operator--()
    {
      --m_iter;
      return *this;
    }
    
    transform_iterator operator--( int )
    {
      auto ret = *this;
      --ret;
      
      return ret;
    }
    
    friend bool operator==( const transform_iterator &_lhs, const transform_iterator &_rhs )
    {
      return _lhs.m_iter == _rhs.m_iter;
    }
    
    friend bool operator!=( const transform_iterator &_lhs, const transform_iterator &_rhs )
    {
      return !( _lhs == _rhs );
    }
    
    Iterator &base_iter() { return m_iter; }
    const Iterator &base_iter() const { return m_iter; }
    
  private:
    
    UnaryFunction m_f;
    Iterator      m_iter;
  };
  
  template< typename UnaryFunction, typename Iterator >
  transform_iterator< UnaryFunction, Iterator > make_transform_iterator( Iterator _iter, UnaryFunction _f )
  {
    return transform_iterator< UnaryFunction, Iterator >{ _iter, _f };
  }
  
  template< typename UnaryFunction, typename Iterator >
  typename transform_iterator< UnaryFunction, Iterator >::difference_type distance( const transform_iterator< UnaryFunction, Iterator > &_lhs,
                                                                           const transform_iterator< UnaryFunction, Iterator > & _rhs )
  {
    return std::distance( _lhs.base_iter(), _rhs.base_iter() );
  }
}

#endif // INC_BVH_TRANSFORM_ITERATOR_HPP
