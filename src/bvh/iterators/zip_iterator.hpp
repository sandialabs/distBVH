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
#ifndef INC_BVH_ZIP_ITERATOR_HPP
#define INC_BVH_ZIP_ITERATOR_HPP

#include <tuple>
#include <iterator>

namespace bvh
{
  namespace detail
  {
    template< typename Tuple >
    struct zip_proxy
    {
      zip_proxy( Tuple &&_proxy )
        : tup( std::move( _proxy ) ) 
      {}
      
      Tuple tup;
    };
      
    template< std::size_t N, typename Tuple >
    auto get( zip_proxy< Tuple > &_z )
    {
      return std::get< N >( _z.tup );
    }
    
    template< typename Tuple >
    void swap( zip_proxy< Tuple > &&_lhs, zip_proxy< Tuple > &&_rhs )
    {
      std::swap( _lhs.tup, _rhs.tup );
    }
    
    template< typename Tuple, std::size_t I >
    using get_iterator_reference = typename std::iterator_traits< std::tuple_element_t< I, std::remove_reference_t< Tuple > > >::reference;
    
    template< typename Tuple, std::size_t... I >
    constexpr auto dereference_tuple( Tuple &&_t, std::index_sequence< I... > )
    {
      using ret_type = std::tuple< get_iterator_reference< Tuple, I >... >;
      return zip_proxy< ret_type >( ret_type( *std::get< I >( std::forward< Tuple >( _t ) )... ) );
    }
    
    template< typename Tuple, typename UnaryFunction, std::size_t... I >
    constexpr void tuple_for_each( Tuple &&_t, UnaryFunction _f, std::index_sequence< I... > )
    {
#ifdef BVH_CXX_STANDARD_17
      ( _f( std::get< I >( std::forward< Tuple >( _t ) ) ), ... );
#else
      ( void )std::initializer_list< int > {
        ( _f( std::get< I >( std::forward< Tuple >( _t ) ) ), void(), 0 )...
      };
#endif
    }
  }
  
  template< typename... Iterators >
  class zip_iterator
  {
  public:
    
    using reference = detail::zip_proxy< std::tuple< typename std::iterator_traits< Iterators >::reference... > >;
    using value_type = reference;
    using pointer = reference *;
    
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;
    
    zip_iterator() = default;
    
    template< typename... Args >
    explicit zip_iterator( Args ..._args )
      : m_iterators( _args... )
    {}
    
    zip_iterator( const zip_iterator & ) = default;
    zip_iterator( zip_iterator && ) = default;
    zip_iterator &operator=( const zip_iterator & ) = default;
    zip_iterator &operator=( zip_iterator && ) = default;
  
    reference operator*()
    {
      return detail::dereference_tuple( m_iterators, std::index_sequence_for< Iterators... >{} );
    }
    
    zip_iterator &operator++()
    {
      detail::tuple_for_each( m_iterators, []( auto &iter ) { ++iter; }, std::index_sequence_for< Iterators... >{} );
      return *this;
    }
    
    zip_iterator &operator++( int )
    {
      auto ret = *this;
      ++ret;
      
      return ret;
    }
    
    zip_iterator &operator--()
    {
      detail::tuple_for_each( m_iterators, []( auto &iter ) { --iter; }, std::index_sequence_for< Iterators... >{} );
      return *this;
    }
    
    zip_iterator &operator--( int )
    {
      auto ret = *this;
      --ret;
      
      return ret;
    }
    
    friend bool operator==( const zip_iterator &_lhs, const zip_iterator &_rhs )
    {
      return _lhs.m_iterators == _rhs.m_iterators;
    }
    
    friend bool operator!=( const zip_iterator &_lhs, const zip_iterator &_rhs )
    {
      return _lhs.m_iterators != _rhs.m_iterators;
    }
    
    template< std::size_t I >
    auto get() const noexcept
    {
      return std::get< I >( m_iterators );
    }
    
    template< std::size_t I >
    using iter_type = std::tuple_element_t< I, std::tuple< Iterators... > >;
    
    friend void swap( zip_iterator &_lhs, zip_iterator &_rhs )
    {
      return std::swap( _lhs.m_iterators, _rhs.m_iterators );
    }
    
    friend void swap( reference _lhs, reference _rhs )
    {
      return std::swap( _lhs, _rhs );
    }
    
  private:
    
    std::tuple< Iterators... > m_iterators;
  };
  
  template< typename... Args >
  zip_iterator< Args... > make_zip_iterator( Args ... _args)
  {
    return zip_iterator< Args... >( _args... );
  }
  
  template< typename... Iterators >
  typename zip_iterator< Iterators... >::difference_type distance( const zip_iterator< Iterators... > &_lhs,
                                                                   const zip_iterator< Iterators... > &_rhs )
  {
    // All iterator distances should be the same
    return std::distance( _lhs.template get< 0 >(), _rhs.template get< 0 >() );
  }
}

#endif  // INC_BVH_ZIP_ITERATOR_HPP
