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
#ifndef INC_BVH_UTIL_SPAN_HPP
#define INC_BVH_UTIL_SPAN_HPP

#include <cstddef>
#include <type_traits>
#include "assert.hpp"
#include <array>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Array.hpp>
#include "../range.hpp"

namespace bvh
{
  inline constexpr std::ptrdiff_t dynamic_extent()
  {
    return -1;
  }

  template< typename T, std::ptrdiff_t Extent >
  class span;

  namespace detail
  {
    template< typename T >
    struct is_std_array_specialization
      : std::false_type
    {};

    template< typename T, std::size_t N >
    struct is_std_array_specialization< std::array< T, N > >
      : std::true_type
    {};

    template< typename T >
    struct is_span_specialization
      : std::false_type
    {};

    template< typename T, std::ptrdiff_t E >
    struct is_span_specialization< span< T, E > >
      : std::true_type
    {};
  }

  template< typename T, std::ptrdiff_t Extent = dynamic_extent() >
  class span
  {
  public:

    using element_type = T;
    using value_type = std::remove_reference_t< std::remove_const_t< T > >;
    using index_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    using iterator = T *;
    using const_iterator = const T *;

    using reverse_iterator = std::reverse_iterator< iterator >;
    using const_reverse_iterator = std::reverse_iterator< const_iterator >;

    static constexpr ptrdiff_t extent = Extent;

    template< typename = std::enable_if_t< ( extent == 0 ) || ( extent == dynamic_extent() ) > >
    constexpr KOKKOS_INLINE_FUNCTION span() noexcept
      : m_data( nullptr ), m_count( 0 )
    {

    }

    constexpr KOKKOS_INLINE_FUNCTION span( pointer _ptr, index_type _count )
      : m_data( _ptr ), m_count( _count )
    {
      BVH_ASSERT( extent == dynamic_extent() || _count == static_cast< index_type >( extent ) );
    }

    constexpr KOKKOS_INLINE_FUNCTION span( pointer _first, pointer _last )
      : span( _first, _last - _first )
    {}

    template< std::size_t N,
              typename = std::enable_if_t< extent == dynamic_extent() || N == static_cast< std::size_t >( extent ) > >
    constexpr KOKKOS_INLINE_FUNCTION span( element_type ( &_arr )[N] ) noexcept : m_data( _arr ),
                                                           m_count( N )
    {

    }

    template< std::size_t N,
              typename = std::enable_if_t< extent == dynamic_extent() || N == static_cast< std::size_t >( extent ) > >
    constexpr span( std::array< value_type, N > &_arr ) noexcept : m_data( _arr.data() ),
                                                                   m_count( N )
    {

    }

    template< std::size_t N,
              typename = std::enable_if_t< extent == dynamic_extent() || N == static_cast< std::size_t >( extent ) > >
    constexpr span( const std::array< value_type, N > &_arr ) noexcept : m_data( _arr.data() ),
                                                                         m_count( N )
    {

    }

    template< std::size_t N,
              typename = std::enable_if_t< extent == dynamic_extent() || N == static_cast< std::size_t >( extent ) > >
    constexpr KOKKOS_INLINE_FUNCTION span( Kokkos::Array< value_type, N > &_arr ) noexcept : m_data( _arr.data() ),
                                                                   m_count( N )
    {

    }

    template< std::size_t N,
              typename = std::enable_if_t< extent == dynamic_extent() || N == static_cast< std::size_t >( extent ) > >
    constexpr KOKKOS_INLINE_FUNCTION span( const Kokkos::Array< value_type, N > &_arr ) noexcept : m_data( _arr.data() ),
                                                                         m_count( N )
    {

    }

    template< typename Container, typename = std::enable_if_t<
      !std::is_array< Container >::value
      && !detail::is_std_array_specialization< Container >::value
      && !detail::is_span_specialization< Container >::value
      && std::is_convertible< std::remove_pointer_t< decltype( std::declval< Container >().data() ) >( * )[],
        element_type (*)[] >::value
    > >
    constexpr span( Container &_container )
      : m_data( _container.data() ), m_count( _container.size() )
    {}

    template< typename Container, typename = std::enable_if_t<
      !std::is_array< Container >::value
      && !detail::is_std_array_specialization< Container >::value
      && !detail::is_span_specialization< Container >::value
      && std::is_convertible< std::remove_pointer_t< decltype( std::declval< Container >().data() ) >( * )[],
                              element_type (*)[] >::value
    > >
    constexpr span( const Container &_container )
      : m_data( _container.data() ), m_count( _container.size() )
    {}

    template< typename U, std::ptrdiff_t N, typename = std::enable_if_t<
      ( extent == dynamic_extent() || N == extent )
      && ( std::is_same_v< std::remove_const_t< element_type >, U > )
    >
    >
    constexpr KOKKOS_INLINE_FUNCTION span( const span< U, N > &_other ) noexcept
      : m_data( _other.data() ),
        m_count( _other.size() )
    {

    }

    constexpr KOKKOS_DEFAULTED_FUNCTION span( const span &_other ) noexcept = default;

    constexpr KOKKOS_INLINE_FUNCTION pointer data() const noexcept { return m_data; }
    constexpr KOKKOS_INLINE_FUNCTION index_type size() const noexcept { return m_count; }
    constexpr KOKKOS_INLINE_FUNCTION index_type size_bytes() const noexcept { return m_count * sizeof( element_type ); }
    constexpr KOKKOS_INLINE_FUNCTION bool empty() const noexcept { return m_count == 0; }

    constexpr KOKKOS_INLINE_FUNCTION reference operator[]( index_type _idx ) const
    {
      BVH_ASSERT( _idx < m_count );
      return m_data[_idx];
    }

    constexpr KOKKOS_INLINE_FUNCTION reference operator()( index_type _idx ) const
    {
      return ( *this )[_idx];
    }

    template< std::ptrdiff_t Count >
    constexpr KOKKOS_INLINE_FUNCTION span< element_type, Count > first() const
    {
      BVH_ASSERT( ( Count >= 0 ) && ( Count <= m_count ) );
      return span< element_type, Count >( m_data, Count );
    }

    constexpr KOKKOS_INLINE_FUNCTION span< element_type, dynamic_extent() > first( std::ptrdiff_t _count )
    {
      BVH_ASSERT( ( _count >= 0 ) && ( _count <= m_count ) );
      return span< element_type, dynamic_extent() >( m_data, _count );
    }

    template< std::ptrdiff_t Count >
    constexpr KOKKOS_INLINE_FUNCTION span< element_type, Count > last() const
    {
      BVH_ASSERT( ( Count >= 0 ) && ( Count <= m_count ) );
      return span< element_type, Count >( m_data + m_count - Count, Count );
    }

    constexpr KOKKOS_INLINE_FUNCTION span< element_type, dynamic_extent() > last( std::ptrdiff_t _count )
    {
      BVH_ASSERT( ( _count >= 0 ) && ( _count <= m_count ) );
      return span< element_type, dynamic_extent() >( m_data + m_count - _count, _count );
    }

    template< std::ptrdiff_t Offset, std::ptrdiff_t Count = dynamic_extent() >
    constexpr KOKKOS_INLINE_FUNCTION auto subspan() const
    {
      BVH_ASSERT( ( Offset >= 0 ) && ( Offset < m_count ) );
      BVH_ASSERT( ( Count >= 0 ) || ( Count == dynamic_extent() ) );
      BVH_ASSERT( Offset + Count <= m_count );

      constexpr std::ptrdiff_t ext = Count != dynamic_extent() ?
                                       Count
                                     : ( Extent != dynamic_extent() ?
                                         Extent - Offset
                                       : dynamic_extent() );


      return span< element_type, ext >( m_data + Offset, ( Count == dynamic_extent() ) ? m_count - Offset : Count );
    }

    constexpr KOKKOS_INLINE_FUNCTION auto subspan( std::ptrdiff_t _offset, std::ptrdiff_t _count = dynamic_extent() ) const
    {
      BVH_ASSERT( ( _offset >= 0 ) && ( _offset < m_count ) );
      BVH_ASSERT( ( _count >= 0 ) || ( _count == dynamic_extent() ) );
      BVH_ASSERT( _offset + _count <= m_count );
      return span< element_type, dynamic_extent() >( m_data + _offset, ( _count == dynamic_extent() ) ? m_count - _offset : _count );
    }

    constexpr KOKKOS_INLINE_FUNCTION iterator begin() const noexcept { return m_data; }
    constexpr KOKKOS_INLINE_FUNCTION iterator cbegin() const noexcept { return m_data; }

    constexpr KOKKOS_INLINE_FUNCTION iterator end() const noexcept { return m_data + m_count; }
    constexpr KOKKOS_INLINE_FUNCTION iterator cend() const noexcept { return m_data + m_count; }

    constexpr KOKKOS_INLINE_FUNCTION iterator rbegin() const noexcept { return std::make_reverse_iterator( end() ); }
    constexpr KOKKOS_INLINE_FUNCTION iterator crbegin() const noexcept { return std::make_reverse_iterator( cend() ); }

    constexpr KOKKOS_INLINE_FUNCTION iterator rend() const noexcept { return std::make_reverse_iterator( begin() ); }
    constexpr KOKKOS_INLINE_FUNCTION iterator crend() const noexcept { return std::make_reverse_iterator( cbegin() ); }

    /* implicit */ operator KOKKOS_INLINE_FUNCTION range< iterator >() { return range< iterator >( begin(), end() ); }

  private:

    pointer m_data;
    index_type m_count;
  };

  template< std::ptrdiff_t Extent = dynamic_extent(), typename T >
  KOKKOS_INLINE_FUNCTION auto make_span( T *_begin, T *_end )
  {
    return span< T, Extent >( _begin, _end );
  }

  template< typename Container >
  KOKKOS_INLINE_FUNCTION auto make_span( Container &_c )
  {
    return span< typename Container::value_type >( _c );
  }

  template< typename Container >
  KOKKOS_INLINE_FUNCTION auto make_const_span( const Container &_c )
  {
    return span< const typename Container::value_type >( _c );
  }
}

#endif  // INC_BVH_UTIL_SPAN_HPP
