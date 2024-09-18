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
#ifndef INC_BVH_TRAITS_HPP
#define INC_BVH_TRAITS_HPP

#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace bvh
{
  template< typename... >
  using void_t = void;

  template< unsigned N >
  struct overload_priority
    : overload_priority< N - 1 >
  {
  };

  template<>
  struct overload_priority< 0 >
  {};

  namespace detail
  {
    template< typename Element >
    constexpr KOKKOS_INLINE_FUNCTION auto
      get_kdop_impl( const Element &_element, overload_priority< 1 > ) -> decltype( get_entity_kdop( _element ) )
    {
      return get_entity_kdop( _element );
    }

    template< typename Element >
    constexpr KOKKOS_INLINE_FUNCTION auto get_kdop_impl( const Element &_element,
                                                         overload_priority< 0 > ) -> decltype( _element.kdop() )
    {
      return _element.kdop();
    }

    template< typename Element > constexpr KOKKOS_INLINE_FUNCTION auto get_kdop( const Element &_element )
    {
      return get_kdop_impl( _element, overload_priority< 1 >{} );
    }

    template< typename Element >
    constexpr KOKKOS_INLINE_FUNCTION auto get_centroid_impl( const Element &_element, overload_priority< 1 > )
      -> decltype( get_entity_centroid( _element ) )
    {
      return get_entity_centroid( _element );
    }

    template< typename Element >
    constexpr KOKKOS_INLINE_FUNCTION auto get_centroid_impl( const Element &_element,
                                                             overload_priority< 0 > ) -> decltype( _element.centroid() )
    {
      return _element.centroid();
    }

    template< typename Element > constexpr KOKKOS_INLINE_FUNCTION auto get_centroid( const Element &_element )
    {
      return get_centroid_impl( _element, overload_priority< 1 >{} );
    }

    template< typename Element >
    constexpr KOKKOS_INLINE_FUNCTION auto get_global_id_impl( const Element &_element, overload_priority< 1 > )
      -> decltype( get_entity_global_id( _element ) )
    {
      return get_entity_global_id( _element );
    }

    template< typename Element >
    constexpr KOKKOS_INLINE_FUNCTION auto
      get_global_id_impl( const Element &_element, overload_priority< 0 > ) -> decltype( _element.global_id() )
    {
      return _element.global_id();
    }

    template< typename Element > constexpr KOKKOS_INLINE_FUNCTION auto get_global_id( const Element &_element )
    {
      return get_global_id_impl( _element, overload_priority< 1 >{} );
    }
  }

  template< typename Element >
  struct element_traits
  {
    using kdop_type = std::remove_cv_t< std::remove_reference_t< decltype( detail::get_kdop( std::declval< Element >() ) ) > >;
    using centroid_type = std::remove_cv_t< std::remove_reference_t< decltype( detail::get_centroid( std::declval< Element >() ) ) > >;
    using global_id_type = std::remove_cv_t< std::remove_reference_t< decltype( detail::get_global_id( std::declval< Element >() ) ) > >;

    static constexpr KOKKOS_INLINE_FUNCTION kdop_type get_kdop( const Element &_element )
    {
      return detail::get_kdop( _element );
    }

    static constexpr KOKKOS_INLINE_FUNCTION centroid_type get_centroid( const Element &_element )
    {
      return detail::get_centroid( _element );
    }

    static constexpr KOKKOS_INLINE_FUNCTION global_id_type get_global_id( const Element &_element )
    {
      return detail::get_global_id( _element );
    }
  };
}

#endif  // INC_BVH_TRAITS_HPP
