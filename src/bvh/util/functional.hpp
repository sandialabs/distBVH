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
#ifndef INC_BVH_UTIL_FUNCTIONAL_HPP
#define INC_BVH_UTIL_FUNCTIONAL_HPP

#include <tuple>

namespace bvh
{
  namespace detail
  {
    template< typename Functor >
    struct func_props_impl
        : public func_props_impl< decltype( &Functor::operator()) >
    {
    };

    template< typename Return, typename... Args >
    struct func_props_impl< Return( Args... ) >
    {
      using return_type = Return;
      using arg_tuple_type = std::tuple< Args... >;
      using arg_build_tuple_type = std::tuple< std::remove_reference_t< Args >... >;
    };

    // Function pointer
    template< typename Return, typename... Args >
    struct func_props_impl< Return( * )( Args... ) > :
        public func_props_impl< Return( Args... ) >
    {};

    // Member function pointer
    template< typename Class, typename Return, typename... Args >
    struct func_props_impl< Return( Class::* )( Args... ) >
        : public func_props_impl< Return( Class &, Args... ) >
    {};

    template< typename Class, typename Return, typename... Args >
    struct func_props_impl< Return( Class::* )( Args... ) const >
        : public func_props_impl< Return( const Class &, Args... ) >
    {};

    template< typename Class, typename Return, typename... Args >
    struct func_props_impl< Return( Class::* )( Args... ) volatile >
        : public func_props_impl< Return( volatile Class &, Args... ) >
    {};

    template< typename Class, typename Return, typename... Args >
    struct func_props_impl< Return( Class::* )( Args... ) const volatile >
        : public func_props_impl< Return( const volatile Class &, Args... ) >
    {};

    template< typename Functor >
    using func_props = func_props_impl< std::remove_reference_t< Functor > >;
  }

  template< typename Functor >
  using arg_tuple = typename detail::func_props< Functor >::arg_tuple_type;

  template< typename Functor >
  using arg_build_tuple = typename detail::func_props< Functor >::arg_build_tuple_type;

  template< typename Functor >
  using return_type = typename detail::func_props< Functor >::return_type;

  template< typename Functor >
  constexpr std::size_t num_args = std::tuple_size< arg_tuple< Functor > >::value;

  namespace detail
  {
    template< typename F, typename Tuple, std::size_t... I, typename... PreArgs >
    constexpr decltype(auto) apply_impl( F &&_fun, Tuple &&_tup, std::index_sequence< I... >, PreArgs &&... _pre )
    {
      // Not a full replacement for invoke
      std::forward< F >( _fun )( std::forward< PreArgs >( _pre )..., std::get< I >( std::forward< Tuple >( _tup ) )... );
    }
  }

  template< typename F, typename Tuple >
  constexpr decltype(auto) apply( F &&_fun, Tuple &&_tup )
  {
    return detail::apply_impl( std::forward< F >( _fun ), std::forward< Tuple >( _tup ),
        std::make_index_sequence< std::tuple_size< std::remove_reference_t< Tuple > >::value >{} );
  }

  /**
   * Call the function with the given arguments, followed by the tuple as arguments.
   */
  template< typename F, typename Tuple, typename... Args >
  constexpr decltype(auto) apply_first( F &&_fun, Tuple &&_tup, Args &&... _args )
  {
    return detail::apply_impl( std::forward< F >( _fun ), std::forward< Tuple >( _tup ),
                               std::make_index_sequence< std::tuple_size< std::remove_reference_t< Tuple > >::value >{},
                               std::forward< Args >( _args )... );
  }
}

#endif  // INC_BVH_UTIL_FUNCTIONAL_HPP
