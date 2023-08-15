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
#ifndef INC_BVH_MATH_VECTOR_TRAITS_HPP
#define INC_BVH_MATH_VECTOR_TRAITS_HPP

#include <type_traits>
#include <array>

namespace bvh
{
  namespace m
  {
    template< typename T, unsigned N >
    struct vec;

    template< typename T, unsigned N >
    struct constant_vec;

    template< typename T >
    struct vector_traits;

    template< typename T, unsigned N >
    struct vector_traits< vec< T, N > >
    {
      using component_type = T;
      static constexpr unsigned num_components = N;
    };

    template< typename T, unsigned N >
    struct vector_traits< constant_vec< T, N > >
    {
      using component_type = T;
      static constexpr unsigned num_components = N;
    };

    template< typename T, unsigned N >
    struct vector_traits< std::array< T, N > >
    {
      using component_type = T;
      static constexpr unsigned num_components = N;
    };

    template< typename T, unsigned N >
    struct vector_traits< T[N] >
    {
      using component_type = T;
      static constexpr unsigned num_components = N;
    };

    template< typename T >
    struct is_vector_type : std::false_type {};

    template< typename T, unsigned N >
    struct is_vector_type< vec< T, N > > : std::true_type {};

    template< typename T, unsigned N >
    struct is_vector_type< constant_vec< T, N > > : std::true_type {};

    template< typename T, unsigned N >
    struct is_vector_type< std::array< T, N > > : std::true_type {};

    template< typename T, unsigned N >
    struct is_vector_type< T[N] > : std::true_type {};

    template< typename U, typename V >
    using common_component_t
      = std::common_type_t< typename vector_traits< U >::component_type, typename vector_traits< V >::component_type >;

    template< typename T >
    using epsilon_type_of_t = std::conditional_t< is_vector_type< T >::value, typename vector_traits< T >::component_type, T >;

    namespace detail
    {
      template< typename Vec, typename Replace >
      struct build_vec_type;

      template< template< typename, unsigned > class Vec, typename T, unsigned N, typename Replace >
      struct build_vec_type< Vec< T, N >, Replace >
      {
        using type = Vec< Replace, N >;
      };

      // Promotion rules for constexpr/array/vec types
      template< typename Vec1, typename Vec2, typename RetType >
      struct vec_promotion_type;

      // Regular non-constexpr vec with anything is a vec
      template< typename T1, typename T2, unsigned N, typename RetType >
      struct vec_promotion_type< vec< T1, N >, vec< T2, N >, RetType >
      {
        using type = vec< RetType, N >;
      };

      template< typename T1, typename Vec2, unsigned N, typename RetType >
      struct vec_promotion_type< vec< T1, N >, Vec2, RetType >
      {
        using type = vec< RetType, N >;
      };

      template< typename Vec1, typename T2, unsigned N, typename RetType >
      struct vec_promotion_type< Vec1, vec< T2, N >, RetType >
      {
        using type = vec< RetType, N >;
      };

      // Constant vec with constant vec is constant vec
      template< typename T1, typename T2, unsigned N, typename RetType >
      struct vec_promotion_type< constant_vec< T1, N >, constant_vec< T2, N >, RetType >
      {
        using type = constant_vec< RetType, N >;
      };
    }
  }
}

#endif  // INC_BVH_MATH_VECTOR_TRAITS_HPP
