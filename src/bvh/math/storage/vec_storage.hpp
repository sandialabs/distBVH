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
#ifndef INC_BVH_MATH_STORAGE_VEC_STORAGE_HPP
#define INC_BVH_MATH_STORAGE_VEC_STORAGE_HPP

#include <type_traits>

#include "../compiler.hpp"
#include "../../util/kokkos.hpp"

#ifdef BVH_SIMD
#include <immintrin.h>
#endif

#include "constexpr_vec_storage.hpp"


namespace bvh
{
  namespace m
  {
    namespace detail
    {
#ifdef BVH_SIMD
      template< typename T >
      struct storage_data;

      template<>
      struct storage_data< double >
      {
        using type = __m256d;
        static constexpr unsigned vector_size = 4;
      };
#else
      template< typename T >
      struct storage_data
      {
        static constexpr unsigned vector_size = 4;
        using type = T[vector_size];
      };
#endif

      template< typename T, unsigned N >
      struct vec_storage
      {
        constexpr KOKKOS_INLINE_FUNCTION
        vec_storage() noexcept
            : chunks{}
        {}

        KOKKOS_DEFAULTED_FUNCTION ~vec_storage() noexcept = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION vec_storage( const vec_storage & ) noexcept = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION vec_storage( vec_storage && ) noexcept = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION vec_storage &operator=( const vec_storage & ) noexcept = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION vec_storage &operator=( vec_storage && ) noexcept = default;

        constexpr KOKKOS_INLINE_FUNCTION T &operator[]( int i ) noexcept
        { return reinterpret_cast< T * >( chunks )[i]; }

        constexpr KOKKOS_INLINE_FUNCTION const T &operator[]( int i ) const noexcept
        {
          return reinterpret_cast< const T * >( chunks )[i];
        }

        static constexpr unsigned chunk_size = storage_data< T >::vector_size;
        static constexpr unsigned num_unmasked = N / chunk_size;
        static constexpr unsigned mask = ( 0x1u << ( N % chunk_size ) ) - 1;
        static constexpr unsigned num_chunks = num_unmasked + ( mask ? 1 : 0 );

        typename storage_data< T >::type chunks[num_chunks];
      };
    } // namespace detail
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_STORAGE_VEC_STORAGE_HPP
