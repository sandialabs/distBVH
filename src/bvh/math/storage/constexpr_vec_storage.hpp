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
#ifndef INC_BVH_MATH_STORAGE_CONSTEXPR_VEC_STORAGE_HPP
#define INC_BVH_MATH_STORAGE_CONSTEXPR_VEC_STORAGE_HPP

#include <Kokkos_Macros.hpp>

#include "../../util/attributes.hpp"

namespace bvh
{
  namespace m
  {
    namespace detail
    {
      template< typename T, unsigned N >
      struct constexpr_vec_storage
      {
        constexpr BVH_INLINE
        constexpr_vec_storage()
            : d{}
        {}

        explicit constexpr BVH_INLINE
        constexpr_vec_storage( T _s )
            : d{}
        {
          for ( unsigned i = 0; i < N; ++i )
            d[i] = _s;
        }

        KOKKOS_DEFAULTED_FUNCTION ~constexpr_vec_storage() = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION constexpr_vec_storage( const constexpr_vec_storage & ) = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION constexpr_vec_storage( constexpr_vec_storage && ) = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION constexpr_vec_storage &operator=( const constexpr_vec_storage & ) = default;

        constexpr KOKKOS_DEFAULTED_FUNCTION constexpr_vec_storage &operator=( constexpr_vec_storage && ) = default;

        constexpr BVH_INLINE T &operator[]( int i )
        { return d[i]; }

        constexpr BVH_INLINE T operator[]( int i ) const
        { return d[i]; }

        T d[N];
      };
    } // namespace detail
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_STORAGE_CONSTEXPR_VEC_STORAGE_HPP
