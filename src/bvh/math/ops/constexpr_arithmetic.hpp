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
#ifndef INC_BVH_MATH_OPS_CONSTEXPR_ARITHMETIC_HPP
#define INC_BVH_MATH_OPS_CONSTEXPR_ARITHMETIC_HPP

#include "../compiler.hpp"
#include "../storage/constexpr_vec_storage.hpp"
#include "vec_generic_ops.hpp"

namespace bvh
{
  namespace m
  {
    template< typename T, unsigned N >
    struct constant_vec;

    namespace ops
    {
      namespace constant
      {
        template< typename T, unsigned N >
        using storage_type = ::bvh::m::detail::constexpr_vec_storage< T, N >;

        template< typename T, unsigned N >
        constexpr BVH_INLINE auto
        undefined()
        {
          return storage_type< T, N >{};
        }

        template< typename T, unsigned N >
        constexpr BVH_INLINE auto
        set1( T _s )
        {
          storage_type< T, N > ret{ _s };
          return ret;
        }

        template< typename T, unsigned N >
        constexpr BVH_INLINE auto
        zeros()
        {
          return set1< T, N >( T{ 0.0 } );
        }

        template< typename T, typename U, unsigned N >
        constexpr BVH_INLINE auto
        add( const constant_vec< T, N > &_lhs, const constant_vec< U, N > &_rhs )
        {
          using result_type = decltype( T{} + U{} );
          constant_vec< result_type, N > ret;
        }
      } // namespace constant
    }   // namespace ops
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_OPS_CONSTEXPR_ARITHMETIC_HPP
