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
#ifndef INC_BVH_MATH_OPS_MEM_HPP
#define INC_BVH_MATH_OPS_MEM_HPP

#if defined( __GNUC__ ) && defined( BVH_SIMD )
#include <immintrin.h>
#endif

#include "../storage/vec_storage.hpp"

namespace bvh
{
  namespace m
  {
    namespace detail
    {
      template< typename T, unsigned N >
      void
      zeros( vec_storage <T, N> &_target )
      {
        for ( unsigned i = 0; i < N; ++i )
          _target[i] = T{ 0 };
      };

#if defined( __GNUC__ ) && defined( BVH_SIMD )
      template<>
      void
      zeros( vec_storage< double, 3 > &_target )
      {
        _target.d.m256 = _mm256_setzero_pd();
      }
#endif

      template< typename T, unsigned N >
      void
      set( vec_storage <T, N> &_target, T _val )
      {
        for ( unsigned i = 0; i < N; ++i )
          _target[i] = _val;
      };

#if defined( __GNUC__ ) && defined( BVH_SIMD )
      template<>
      void
      set( vec_storage< double, 3 > &_target, double _val )
      {
        _target.d.m256 = _mm256_set1_pd( _val );
      }
#endif

      template< typename T, unsigned N >
      void
      ones( vec_storage <T, N> &_target )
      {
        set( _target, T{ 1 } );
      };
    } // namespace detail
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_OPS_MEM_HPP
