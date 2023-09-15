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
#ifndef INC_BVH_MATH_CONSTANT_VEC_HPP
#define INC_BVH_MATH_CONSTANT_VEC_HPP

#include "named_access.hpp"
#include "ops/constexpr_arithmetic.hpp"
#include "ops/vec_base_ops.hpp"
#include "storage/constexpr_vec_storage.hpp"
#include "vec.hpp"

namespace bvh
{
  namespace m
  {
    template< typename T, unsigned N >
    struct alignas( 32 ) constant_vec : vec_named_access< constant_vec, T, N >
    {
      constexpr BVH_INLINE
      constant_vec()
          : d( ops::constant::zeros< T, N >() )
      {}

      explicit constexpr BVH_INLINE
      constant_vec( T _s )
          : d( ops::init_set1_storage_op< constant_vec, T, N >::execute( _s ) )
      {}

      template< typename... Args, typename = std::enable_if_t< sizeof...( Args ) == N > >
      constexpr BVH_INLINE
      constant_vec( Args &&... _args )
          : d( ops::init_set_storage_op< constant_vec, T, N >::execute( std::forward< Args >( _args )... ) )
      {}

      explicit constexpr BVH_INLINE
      constant_vec( const detail::constexpr_vec_storage< T, N > &_d )
          : d( _d )
      {}


      static constexpr BVH_INLINE constant_vec
      undefined()
      {
        return constant_vec( ops::init_undefined_storage_op< constant_vec, T, N >::execute() );
      }

      static constexpr BVH_INLINE constant_vec
      zeros()
      {
        return constant_vec();
      }

      static constexpr BVH_INLINE constant_vec
      set1( T _s )
      {
        return constant_vec( _s );
      }

      static constexpr BVH_INLINE constant_vec
      ones()
      {
        return set1( 1.0 );
      }

      constexpr BVH_INLINE T operator[]( unsigned _n ) const
      { return d[_n]; }

      constexpr BVH_INLINE T &operator[]( unsigned _n )
      { return d[_n]; }

      constexpr BVH_INLINE constant_vec
      normal() const
      {
        return ::bvh::m::normal( *this );
      }

      using storage_type = detail::constexpr_vec_storage< T, N >;

      storage_type d;
    };

    template< typename T >
    using constant_vec3 = constant_vec< T, 3 >;

    template< typename T >
    using constant_vec4 = constant_vec< T, 4 >;

    using constant_vec4d = constant_vec4< double >;

    using constant_vec3d = constant_vec3< double >;

  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_CONSTANT_VEC_HPP
