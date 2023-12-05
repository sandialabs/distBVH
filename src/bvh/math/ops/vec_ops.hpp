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
#ifndef INC_BVH_MATH_OPS_VEC_OPS_HPP
#define INC_BVH_MATH_OPS_VEC_OPS_HPP

#include "avx2_vec_ops.hpp"
#include "constexpr_arithmetic.hpp"
#include "vec_base_ops.hpp"
#include "vec_arithmetic_ops.hpp"
#include "vec_bool_ops.hpp"

namespace bvh
{
  namespace m
  {
    /**
     * \addtogroup vaopt Vector Arithmetic Operations
     * @{
     */

    template< typename Vec1, typename Vec2,
              typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE auto
    operator+( const Vec1 &_lhs, const Vec2 &_rhs )
    {
      return ops::add_op< Vec1, Vec2 >::execute( _lhs, _rhs );
    }

    template< typename SelfType, typename OtherType,
              typename = std::enable_if_t<
                  ops::is_valid_binary_component_op< SelfType, OtherType >::value
              > >
    BVH_INLINE auto &operator+=( SelfType &_self, const OtherType &_other )
    {
      return ops::add_op< SelfType, OtherType >::execute_self( _self, _other );
    }

    template< typename Vec1, typename Vec2,
              typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE auto
    operator-( const Vec1 &_lhs, const Vec2 &_rhs )
    {
      return ops::sub_op< Vec1, Vec2 >::execute( _lhs, _rhs );
    }

    template< typename SelfType, typename OtherType,
              typename = std::enable_if_t<
                  ops::is_valid_binary_component_op< SelfType, OtherType >::value
              > >
    BVH_INLINE auto &operator-=( SelfType &_self, const OtherType &_other )
    {
      return ops::sub_op< SelfType, OtherType >::execute_self( _self, _other );
    }

    template< typename Vec1, typename Vec2,
              typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE auto
    operator*( const Vec1 &_lhs, const Vec2 &_rhs )
    {
      return ops::mul_op< Vec1, Vec2 >::execute( _lhs, _rhs );
    }

    template< typename SelfType, typename OtherType,
              typename = std::enable_if_t<
                  ops::is_valid_binary_component_op< SelfType, OtherType >::value
              > >
    BVH_INLINE auto &operator*=( SelfType &_self, const OtherType &_other )
    {
      return ops::mul_op< SelfType, OtherType >::execute_self( _self, _other );
    }

    template< typename Vec1, typename Vec2,
              typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE auto
    operator/( const Vec1 &_lhs, const Vec2 &_rhs )
    {
      return ops::div_op< Vec1, Vec2 >::execute( _lhs, _rhs );
    }

    template< typename SelfType, typename OtherType,
              typename = std::enable_if_t<
                  ops::is_valid_binary_component_op< SelfType, OtherType >::value
              > >
    BVH_INLINE auto &operator/=( SelfType &_self, const OtherType &_other )
    {
      return ops::div_op< SelfType, OtherType >::execute_self( _self, _other );
    }

    // Scalar ops

    template< typename Vec, typename Scalar,
              typename = std::enable_if_t< ops::is_valid_binary_scalar_op< Vec, Scalar >::value > >
    constexpr BVH_INLINE auto operator*( const Vec &_rhs, Scalar _scalar )
    {
      return ops::mul_op< Vec, Scalar >::execute( _rhs, _scalar );
    }

    template< typename Scalar, typename Vec,
              typename = std::enable_if_t< ops::is_valid_binary_scalar_op< Vec, Scalar >::value > >
    constexpr BVH_INLINE auto operator*( Scalar _scalar, const Vec &_lhs )
    {
      return ops::mul_op< Scalar, Vec >::execute( _scalar, _lhs );
    }

    template< typename SelfType, typename Scalar,
              typename = std::enable_if_t<
                  ops::is_valid_binary_scalar_op< SelfType, Scalar >::value
              > >
    BVH_INLINE auto &operator*=( SelfType &_self, Scalar _other )
    {
      return ops::mul_op< SelfType, Scalar >::execute_self( _self, _other );
    }

    template< typename Vec, typename Scalar,
              typename = std::enable_if_t< ops::is_valid_binary_scalar_op< Vec, Scalar >::value > >
    constexpr BVH_INLINE auto operator/( const Vec &_rhs, Scalar _scalar )
    {
      return ops::div_op< Vec, Scalar >::execute( _rhs, _scalar );
    }

    template< typename Scalar, typename Vec,
              typename = std::enable_if_t< ops::is_valid_binary_scalar_op< Vec, Scalar >::value > >
    constexpr BVH_INLINE auto operator/( Scalar _scalar, const Vec &_lhs )
    {
      return ops::div_op< Scalar, Vec >::execute( _scalar, _lhs );
    }

    template< typename SelfType, typename Scalar,
              typename = std::enable_if_t<
                  ops::is_valid_binary_scalar_op< SelfType, Scalar >::value
              > >
    BVH_INLINE auto &operator/=( SelfType &_self, Scalar _other )
    {
      return ops::div_op< SelfType, Scalar >::execute_self( _self, _other );
    }

    /**
     * @}
     */


    /**
     * \addtogroup vcomp Vector Comparison Operations
     * @{
     */

    template< typename Vec1, typename Vec2,
              typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE bool
    operator==( const Vec1 &_lhs, const Vec2 &_rhs )
    {
      return ops::equals_op< Vec1, Vec2 >::execute( _lhs, _rhs );
    }

    template< typename Vec1, typename Vec2,
        typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE auto
    approx_equals( const Vec1 &_lhs, const Vec2 &_rhs,
      common_component_t< Vec1, Vec2 > _eps = epsilon_value< common_component_t< Vec1, Vec2 > > )
    {
      auto ret = vec< bool, vector_traits< Vec1 >::num_components >::zeros();
      for ( std::size_t i = 0; i < vector_traits< Vec1 >::num_components; ++i )
        ret[i] = approx_equals( _lhs[i], _rhs[i], _eps );
      return ret;
    }

    template< typename Vec1, typename Vec2,
              typename = std::enable_if_t< ops::is_valid_binary_component_op< Vec1, Vec2 >::value > >
    constexpr BVH_INLINE bool
    operator!=( const Vec1 &_lhs, const Vec2 &_rhs )
    {
      return ops::equals_op< Vec1, Vec2 >::execute( _lhs, _rhs );
    }

    /**
     * @}
     */
  }
}

#endif // INC_BVH_MATH_OPS_VEC_OPS_HPP
