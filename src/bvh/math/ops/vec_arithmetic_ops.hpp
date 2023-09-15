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
#ifndef INC_BVH_MATH_OPS_VEC_ARITHMETIC_OPS_HPP
#define INC_BVH_MATH_OPS_VEC_ARITHMETIC_OPS_HPP

#include "../vector_traits.hpp"
#include "vec_base_ops.hpp"
#include "../compiler.hpp"

namespace bvh
{
  namespace m
  {
    namespace ops
    {
      template< typename Derived, typename Vec1, typename Vec2 >
      struct binary_arithmetic_op : binary_component_op< Derived, Vec1, Vec2 >
      {
        using base = binary_component_op< Derived, Vec1, Vec2 >;
        using typename base::type1;
        using typename base::type2;
        using base::num_components;
        using derived_type = Derived;

        BVH_INLINE static constexpr auto
        execute( const Vec1 &_lhs, const Vec2 &_rhs )
        {
          using result_component_type = decltype( derived_type::component_op( std::declval< type1 >(),
                                                                         std::declval< type2 >() ) );
          using result_type = typename ::bvh::m::detail::vec_promotion_type< Vec1, Vec2, result_component_type >::type;

          result_type ret;
          for ( unsigned i = 0; i < num_components; ++i )
          {
            ret[i] = derived_type::component_op( _lhs[i], _rhs[i] );
          }

          return ret;
        }

        BVH_INLINE static constexpr Vec1 &
        execute_self( Vec1 &_lhs, const Vec2 &_rhs )
        {
          for ( unsigned i = 0; i < num_components; ++i )
          {
            derived_type::component_modifying_op( _lhs[i], _rhs[i] );
          }

          return _lhs;
        }
      };

      template< typename Derived, typename Vec, typename Scalar >
      struct binary_scalar_arithmetic_op : binary_scalar_op< Derived, Vec, Scalar >
      {
        using base = binary_scalar_op< Derived, Vec, Scalar >;
        using typename base::vector_component_type;
        using typename base::scalar_type;
        using base::num_components;
        using derived_type = Derived;

        BVH_INLINE static constexpr auto
        execute( const Vec &_lhs, Scalar _s )
        {
          using result_component_type = decltype( derived_type::component_op( std::declval< vector_component_type >(),
                                                                         std::declval< scalar_type >() ) );
          using result_type = typename ::bvh::m::detail::build_vec_type< Vec, result_component_type >::type;

          result_type ret;
          for ( unsigned i = 0; i < num_components; ++i )
          {
            ret[i] = derived_type::component_op( _lhs[i], _s );
          }

          return ret;
        }

        static constexpr auto BVH_INLINE
        execute( Scalar _s, const Vec &_rhs )
        {
          using result_component_type = decltype( derived_type::component_op( std::declval< scalar_type >(),
                                                                         std::declval< vector_component_type >() ) );
          using result_type = typename ::bvh::m::detail::build_vec_type< Vec, result_component_type >::type;

          result_type ret;
          for ( unsigned i = 0; i < num_components; ++i )
          {
            ret[i] = derived_type::component_op( _s, _rhs[i] );
          }

          return ret;
        }

        BVH_INLINE static constexpr Vec &
        execute_self( Vec &_lhs, Scalar _s )
        {

          for ( unsigned i = 0; i < num_components; ++i )
          {
            derived_type::component_modifying_op( _lhs[i], _s );
          }

          return _lhs;
        }
      };


      template< typename Vec1, typename Vec2 >
      struct add_op : binary_arithmetic_op< add_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_arithmetic_op< add_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr auto
        component_op( typename base::type1 _l, typename base::type2 _r )
        {
          return _l + _r;
        }

        BVH_INLINE static constexpr void
        component_modifying_op( typename base::type1 &_l, typename base::type2 _r )
        {
          _l += _r;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct sub_op : binary_arithmetic_op< sub_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_arithmetic_op< sub_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr auto component_op( typename base::type1 _l, typename base::type2 _r )
        {
          return _l - _r;
        }

        BVH_INLINE static constexpr void
        component_modifying_op( typename base::type1 &_l, typename base::type2 _r )
        {
          _l -= _r;
        }
      };

      template< typename Vec1, typename Vec2, typename Enabled = void >
      struct mul_op : binary_arithmetic_op< mul_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_arithmetic_op< mul_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr auto component_op( typename base::type1 _l, typename base::type2 _r )
        {
          return _l * _r;
        }

        BVH_INLINE static constexpr void
        component_modifying_op( typename base::type1 &_l, typename base::type2 _r )
        {
          _l *= _r;
        }
      };

      template< typename Vec, typename Scalar >
      struct mul_op< Vec, Scalar, std::enable_if_t< std::is_scalar< Scalar >::value > > : binary_scalar_arithmetic_op<
          mul_op< Vec, Scalar >, Vec, Scalar >
      {
        using base = binary_scalar_arithmetic_op< mul_op, Vec, Scalar >;

        BVH_INLINE static constexpr auto
        component_op( typename base::vector_component_type _l, typename base::scalar_type _r )
        {
          return _l * _r;
        }

        BVH_INLINE static constexpr auto
        component_modifying_op( typename base::vector_component_type &_l, typename base::scalar_type _s )
        {
          _l *= _s;
        }
      };

      template< typename Scalar, typename Vec >
      struct mul_op< Scalar, Vec, std::enable_if_t< std::is_scalar< Scalar >::value > > : binary_scalar_arithmetic_op<
          mul_op< Scalar, Vec >, Vec, Scalar >
      {
        using base = binary_scalar_arithmetic_op< mul_op, Vec, Scalar >;

        BVH_INLINE static constexpr auto
        component_op( typename base::scalar_type _l, typename base::vector_component_type _r )
        {
          return _l * _r;
        }
      };

      template< typename Vec1, typename Vec2, typename Enabled = void >
      struct div_op : binary_arithmetic_op< div_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_arithmetic_op< div_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr auto component_op( typename base::type1 _l, typename base::type2 _r )
        {
          return _l / _r;
        }

        BVH_INLINE static constexpr void
        component_modifying_op( typename base::type1 &_l, typename base::type2 _r )
        {
          _l /= _r;
        }
      };

      template< typename Vec, typename Scalar >
      struct div_op< Vec, Scalar, std::enable_if_t< std::is_scalar< Scalar >::value > > : binary_scalar_arithmetic_op<
          div_op< Vec, Scalar >, Vec, Scalar >
      {
        using base = binary_scalar_arithmetic_op< div_op, Vec, Scalar >;

        BVH_INLINE static constexpr auto
        component_op( typename base::vector_component_type _l, typename base::scalar_type _r )
        {
          return _l / _r;
        }

        static constexpr BVH_INLINE auto
        component_modifying_op( typename base::vector_component_type &_l, typename base::scalar_type _s )
        {
          _l /= _s;
        }
      };

      template< typename Scalar, typename Vec >
      struct div_op< Scalar, Vec, std::enable_if_t< std::is_scalar< Scalar >::value > > : binary_scalar_arithmetic_op<
          div_op< Scalar, Vec >, Vec, Scalar >
      {
        using base = binary_scalar_arithmetic_op< div_op, Vec, Scalar >;

        BVH_INLINE static constexpr auto
        component_op( typename base::scalar_type _l, typename base::vector_component_type _r )
        {
          return _l / _r;
        }
      };
    }
  }
}

#endif  // INC_BVH_MATH_OPS_VEC_ARITHMETIC_OPS_HPP
