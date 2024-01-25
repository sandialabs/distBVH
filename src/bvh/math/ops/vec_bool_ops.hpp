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
#ifndef INC_BVH_MATH_OPS_VEC_BOOL_OPS_HPP
#define INC_BVH_MATH_OPS_VEC_BOOL_OPS_HPP

#include "../vector_traits.hpp"
#include "vec_base_ops.hpp"
#include "../promotion.hpp"
#include "../../util/bits.hpp"

namespace bvh
{
  namespace m
  {
    namespace ops
    {
      template< typename Derived, typename Vec1, typename Vec2 >
      struct binary_comparison_op : binary_component_op< Derived, Vec1, Vec2 >
      {
        using base = binary_component_op< Derived, Vec1, Vec2 >;
        using typename base::type1;
        using typename base::type2;
        using base::num_components;
        using derived_type = Derived;

        BVH_INLINE static constexpr auto
        execute( const Vec1 &_lhs, const Vec2 &_rhs )
        {
          using result_component_type = bool;
          using result_type = vec< bool, num_components >;

          result_type ret;
          for ( unsigned i = 0; i < num_components; ++i )
          {
            ret[i] = derived_type::component_comparison( _lhs[i], _rhs[i] ) ? fill< result_component_type, true >() : fill< result_component_type, false >();
          }

          return ret;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct equals_op : binary_comparison_op< equals_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_comparison_op< equals_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr bool
        component_comparison( typename base::type1 _l, typename base::type2 _r )
        {
          return _l == _r;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct not_equals_op : binary_comparison_op< not_equals_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_comparison_op< not_equals_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr bool
        component_comparison( typename base::type1 _l, typename base::type2 _r )
        {
          return _l != _r;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct less_equals_op : binary_comparison_op< less_equals_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_comparison_op< less_equals_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr bool
        component_comparison( typename base::type1 _l, typename base::type2 _r )
        {
          return _l <= _r;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct greater_equals_op : binary_comparison_op< greater_equals_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_comparison_op< greater_equals_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr bool
        component_comparison( typename base::type1 _l, typename base::type2 _r )
        {
          return _l >= _r;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct less_op : binary_comparison_op< less_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_comparison_op< less_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr bool
        component_comparison( typename base::type1 _l, typename base::type2 _r )
        {
          return _l < _r;
        }
      };

      template< typename Vec1, typename Vec2 >
      struct greater_op : binary_comparison_op< greater_op< Vec1, Vec2 >, Vec1, Vec2 >
      {
        using base = binary_comparison_op< greater_op, Vec1, Vec2 >;

        BVH_INLINE static constexpr bool
        component_comparison( typename base::type1 _l, typename base::type2 _r )
        {
          return _l > _r;
        }
      };
    }
  }
}

#endif  // INC_BVH_MATH_OPS_VEC_BOOL_OPS_HPP
