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
#ifndef INC_BVH_MATH_OPS_VEC_BASE_OPS_HPP
#define INC_BVH_MATH_OPS_VEC_BASE_OPS_HPP

#include <cmath>
#include <type_traits>
#include <utility>

#include "../compiler.hpp"
#include "../storage/vec_storage.hpp"
#include "vec_generic_ops.hpp"
#include "../vector_traits.hpp"

namespace bvh
{
  namespace m
  {
    template< typename T, unsigned N >
    struct vec;

    namespace ops
    {
      namespace detail
      {

        template< typename T, unsigned N >
        using storage_type = ::bvh::m::detail::vec_storage< T, N >;

        template< typename T, std::size_t I >
        using expand = T;

        template< template< typename, unsigned > class Vec, typename T, typename S >
        struct storage_set_op_impl;

        template< template< typename, unsigned > class Vec, typename T, std::size_t... Indices >
        struct storage_set_op_impl< Vec, T, std::index_sequence< Indices... > >
        {
          using result_type = typename Vec< T, sizeof...( Indices ) >::storage_type;
#if defined( __GNUC__ ) && !defined( __NVCC__ )

          template< std::size_t TIndex >
          static constexpr BVH_INLINE auto
          assign( result_type &_res, T _arg )
          {
            _res[TIndex] = _arg;
          }

          template< std::size_t TIndex, std::size_t... TIndices >
          static constexpr BVH_INLINE auto
          assign( result_type &_res, T _arg, expand< T, TIndices >... _args )
          {
            _res[TIndex] = _arg;
            assign< TIndices... >( _res, _args... );
          }

          static constexpr BVH_INLINE auto
          execute( expand< T, Indices >... _args )
          {
            result_type ret;
            assign< Indices... >( ret, _args... );

            return ret;
          }

#else
          // Unsafe version for Intel and NVCC since they do not compile C++ properly
          template< typename Arg, typename... Args >
          static constexpr BVH_INLINE void
          assign( result_type &_res, std::size_t _index, Arg &&_arg, Args &&... _args )
          {
            _res[_index] = _arg;
            assign( _res, _index + 1, std::forward< Args >( _args )... );
          }

          template< typename Arg >
          static constexpr BVH_INLINE void
          assign( result_type &_res, std::size_t _index, Arg &&_arg )
          {
            _res[_index] = _arg;
          }

          template< typename... Args >
          static constexpr BVH_INLINE auto
          execute( Args &&... _args )
          {
            result_type ret;
            assign( ret, 0, std::forward< Args >( _args )... );
            return ret;
          }
#endif
        };
      } // namespace detail

      template< typename T, unsigned N >
      using storage_type = ::bvh::m::detail::vec_storage< T, N >;

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct init_undefined_storage_op
      {
        using result_type = T;

        constexpr static BVH_INLINE auto
        execute()
        {
          typename Vec< result_type, N >::storage_type ret{};

          return ret;
        }
      };

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct init_zero_storage_op
      {
        using result_type = T;

        constexpr static BVH_INLINE auto
        execute()
        {
          typename Vec< result_type, N >::storage_type ret{};

          for ( unsigned i = 0; i < N; ++i )
            ret[i] = 0.0;

          return ret;
        }
      };

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct init_set1_storage_op
      {
        using result_type = T;

        constexpr static BVH_INLINE auto
        execute( T _scalar )
        {
          typename Vec< result_type, N >::storage_type ret{};

          for ( unsigned i = 0; i < N; ++i )
            ret[i] = _scalar;

          return ret;
        }
      };

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      using init_set_storage_op = detail::storage_set_op_impl< Vec, T, std::make_index_sequence< N > >;


      template< template< typename, unsigned > class Vec, typename T, typename U, unsigned N >
      struct init_load_storage_op
      {
        using result_type = T;

        constexpr static auto BVH_INLINE
        execute( const U *_data )
        {
          typename Vec< result_type, N >::storage_type ret{};
          for ( unsigned i = 0; i < N; ++i )
            ret[i] = _data[i];
          return ret;
        }
      };

      template< typename Derived, typename Vec1, typename Vec2 >
      struct binary_component_op
      {
        static_assert( is_vector_type< Vec1 >::value && is_vector_type< Vec2 >::value,
                       "both operands must be vector types" );
        static_assert( vector_traits< Vec1 >::num_components == vector_traits< Vec2 >::num_components,
                       "both operands must have the same number of components" );
        using type1 = typename vector_traits< Vec1 >::component_type;
        using type2 = typename vector_traits< Vec2 >::component_type;
        static constexpr unsigned num_components = vector_traits< Vec1 >::num_components;
      };

      template< typename Vec1, typename Vec2, typename Enabled = void >
      struct is_valid_binary_component_op : std::false_type
      {
      };

      template< typename Vec1, typename Vec2 >
      struct is_valid_binary_component_op< Vec1, Vec2, std::enable_if_t<
          is_vector_type< Vec1 >::value
          && is_vector_type< Vec2 >::value
          && ( vector_traits< Vec1 >::num_components == vector_traits< Vec2 >::num_components )
      >
      > : std::true_type
      {
      };

      template< typename Derived, typename Vec, typename Scalar >
      struct binary_scalar_op
      {
        static_assert( is_vector_type< Vec >::value && std::is_scalar< Scalar >::value,
                       "operands must be vector and scalar" );
        using vector_component_type = typename vector_traits< Vec >::component_type;
        using scalar_type = Scalar;
        static constexpr unsigned num_components = vector_traits< Vec >::num_components;
      };

      template< typename Vec, typename Scalar, typename Enabled = void >
      struct is_valid_binary_scalar_op : std::false_type
      {
      };

      template< typename Vec, typename Scalar >
      struct is_valid_binary_scalar_op< Vec, Scalar, std::enable_if_t<
          is_vector_type< Vec >::value
          && std::is_scalar< Scalar >::value
      >
      > : std::true_type
      {
      };


      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct haddv_op
      {
        using result_type = T;

        constexpr static auto BVH_INLINE
        execute( const Vec< T, N > &_vec )
        {
          Vec< result_type, N > ret;
          result_type sum = result_type{ 0 };
          for ( unsigned i = 0; i < N; ++i )
            sum += _vec[i];

          for ( unsigned i = 0; i < N; ++i )
            ret[i] = sum;

          return ret;
        }
      };

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct sqrt_op
      {
        using result_type = T;

        constexpr static auto BVH_INLINE
        execute( const Vec< T, N > &_vec )
        {
          Vec< result_type, N > ret;
          for ( unsigned i = 0; i < N; ++i )
            ret[i] = sqrt( _vec[i] );

          return ret;
        }
      };

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct floor_op
      {
        using result_type = T;

        constexpr static auto BVH_INLINE
        execute( const Vec< T, N > &_vec )
        {
          Vec< result_type, N > ret;
          for ( unsigned i = 0; i < N; ++i )
            ret[i] = floor( _vec[i] );

          return ret;
        }
      };

      template< template< typename, unsigned > class Vec, typename T, unsigned N >
      struct ceil_op
      {
        using result_type = T;

        constexpr static auto BVH_INLINE
        execute( const Vec< T, N > &_vec )
        {
          Vec< result_type, N > ret;
          for ( unsigned i = 0; i < N; ++i )
            ret[i] = ceil( _vec[i] );

          return ret;
        }
      };

      template< typename T, unsigned N >
      struct truthiness_op
      {
        using result_type = bool;

        static auto BVH_INLINE
        execute( const vec< T, N > &_vec )
        {
          for ( unsigned i = 0; i < N; ++i )
          {
            if ( !sign( _vec[i] ) ) return false;
          }

          return true;
        }
      };

      template< unsigned N >
      struct truthiness_op< bool, N >
      {
        using result_type = bool;

        static auto BVH_INLINE
        execute( const vec< bool, N > &_vec )
        {
          for ( unsigned i = 0; i < N; ++i )
          {
            if ( !_vec[i] ) return false;
          }

          return true;
        }
      };

      template< unsigned N >
      struct truthiness_op< std::uint32_t, N >
      {
        using result_type = bool;

        static auto BVH_INLINE
        execute( const vec< std::uint32_t, N > &_vec )
        {
          for ( unsigned i = 0; i < N; ++i )
          {
            if ( !_vec[i] ) return false;
          }

          return true;
        }
      };

      template< unsigned N >
      struct truthiness_op< std::uint64_t, N >
      {
        using result_type = bool;

        static auto BVH_INLINE
        execute( const vec< std::uint64_t, N > &_vec )
        {
          for ( unsigned i = 0; i < N; ++i )
          {
            if ( !_vec[i] ) return false;
          }

          return true;
        }
      };
    } // namespace ops
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_OPS_VEC_BASE_OPS_HPP
