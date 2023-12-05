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
#ifndef INC_BVH_MATH_VEC_HPP
#define INC_BVH_MATH_VEC_HPP

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "common.hpp"
#include "../util/attributes.hpp"
#include "../util/kokkos.hpp"
#include "named_access.hpp"
#include "ops/vec_ops.hpp"

namespace bvh
{
  namespace m
  {
    /**
     * A vector utility class.
     *
     * Defines various geometric operations.
     *
     * @tparam T    The type of an element in a vector. Should be a primitive numerical type.
     * @tparam N    The number of elements in the vector.
     */
    template< typename T, unsigned N >
    struct vec : public vec_named_access< vec, T, N >
    {
      /**
       * Construct the vector with storage initialized to 0
       */
      KOKKOS_INLINE_FUNCTION
      vec()
          : d( ops::init_zero_storage_op< vec, T, N >::execute() )
      {}

      /**
       * Construct the vector from another vector type vec< U, N >.
       *
       * Only participates in overload resolution if `std::is_same_v< T, U >` is false and
       * `std::is_convertible_v< U, T >` is true.
       *
       * @tparam U      The type of the element in the other vector
       * @param _other  The other vector of type vec< U, N >
       */
      template< typename U,
                typename = std::enable_if_t< !std::is_same< T, U >::value && std::is_convertible< U, T >::value > >
      KOKKOS_INLINE_FUNCTION
      vec( const vec< U, N > &_other )
      {
        for ( unsigned i = 0; i < N; ++i )
          d[i] = _other[i];
      }

      /**
       * Construct the vector from a scalar.
       *
       * Set each element of the vector to the scalar value.
       *
       * @param _s  The value to set each element to
       */
      explicit KOKKOS_INLINE_FUNCTION
      vec( T _s )
          : d( ops::init_set1_storage_op< vec, T, N >::execute( _s ) )
      {}

      /**
       * Construct a vector from vector storage.
       *
       * @param _d  The vector storage.
       */
      explicit KOKKOS_INLINE_FUNCTION
      vec( const detail::vec_storage< T, N > &_d )
          : d( _d )
      {}

      /**
       * Element-wise construction of a vector.
       *
       * Constructs the elements of a vector in order from left to right. Only participates in overload resolution if
       * `sizeof...( Args ) == N`.
       *
       * @param _args   The N arguments that initialize the vector.
       */
      template< typename... Args, typename = std::enable_if_t< sizeof...( Args ) == N > >
      KOKKOS_INLINE_FUNCTION
      vec( Args &&... _args )
          : d( ops::init_set_storage_op< vec, T, N >::execute( std::forward< Args >( _args )... ) )
      {}


      /**
       * Implicit converting constructor from a constant_vec.
       *
       * Only participates in overload resolution if `std::is_convertible_v< U, T >` is true.
       *
       * @tparam U      The type of the constant_vec's element
       * @param _cvec   The constant_vec
       */
      template< typename U, typename = std::enable_if_t< std::is_convertible< U, T >::value > >
      KOKKOS_INLINE_FUNCTION
      vec( const constant_vec< U, N > &_cvec )
          : d( ops::init_load_storage_op< vec, T, U, N >::execute( _cvec.d.d ) )
      {}

      /**
       * Create a vec that has uninitialized elements.
       *
       * Does not perform any initialization on elements.
       *
       * @return The uninitialized vector
       */
      static KOKKOS_INLINE_FUNCTION vec
      undefined()
      {
        return vec( ops::init_undefined_storage_op< vec, T, N >::execute() );
      }

      /**
       * Create a vec that has zero-initialized elements.
       *
       * @return The zero-initialized vector
       */
      static KOKKOS_INLINE_FUNCTION vec
      zeros()
      {
        return vec();
      }

      /**
       * Create a vec that has its elements initialized by a scalar value.
       *
       * @param _s  The value to initialize all elements
       * @return    The initialized vector
       */
      static KOKKOS_INLINE_FUNCTION vec
      set1( T _s )
      {
        return vec( _s );
      }

      /**
       * Create a vec that has its elements initialized by the value 1.
       *
       * Identical to `set1( 1.0 )`
       *
       * @return    The initialized vector
       */
      static KOKKOS_INLINE_FUNCTION vec
      ones()
      {
        return set1( 1.0 );
      }

      KOKKOS_INLINE_FUNCTION T &operator[]( unsigned _n )
      { return d[_n]; }

      KOKKOS_INLINE_FUNCTION const T &operator[]( unsigned _n ) const
      { return d[_n]; }

      KOKKOS_INLINE_FUNCTION operator bool() const noexcept
      { return ops::truthiness_op< T, N >::execute( *this ); }

      friend std::ostream &
      operator<<( std::ostream &_os, vec _v )
      {
        _os << "< " << _v[0];
        for ( unsigned i = 1; i < N; ++i )
          _os << ", " << _v[i];

        _os << " >";

        return _os;
      }

      using storage_type = detail::vec_storage< T, N >;

      storage_type d;
    };

    template< typename T >
    using vec3 = vec< T, 3 >;
    using vec3d = vec3< double >;

    // Horizontal ops

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    haddv( const Vec< T, N > &_vec )
    {
      return ops::haddv_op< Vec, T, N >::execute( _vec );
    }

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    hadd( const Vec< T, N > &_vec )
    {
      return haddv( _vec )[0];
    }

    template< template< typename, unsigned > class Vec, typename T, typename U, unsigned N >
    constexpr BVH_INLINE auto
    dotv( const Vec< T, N > &_lhs, const Vec< U, N > &_rhs )
    {
      return haddv( _lhs * _rhs );
    }

    template< template< typename, unsigned > class Vec, typename T, typename U, unsigned N >
    constexpr BVH_INLINE auto
    dot( const Vec< T, N > &_lhs, const Vec< U, N > &_rhs )
    {
      return hadd( _lhs * _rhs );
    }

    template< template< typename, unsigned > class Vec, typename T, typename U >
    constexpr BVH_INLINE auto
    cross( const Vec< T, 3 > &_lhs, const Vec< U, 3 > &_rhs )
    {
      const auto t0 = Vec< T, 3 >{ _lhs[1], _lhs[2], _lhs[0] };
      const auto t1 = Vec< U, 3 >{ _rhs[2], _rhs[0], _rhs[1] };
      const auto t2 = t0 * _rhs;
      const auto lhs = t0 * t1;
      const auto rhs = decltype( t2 ){ t2[1], t2[2], t2[0] };
      return lhs - rhs;
    }

    // Unary ops

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    length2( const Vec< T, N > &_vec )
    {
      return dot( _vec, _vec );
    }

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    length( const Vec< T, N > &_vec )
    {
      return sqrt( dot( _vec, _vec ) );
    }

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    sqrt( const Vec< T, N > &_vec )
    {
      return ops::sqrt_op< Vec, T, N >::execute( _vec );
    }

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    normal( const Vec< T, N > &_vec )
    {
      return _vec / sqrt( dotv( _vec, _vec ) );
    }

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    floor( const Vec< T, N > &_vec )
    {
      return ops::floor_op< Vec, T, N >::execute( _vec );
    }

    template< template< typename, unsigned > class Vec, typename T, unsigned N >
    constexpr BVH_INLINE auto
    ceil( const Vec< T, N > &_vec )
    {
      return ops::ceil_op< Vec, T, N >::execute( _vec );
    }

    // Metric ops

    template< template< typename, unsigned > class Vec, typename T, typename U, unsigned N >
    constexpr BVH_INLINE auto
    distance2( const Vec< T, N > &_lhs, const Vec< U, N > &_rhs )
    {
      return length2( _rhs - _lhs );
    }

    template< template< typename, unsigned > class Vec, typename T, typename U, unsigned N >
    constexpr BVH_INLINE auto
    distance( const Vec< T, N > &_lhs, const Vec< U, N > &_rhs )
    {
      return length( _rhs - _lhs );
    }

    // specialization for vectors
    namespace detail
    {
      template< typename T, unsigned N >
      struct clamp_impl< vec< T, N > >
      {
        static BVH_INLINE vec< T, N >
        clamp( const vec< T, N > &_v, const vec< T, N > &_min, const vec< T, N > &_max )
        {
          vec< T, N > ret;
          for ( std::size_t i = 0; i < N; ++i )
          {
            ret[i] = clamp_impl< T >::clamp( _v[i], _min[i], _max[i] );
          }

          return ret;
        }
      };
    } // namespace detail
  } // namespace m
} // namespace bvh

#endif // INC_BVH_MATH_VEC_HPP
