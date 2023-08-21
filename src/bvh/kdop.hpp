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
#ifndef INC_BVH_KDOP_HPP
#define INC_BVH_KDOP_HPP

#include "math/vec.hpp"
#include "math/constant_vec.hpp"
#include "math/common.hpp"
#include <algorithm>
#include <cmath>
#include "range.hpp"
#include "util/array.hpp"
#include "util/attributes.hpp"
#include "util/kokkos.hpp"
#include "iterators/transform_iterator.hpp"

namespace bvh
{
  /**
   *  An upper-bound exclusive range along an axis. A point is in an extent if it is greater than or equal to the
   *  minimum and less than the maximum.
   *
   *  \tparam T the arithmetic type to use.
   */
  template< typename T >
  struct extent
  {
    T min = std::numeric_limits< T >::max(); ///< The lower bound of an extent.
    T max = std::numeric_limits< T >::lowest(); ///< The upper bound of an extent.

    /**
     *  The length of an extent.
     *
     *  \return the length of the extent.
     */
    constexpr BVH_INLINE T length() const noexcept
    {
      return max - min;
    }

    /**
     * Get the normalized coordinates if the given coordinate within the extent.
     * \param _coord the coordinate to normalize
     * \return the normalized coordinate
     */
    constexpr BVH_INLINE T get_normalized_coord( T _coord ) const noexcept
    {
      const auto inv_fac = T{ 1 } / length();
      return ( _coord - min ) * inv_fac;
    }

    /**
     * Test equality of two extents. Returns true iff `_lhs.min == _rhs.min` and `_lhs.max == _rhs.max`.
     * \param _lhs  the left extent
     * \param _rhs  the right extent
     * \return      whether the two extents are equal
     */
    friend BVH_INLINE bool operator==( const extent &_lhs, const extent &_rhs )
    {
      return ( _lhs.min == _rhs.min ) && ( _lhs.max == _rhs.max );
    }

    /**
     * Test inequality of two extents. Returns true if `_lhs.min != _rhs.min` or `_lhs.max != _rhs.max`.
     * \param _lhs  the left extent
     * \param _rhs  the right extent
     * \return      whether the two extents are not equal
     */
    friend BVH_INLINE bool operator!=( const extent &_lhs, const extent &_rhs )
    {
      return !( _lhs == _rhs );
    }

    friend BVH_INLINE bool approx_equals( const extent &_lhs, const extent &_rhs )
    {
      using namespace m;
      return ( approx_equals( _lhs.min, _rhs.min ) && ( approx_equals( _lhs.max, _rhs.max ) ) );
    }
  };

  /**
   *  Determine whether two extents overlap.
   *
   *  \tparam T     the arithmetic type of the extent.
   *  \param _lhs   the first extent.
   *  \param _rhs   the second extent.
   *  \return       whether the two extents overlap.
   */
  template< typename T >
  BVH_INLINE constexpr bool overlap( const extent< T > &_lhs,
                                 const extent< T > &_rhs ) noexcept
  {
    return _lhs.min < _rhs.max && _rhs.min < _lhs.max;
  }

  /**
   * Merge two extents. Returns a new extent that has a min equal to the minimum of `_lhs.min` and `_rhs.min` and has
   * a max equal to the maximum of `_lhs.max` and `_rhs.max`.
   *
   * \tparam T      the arithmetic type of the extent
   * \param _lhs    the first extent
   * \param _rhs    the second extent
   * \return        a new extent that contains both extents
   */
  template< typename T >
  BVH_INLINE constexpr extent< T > merge( const extent< T > &_lhs,
    const extent< T > &_rhs ) noexcept
  {
    extent< T > ret;
    ret.min = std::min( _lhs.min, _rhs.min );
    ret.max = std::max( _lhs.max, _rhs.max );

    return ret;
  }

  /**
   *  Base class for k-DOPs.
   *
   *  Implementors of different k-DOPs should statically inherit from this
   *  class by providing the Derived template parameter.
   *
   *  \tparam T           the arithmetic type of the domain.
   *  \tparam K           the number of axes in the discrete-oriented polytope.
   *  \tparam Derived     the derived class.
   */
  template< typename T, int K, typename Derived >
  struct kdop_base
  {
    static constexpr int k = K;
    static constexpr int num_axis = k / 2;
    using arithmetic_type = T;

    BVH_INLINE kdop_base() = default;
    BVH_INLINE ~kdop_base() = default;

    /**
     *  Create a k-DOP by merging a range of k-DOPs.
     *
     *  \tparam InputIterator     the iterator type.
     *  \param _begin             the beginning k-DOP range
     *  \param _end               the end of the k-DOP range
     */
    template< typename InputIterator >
    kdop_base( InputIterator _begin, InputIterator _end )
    {
      if ( std::distance( _begin, _end ) == 0 )
        return;

      for ( int axis = 0; axis < K / 2; ++axis )
      {
        auto iter = _begin;

        T min = ( *iter ).extents[axis].min;
        T max = ( *iter ).extents[axis].max;

        ++iter;

        // Iterate through the list of k-DOPs
        for ( ; iter != _end; ++iter )
        {
          min = m::min( iter->extents[axis].min, min );
          max = m::max( iter->extents[axis].max, max );
        }

        // global extents for the axis are min and max of the element extents
        extents[axis].min = min;
        extents[axis].max = max;
      }
    }

    /**
     *  Create a k-DOP by merging a range of k-DOPs.
     *
     *  \tparam InputIterator     the iterator type.
     *  \param _begin             the beginning k-DOP range
     *  \param _end               the end of the k-DOP range
     *  \return                   the constructed k-DOP.
     */
    template< typename InputIterator >
    static BVH_INLINE Derived from_kdops( InputIterator _begin, InputIterator _end )
    {
      return Derived( _begin, _end );
    }

    /**
     *  Project a vector along an axis.
     *
     *  \param _v     the vector to project.
     *  \param _axis  the axis vector to project on.
     *  \return       the distance along the axis normal of the projected vector.
     */
    template< typename Vec >
    static BVH_INLINE T project( const Vec &_v, const m::vec3< T > &ax)
    {
      return _v[0] * ax[0] + _v[1] * ax[1] + _v[2] * ax[2];
    }

    /**
     *  Create a k-DOP from a range of vertices.
     *
     *  \tparam InputIterator     the iterator type.
     *  \param _begin             the beginning of the vertices.
     *  \param _end               the end of the vertices.
     *  \param _epsilon           expand the k_DOP by this amount in each axis
     *  \return                   the constructed k-DOP.
     */
    template< typename InputIterator >
    static Derived from_vertices( InputIterator _begin, InputIterator _end, T _epsilon = T{ 0 } )
    {
      Derived ret;

      if ( _begin == _end )
        return ret;

      const auto &normal_list = Derived::normals();
      const auto &first = *_begin;
      auto beg = std::next( _begin );
      for ( int i = 0; i < num_axis; ++i )
      {
        auto proj = Derived::project( first, normal_list[i] );
        ret.extents[i].min = proj - _epsilon;
        ret.extents[i].max = proj + _epsilon;

        for ( auto iter = beg; iter != _end; ++iter )
        {
          proj = Derived::project( *iter, normal_list[i] );
          ret.extents[i].min = std::min( ret.extents[i].min, proj - _epsilon );
          ret.extents[i].max = std::max( ret.extents[i].max, proj + _epsilon );
        }
      }

      return ret;
    }

    /**
     *  Create a k-DOP bounding a sphere.
     *
     *  \param _center    the center of the sphere.
     *  \param _radius    the radius of the sphere.
     *  \return           the constructed k-DOP.
     */
    template< typename Vec >
    static Derived from_sphere( const Vec &_center, T _radius )
    {
      Derived ret;

      const auto &normal_list = Derived::normals();
      for ( int i = 0; i < K / 2; ++i )
      {
        T center = project( _center, normal_list[i] );

        ret.extents[i].min = center - _radius;
        ret.extents[i].max = center + _radius;
      }

      return ret;
    }

    /**
     *  Create a \f$k\f$-DOP bounding a sphere.
     *
     *  \param _x, _y, _z the center of the sphere.
     *  \param _radius    the radius of the sphere.
     *  \return           the constructed \f$k\f$-DOP.
     */
    static Derived from_sphere( T _x, T _y, T _z, T _radius )
    {
      return from_sphere( m::vec3< T >( _x, _y, _z ), _radius );
    }

    /**
     * Grow the kdop by the specified amount. Useful for dealing with degenerate
     * kdops.
     *
     * \param _amount The amount to grow.
     */
    void inflate( arithmetic_type _amount )
    {
      for ( int i = 0; i < K / 2; ++i )
      {
        extents[i].min -= _amount;
        extents[i].max += _amount;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void union_with( const Derived &_other )
    {
      for ( int i = 0; i < K / 2; ++i )
      {
        extents[i].min = m::min( extents[i].min, _other.extents[i].min );
        extents[i].max = m::max( extents[i].max, _other.extents[i].max );
      }
    }

    /**
     * Get the normalized coordinates of a point within the extents. Returns the factor \f$\alpha\f$ that can be used
     * to linearly interpolate between the min and max of the first `L` extents to get the original vector. Only
     * participates in overload resolution if `L` \f$\leq\f$ `K`.
     *
     * \tparam L        the number of axes to use for normalizing
     * \param _coords   the point to normalize
     * \return          the normalized coordinates of `_coords`
     */
    template< unsigned L, typename = std::enable_if_t< std::less_equal<>{}( L, K ) > >
    m::vec< T, L > get_normalized_coords( const m::vec< T, L > _coords ) const noexcept
    {
      m::vec< T, L > ret;

      for ( int i = 0; i < L; ++i )
      {
        ret[i] = extents[i].get_normalized_coord( _coords[i] );
      }

      return ret;
    }

    /**
     *  Get the longest axis of the \f$k\f$-DOP. The longest axis is the axis of the extent with the largest span.
     *
     *  \return the longest axis of the \f$k\f$-DOP.
     */
    int longest_axis() const
    {
      auto iter = std::max_element( extents.begin(), extents.end(),
                                []( const extent< T > &_lhs, const extent< T > &_rhs ) {
                                  return _lhs.length() < _rhs.length();
                                });

      return static_cast< int >( std::distance( extents.begin(), iter ) );
    }

    /**
     * Compute the centroid of the \f$k\f$-DOP. Runs in \f$O(k)\f$ time.
     *
     * \return  the computed centroid
     */
    BVH_INLINE m::vec3< T > centroid() const noexcept
    {
      using m::approx_equals;

      // Quick'n dirty approximation for the centroid
      // For each k/2 slab, find hyperplane equidistant to the extents of the slab.
      // For every combination of three planes, find the intersection. Take the average
      // of all intersection points to find the centroid.
      std::array< T, num_axis > center_planes;
      m::vec3< T > ret;
      const auto &normal_list = Derived::normals();
      for ( int i = 0; i < num_axis; ++i )
      {
        center_planes[i] = ( extents[i].min + extents[i].max ) * T{ 0.5 };
      }

      int count = 0;
      for ( int i = 0; i < num_axis - 2; ++i )
      {
        const auto &n1 = normal_list[i];

        for ( int j = i + 1; j < num_axis - 1; ++j )
        {
          const auto &n2 = normal_list[j];

          for ( int l = j + 1; l < num_axis; ++l )
          {
            const auto &n3 = normal_list[l];

            const auto n1xn2 = m::cross( n1, n2 );
            const auto n2xn3 = m::cross( n2, n3 );
            const auto n3xn1 = m::cross( n3, n1 );

            const auto det = dot( n1, n2xn3 );
            // Planes don't intersect
            // Note there are some numerical issues here, we are using approx_equals
            // To see whether det is within the epsilon, but this might cause further problems
            if ( approx_equals( det, 0.0 ) )
              continue;

            const auto isect = ( center_planes[i] * n2xn3 + center_planes[j] * n3xn1 + center_planes[l] * n1xn2 ) / det;
            ret += isect;
            ++count;
          }
        }
      }

      ret /= T( count );

      return ret;
    }

    KOKKOS_INLINE_FUNCTION
    std::enable_if_t< ( K >= 6 ), m::vec3< T > >
    cardinal_min() const noexcept
    {
      return m::vec3< T >{ extents[0].min, extents[1].min, extents[2].min };
    }

    KOKKOS_INLINE_FUNCTION
    std::enable_if_t< ( K >= 6 ), m::vec3< T > >
    cardinal_max() const noexcept
    {
      return m::vec3< T >{ extents[0].max, extents[1].max, extents[2].max };
    }

    /**
     * Expand the \f$k\f$-DOP by adding a point with the given radius. Modifies the \f$k\f$-DOP so that it encompasses
     * the new point with a radius of `_epsilon`.
     *
     * \param _point    the point to add to the \f$k\f$-DOP
     * \param _epsilon  the radius around the point to also include
     */
    BVH_INLINE void expand( const m::vec3< T > &_point, T _epsilon = m::epsilon ) noexcept
    {
      const auto &normal_list = Derived::normals();
      for ( int i = 0; i < K / 2; ++i )
      {
        auto projected = Derived::project( _point, normal_list[i] );
        extents[i].min = std::min( extents[i].min, projected - _epsilon );
        extents[i].max = std::max( extents[i].max, projected + _epsilon );
      }
    }

    /**
     *  Project a vector along an axis.
     *
     *  \param _v     the vector to project.
     *  \param _axis  the axis id to project on.
     *  \return       the distance along the axis normal of the projected vector.
     */
    template< typename Vec >
    static BVH_INLINE T project( const Vec &_v, int _axis )
    {
      const m::vec3< T > &ax = Derived::normals()[_axis];
      return _v[0] * ax[0] + _v[1] * ax[1] + _v[2] * ax[2];
    }

    /**
     * Write the kdop to a stream.
     *
     * \param os        the output stream
     * \param _kdop     the \f$k\f$-DOP to output
     * \return          the modified stream
     */
    friend std::ostream &operator<<( std::ostream &os, const kdop_base &_kdop )
    {
      os << K << "-dop: ";
      for ( auto &&e : _kdop.extents )
        os << "[" << e.min << ", " << e.max << "] ";

      return os;
    }

    /**
     * Test the equality of two \f$k\f$-DOPs. Returns true iff all the extents of `_lhs` are equal to the corresponding
     * extents in `_rhs`.
     *
     * \param _lhs  the first \f$k\f$-DOP to compare
     * \param _rhs  the second \f$k\f$-DOP to compare
     * \return      whether the two \f$k\f$-DOPs are equivalent.
     */
    friend BVH_INLINE bool operator==( const kdop_base &_lhs, const kdop_base &_rhs )
    {
      for ( int i = 0; i < K / 2; ++i )
      {
        if ( _lhs.extents[i] != _rhs.extents[i] )
          return false;
      }

      return true;
    }

    /**
     * Test the inequality of two \f$k\f$-DOPs. Returns true if any extent of `_lhs` is not equal to the corresponding
     * extent in `_rhs`.
     *
     * \param _lhs  the first \f$k\f$-DOP to compare
     * \param _rhs  the second \f$k\f$-DOP to compare
     * \return      whether the two \f$k\f$-DOPs are equivalent.
     */
    friend BVH_INLINE bool operator!=( const kdop_base &_lhs, const kdop_base &_rhs )
    {
      return !( _lhs == _rhs );
    }

    friend BVH_INLINE bool approx_equals( const kdop_base &_lhs, const kdop_base &_rhs )
    {
      for ( int i = 0; i < K / 2; ++i )
      {
        if ( !approx_equals( _lhs.extents[i], _rhs.extents[i] ) )
          return false;
      }

      return true;
    }


    array< extent< T >, K / 2 > extents;
  };

  /**
   *  Determine whether two \f$k\f$-DOPs of the same type overlap. This function should generally be called using
   *  automatic type deduction.
   *
   *  \param _lhs       the first k-DOP.
   *  \param _rhs       the second k-DOP.
   *  \return           whether the two k-DOPs overlap.
   */
  template< typename T, int K, typename Derived >
  BVH_INLINE bool overlap( const kdop_base< T, K, Derived > &_lhs,
                const kdop_base< T, K, Derived > &_rhs )
  {
    for ( int i = 0; i < K / 2; ++i )
    {
      if ( !overlap( _lhs.extents[i], _rhs.extents[i] ) )
        return false;
    }

    return true;
  }

  /**
   * Merge two \f$k\f$-DOPs. The resulting \f$k\f$-DOP's extents contain the extents of both parameters.
   *
   * \param _lhs        the first \f$k\f$-DOP
   * \param _rhs        the second \f$k\f$-DOP
   * \return            the merged \f$k\f$-DOP
   */
  template< typename T, int K, typename Derived >
  BVH_INLINE Derived merge( const kdop_base< T, K, Derived > &_lhs,
                                 const kdop_base< T, K, Derived > &_rhs )
  {
    Derived ret;
    for ( int i = 0; i < K / 2; ++i )
    {
      ret.extents[i] = merge( _lhs.extents[i], _rhs.extents[i] );
    }

    return ret;
  }

  /**
   *  A 6-DOP. The 6-DOP is equivalent to an axis-oriented bounding box (AABB). It has 3 slabs representing the volume of the
   *  k-DOP, one for each cardinal axis. See \ref kdops for more details on how the class is used.
   *
   *  \tparam T     The arithmetic type of the k-DOP.
   */
  template< typename T >
  struct dop_6 : public kdop_base< T, 6, dop_6< T > >
  {
    using typename kdop_base< T, 6, dop_6 >::arithmetic_type;
    using kdop_base< T, 6, dop_6 >::kdop_base;

    static constexpr BVH_INLINE array<m::constant_vec3<T>, 6> normals() {
      return array<m::constant_vec3<T>, 6> {{
          m::constant_vec3<T>(1., 0., 0.), m::constant_vec3<T>(0., 1., 0.),
              m::constant_vec3<T>(0., 0., 1.)
      }};
    }
  };

  /**
   * \f$6\f$-DOP with double precision extents.
   */
  using dop_6d = dop_6< double >;

  /**
   *  An 18-DOP. The 6-DOP defines a volume enclosed by 9 slabs along 9 axes. These axes include the cardinal directions on the
   *  in addition to diagonals 45 degrees between each of the axes and their negatives, making 18 normals in total. See
   *  \ref kdops for more details on how the class is used.
   *
   *  \tparam T     The arithmetic type of the k-DOP.
   */
  template< typename T >
  struct dop_18 : public kdop_base< T, 18, dop_18< T > >
  {
    using typename kdop_base< T, 18, dop_18 >::arithmetic_type;
    using kdop_base< T, 18, dop_18 >::kdop_base;

    static constexpr T over_root_2 = static_cast<T>(0.7071067811865475);
    static const array< m::vec3< T >, 18> normals;
  };

  template< typename T >
  const array< m::vec3< T >, 18 > dop_18< T >::normals = {{
                                                                 m::vec3< T >( 1., 0., 0. ),
                                                                 m::vec3< T >( 0., 1., 0. ),
                                                                 m::vec3< T >( 0., 0., 1. ),
                                                                 // Edges
                                                                 m::vec3< T >( over_root_2, over_root_2, 0. ),
                                                                 m::vec3< T >( over_root_2, 0., over_root_2 ),
                                                                 m::vec3< T >( 0., over_root_2, over_root_2 ),
                                                                 m::vec3< T >( over_root_2, -over_root_2, 0. ),
                                                                 m::vec3< T >( over_root_2, 0., -over_root_2 ),
                                                                 m::vec3< T >( 0., over_root_2, -over_root_2 ),
                                                                 // Negative cardinal
                                                                 m::vec3< T >( -1., 0., 0. ),
                                                                 m::vec3< T >( 0., -1., 0. ),
                                                                 m::vec3< T >( 0., 0., -1. ),
                                                                 // Negative edges
                                                                 m::vec3< T >( -over_root_2, -over_root_2, 0. ),
                                                                 m::vec3< T >( -over_root_2, 0., -over_root_2 ),
                                                                 m::vec3< T >( 0., -over_root_2, -over_root_2 ),
                                                                 m::vec3< T >( -over_root_2, over_root_2, 0. ),
                                                                 m::vec3< T >( -over_root_2, 0., over_root_2 ),
                                                                 m::vec3< T >( 0., -over_root_2, over_root_2 )
                                                               }};

  /**
   * \f$18\f$-DOP with double precision extents.
   */
  using dop_18d = dop_18< double >;

  /**
   *  A 26-DOP. The 26-DOP has 13 axes in 3D space in addition to the negatives of those
   *  axes. These include the 6 cartesian axes, the 12 axes corresponding to the
   *  edges of a cube, and the 8 axes corresponding to the corners of a cube. See
   *  \ref kdops for more details on how the class is used.
   *
   *  \tparam T     The arithmetic type of the k-DOP.
   */
  template< typename T >
  struct dop_26 : public kdop_base< T, 26, dop_26< T > >
  {
    using typename kdop_base< T, 26, dop_26 >::arithmetic_type;
    using kdop_base< T, 26, dop_26 >::kdop_base;

    static constexpr T one_over_root_2 = static_cast<T>(0.7071067811865475);
    static constexpr T one_over_root_3 = static_cast<T>(0.5773502691896258);

    static constexpr BVH_INLINE array< m::constant_vec3< T >, 26 > normals()
    {
      return array< m::constant_vec3< T >, 26 >{{
    m::constant_vec3< T >( 1., 0., 0. ),
    m::constant_vec3< T >( 0., 1., 0. ),
    m::constant_vec3< T >( 0., 0., 1. ),
// Corners
    m::constant_vec3< T >( one_over_root_3, one_over_root_3, one_over_root_3),
    m::constant_vec3< T >( one_over_root_3, -one_over_root_3, one_over_root_3),
    m::constant_vec3< T >( one_over_root_3, one_over_root_3, -one_over_root_3),
    m::constant_vec3< T >( one_over_root_3, -one_over_root_3, -one_over_root_3),
// Edges
    m::constant_vec3< T >( one_over_root_2, one_over_root_2,  0.0 ),
    m::constant_vec3< T >( one_over_root_2, 0.0, one_over_root_2),
    m::constant_vec3< T >( 0.0,  one_over_root_2, one_over_root_2),
    m::constant_vec3< T >( one_over_root_2, -one_over_root_2,  0.0 ),
    m::constant_vec3< T >( one_over_root_2,  0.0, -one_over_root_2),
    m::constant_vec3< T >( 0.0,  one_over_root_2, -one_over_root_2),
// Negative cardinal
    m::constant_vec3< T >( -1., 0., 0. ),
    m::constant_vec3< T >( 0., -1., 0. ),
    m::constant_vec3< T >( 0., 0., -1. ),
// Negative corners
    m::constant_vec3< T >( -one_over_root_3, -one_over_root_3, -one_over_root_3),
    m::constant_vec3< T >( -one_over_root_3, one_over_root_3, -one_over_root_3),
    m::constant_vec3< T >( -one_over_root_3, -one_over_root_3, one_over_root_3),
    m::constant_vec3< T >( -one_over_root_3, one_over_root_3, one_over_root_3),
// Negative edges
    m::constant_vec3< T >( -one_over_root_2, -one_over_root_2,  0.0),
    m::constant_vec3< T >( -one_over_root_2, 0.0, -one_over_root_2),
    m::constant_vec3< T >( 0.0, -one_over_root_2, -one_over_root_2),
    m::constant_vec3< T >( -one_over_root_2, one_over_root_2,  0.0),
    m::constant_vec3< T >( -one_over_root_2,  0.0, one_over_root_2),
    m::constant_vec3< T >( 0.0,  -one_over_root_2, one_over_root_2)
  }};
    }
  };

  /**
   * \f$26\f$-DOP with double precision extents.
   */
  using dop_26d = dop_26< double >;

  template< typename T, typename Enabled = void >
  struct is_kdop_type : std::false_type {};

  template< typename KDop >
  struct is_kdop_type< KDop, std::enable_if_t< std::is_base_of< kdop_base< typename KDop::arithmetic_type, KDop::k, KDop >, KDop >::value > >
      : std::true_type
  {};
}

#endif  // INC_BVH_KDOP_HPP
