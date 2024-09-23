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
#ifndef INC_TEST_COMMON_HPP
#define INC_TEST_COMMON_HPP

#include <Kokkos_Macros.hpp>
#include <bvh/types.hpp>
#include <catch2/catch.hpp>

#include <Kokkos_Array.hpp>
#include <bvh/math/vec.hpp>
#include <bvh/kdop.hpp>
#include <bvh/patch.hpp>
#include <bvh/util/container.hpp>
#include <random>


class Element
{
public:

  using kdop_type = bvh::bphase_kdop;
  static constexpr std::size_t vertex_count = 8;

  KOKKOS_INLINE_FUNCTION Element( std::size_t _index = static_cast< std::size_t >( -1 ) ) : m_index( _index ) {};

  KOKKOS_DEFAULTED_FUNCTION Element( const Element &_other ) = default;
  KOKKOS_DEFAULTED_FUNCTION Element( Element &&_other ) noexcept = default;
  KOKKOS_DEFAULTED_FUNCTION Element &operator=( const Element &_other ) = default;
  KOKKOS_DEFAULTED_FUNCTION Element &operator=( Element &&_other ) noexcept = default;

  KOKKOS_INLINE_FUNCTION auto begin() { return m_vertices.data(); }

  KOKKOS_INLINE_FUNCTION auto begin() const { return m_vertices.data(); }

  KOKKOS_INLINE_FUNCTION auto end() { return m_vertices.data() + 8; }

  KOKKOS_INLINE_FUNCTION auto end() const { return m_vertices.data() + 8; }

  KOKKOS_INLINE_FUNCTION bvh::m::vec3d centroid() const
  {
    auto ret = bvh::m::vec3d::zeros();
    for ( std::size_t i = 0; i < m_vertices.size(); ++i )
      ret += m_vertices[i];
    return ret / static_cast< double >( m_vertices.size() );
  };

  template< typename... Args >
  KOKKOS_INLINE_FUNCTION std::enable_if_t< ( sizeof...( Args ) == vertex_count ) > setVertices( Args &&..._args )
  {
    m_vertices = Kokkos::Array< bvh::m::vec3d, vertex_count >{ std::forward< Args >( _args )... };
    m_bounds = kdop_type::from_vertices( m_vertices );
  }

  KOKKOS_INLINE_FUNCTION void setIndex( std::size_t _index ) { m_index = _index; }

  KOKKOS_INLINE_FUNCTION bvh::span< const bvh::m::vec3d > vertices() const { return m_vertices; }

  KOKKOS_INLINE_FUNCTION const kdop_type &kdop() const { return m_bounds; }

  KOKKOS_INLINE_FUNCTION std::size_t global_id() const { return m_index; }

  friend std::ostream &operator<<( std::ostream &os, const Element &el )
  {
    os << "[ ";
    if ( !el.m_vertices.empty() )
      os << '{' << el.m_vertices[0] << '}';
    for ( std::size_t i = 1; i < el.m_vertices.size(); ++i )
    {
      os << ", {" << el.m_vertices[i] << '}';
    }
    os << " ]";

    return os;
  }

  friend KOKKOS_INLINE_FUNCTION bool operator==( const Element &_lhs, const Element &_rhs )
  {
    if ( _lhs.m_index != _rhs.m_index )
      return false;

    for ( std::size_t i = 0; i < _lhs.m_vertices.size(); ++i )
      if ( _lhs.vertices()[i] != _rhs.vertices()[i] )
        return false;

    return _lhs.m_bounds == _rhs.m_bounds;
  }


  template< typename Serializer >
  friend void
  serialize( Serializer &_s, const Element &_e )
  {
    _s | _e.m_index | _e.m_vertices | _e.m_bounds;
  }

private:

  std::size_t m_index;
  Kokkos::Array< bvh::m::vec3d, 8 > m_vertices;
  kdop_type m_bounds;
};

KOKKOS_INLINE_FUNCTION auto
  get_entity_kdop( const Element &_element )
{
  return _element.kdop();
}

KOKKOS_INLINE_FUNCTION auto
  get_entity_global_id( const Element &_element )
{
  return _element.global_id();
}

KOKKOS_INLINE_FUNCTION auto
  get_entity_centroid( const Element &_element )
{
  return _element.centroid();
}

inline bvh::dynarray< Element > buildElementGrid( int _x, int _y, int _z, std::size_t _base_index = 0,
                                                  double _origin_shift = 0.0 )
{
  bvh::dynarray< Element > ret;
  ret.reserve( _x * _y * _z );

  double dx = 1.0 / _x;
  double dy = 1.0 / _y;
  double dz = 1.0 / _z;

  std::size_t index = _base_index;

  for ( int k = 0; k < _z; ++k )
  {
    for ( int j = 0; j < _y; ++j )
    {
      for ( int i = 0; i < _x; ++i )
      {
        double x = _origin_shift + i * dx;
        double y = _origin_shift + j * dy;
        double z = _origin_shift + k * dz;
        Element el( index++ );
        el.setVertices( bvh::m::vec3d{ x, y, z },
                        bvh::m::vec3d{ x + dx, y, z },
                        bvh::m::vec3d{ x + dx, y + dy, z },
                        bvh::m::vec3d{ x, y + dy, z },
                        bvh::m::vec3d{ x, y, z + dz },
                        bvh::m::vec3d{ x + dx, y, z + dz },
                        bvh::m::vec3d{ x + dx, y + dy, z + dz },
                        bvh::m::vec3d{ x, y + dy, z + dz } );
        ret.emplace_back( std::move( el ) );
      }
    }
  }

  return ret;
}

inline bvh::view< Element * >
build_element_grid( int _x, int _y, int _z, std::size_t _base_index = 0, double _origin_shift = 0.0 )
{
  bvh::view< Element * > ret( "elements", _x * _y * _z );

  double dx = 1.0 / _x;
  double dy = 1.0 / _y;
  double dz = 1.0 / _z;

  auto rp = Kokkos::MDRangePolicy<Kokkos::Rank<3>>{{ 0, 0, 0 }, { _x, _y, _z }};
  Kokkos::parallel_for(rp, KOKKOS_LAMBDA( int _i, int _j, int _k ) {
    double x = _origin_shift + _i * dx;
    double y = _origin_shift + _j * dy;
    double z = _origin_shift + _k * dz;

    const std::size_t index = _i + _j * _x + _k * _x * _y;
    // assert( index < ret.extent( 0 ) );
    auto &el = ret( index );
    el.setIndex( _base_index + index );
    el.setVertices( bvh::m::vec3d{ x, y, z }, bvh::m::vec3d{ x + dx, y, z }, bvh::m::vec3d{ x + dx, y + dy, z },
                    bvh::m::vec3d{ x, y + dy, z }, bvh::m::vec3d{ x, y, z + dz }, bvh::m::vec3d{ x + dx, y, z + dz },
                    bvh::m::vec3d{ x + dx, y + dy, z + dz }, bvh::m::vec3d{ x, y + dy, z + dz } );
  } );

  return ret;
}

inline bvh::dynarray< bvh::patch<> > buildElementPatchGrid( int _x, int _y, int _z, std::size_t _base_index = 0 )
{
  bvh::dynarray< bvh::patch<> > ret;
  ret.reserve( _x * _y * _z );

  double dx = 1.0 / _x;
  double dy = 1.0 / _y;
  double dz = 1.0 / _z;

  std::size_t index = _base_index;

  for ( int k = 0; k < _z; ++k )
  {
    for ( int j = 0; j < _y; ++j )
    {
      for ( int i = 0; i < _x; ++i )
      {
        double x = i * dx;
        double y = j * dy;
        double z = k * dz;
        Element el( index++ );
        el.setVertices( bvh::m::vec3d{ x, y, z },
                        bvh::m::vec3d{ x + dx, y, z },
                        bvh::m::vec3d{ x + dx, y + dy, z },
                        bvh::m::vec3d{ x, y + dy, z },
                        bvh::m::vec3d{ x, y, z + dz },
                        bvh::m::vec3d{ x + dx, y, z + dz },
                        bvh::m::vec3d{ x + dx, y + dy, z + dz },
                        bvh::m::vec3d{ x, y + dy, z + dz } );
        ret.emplace_back( el.global_id(), bvh::span< const Element >( &el, &el + 1 ) );
      }
    }
  }

  return ret;
}

#include <bvh/vt/print.hpp>

inline bvh::dynarray< Element >
buildElementGridParallel( std::size_t _rank, std::size_t _nranks,
                          std::size_t _x, std::size_t _y, std::size_t _z )
{
  const double dx = 1.0 / _x;
  const double dy = 1.0 / _y;
  const double dz = 1.0 / _z;

  auto fac = ( _x * _y * _z ) / _nranks;
  auto base_index = _rank * fac;
  auto end_index = ( _rank == _nranks - 1 ) ? ( _x * _y * _z ) : ( ( _rank + 1 ) * fac );
  bvh::vt::debug( "{}: fac: {}\n", _rank, fac );
  bvh::vt::debug( "{}: base_index: {}\n", _rank, base_index );
  bvh::vt::debug( "{}: end_index: {}\n", _rank, end_index );

  bvh::dynarray< Element > ret;
  for ( auto idx = base_index; idx < end_index; ++idx )
  {
    const std::size_t i = idx % _x;
    const std::size_t j = ( idx / _x ) % _y;
    const std::size_t k = idx / ( _x * _y );

    const double x = i * dx;
    const double y = j * dy;
    const double z = k * dz;
    ret.emplace_back( idx );
    ret.back().setVertices( bvh::m::vec3d{ x, y, z },
                            bvh::m::vec3d{ x + dx, y, z },
                            bvh::m::vec3d{ x + dx, y + dy, z },
                            bvh::m::vec3d{ x, y + dy, z },
                            bvh::m::vec3d{ x, y, z + dz },
                            bvh::m::vec3d{ x + dx, y, z + dz },
                            bvh::m::vec3d{ x + dx, y + dy, z + dz },
                            bvh::m::vec3d{ x, y + dy, z + dz } );
  }

  return ret;
}

inline bvh::dynarray< bvh::patch<> >
buildElementPatchGridParallel( std::size_t _rank, std::size_t _nranks,
                          std::size_t _x, std::size_t _y, std::size_t _z )
{
  const double dx = 1.0 / _x;
  const double dy = 1.0 / _y;
  const double dz = 1.0 / _z;

  auto fac = ( _x * _y * _z ) / _nranks;
  auto base_index = _rank * fac;
  auto end_index = ( _rank == _nranks - 1 ) ? ( _x * _y * _z ) : ( ( _rank + 1 ) * fac );
  bvh::vt::debug( "{}: fac: {}\n", _rank, fac );
  bvh::vt::debug( "{}: base_index: {}\n", _rank, base_index );
  bvh::vt::debug( "{}: end_index: {}\n", _rank, end_index );

  bvh::dynarray< bvh::patch<> > ret;
  for ( auto idx = base_index; idx < end_index; ++idx )
  {
    const std::size_t i = idx % _x;
    const std::size_t j = ( idx / _x ) % _y;
    const std::size_t k = idx / ( _x * _y );

    const double x = i * dx;
    const double y = j * dy;
    const double z = k * dz;

    Element e( idx );
    e.setVertices( bvh::m::vec3d{ x, y, z },
                   bvh::m::vec3d{ x + dx, y, z },
                   bvh::m::vec3d{ x + dx, y + dy, z },
                   bvh::m::vec3d{ x, y + dy, z },
                   bvh::m::vec3d{ x, y, z + dz },
                   bvh::m::vec3d{ x + dx, y, z + dz },
                   bvh::m::vec3d{ x + dx, y + dy, z + dz },
                   bvh::m::vec3d{ x, y + dy, z + dz } );

    ret.emplace_back( idx, bvh::span< const Element >( &e, &e + 1 ) );
  }

  return ret;
}

#include <bvh/util/kokkos.hpp>

template< typename T, typename... Args >
inline void gen_array( bvh::host_view< T > _arr, Args &&... _args )
{
  using value_type = typename bvh::host_view< T  >::value_type;

  std::array< value_type, sizeof...( _args ) > varr = {{ static_cast< value_type >( std::forward< Args >( _args ) )... }};
  Kokkos::parallel_for( Kokkos::RangePolicy< bvh::host_execution_space >( 0, sizeof...( _args ) ), [_arr, varr]( int i ){
    _arr( i ) = varr[i];
  } );
}

template< typename T, typename... Args >
inline void test_array( bvh::host_view< T > _arr, Args &&... _args )
{
  using value_type = typename bvh::host_view< T >::value_type;

  std::array< value_type, sizeof...( _args ) > expected = {{ static_cast< value_type >( std::forward< Args >( _args ) )... }};
  for ( std::size_t i = 0; i < sizeof...( _args ); ++i )
  {
    REQUIRE( _arr( i ) == expected[i] );
  }
}

inline bvh::view< bvh::bphase_kdop * >
generate_random_kdops( std::default_random_engine &_eng,
                       std::size_t _count,
                       const bvh::m::vec3d &_min,
                       const bvh::m::vec3d &_max,
                       double _max_size )
{
  bvh::host_view< bvh::bphase_kdop * > ret( "random_kdops", _count );

  std::uniform_real_distribution<> xdist( _min.x() + _max_size, _max.x() - _max_size );
  std::uniform_real_distribution<> ydist( _min.y() + _max_size, _max.y() - _max_size );
  std::uniform_real_distribution<> zdist( _min.z() + _max_size, _max.z() - _max_size );
  std::uniform_real_distribution<> rdist( std::numeric_limits< double >::epsilon(), _max_size );

  for ( std::size_t i = 0; i < _count; ++i )
  {
    ret( i ) = bvh::bphase_kdop::from_sphere( bvh::m::vec3d( xdist( _eng ), ydist( _eng ), zdist( _eng ) ), rdist( _eng ) );
  }

  return Kokkos::create_mirror_view_and_copy( bvh::default_execution_space{}, ret );
}

inline bvh::view< bvh::bphase_kdop * >
generate_kdop_grid( std::size_t _count,
                    const bvh::m::vec3d &_min,
                    const bvh::m::vec3d &_max,
                    double _radius )
{
  bvh::host_view< bvh::bphase_kdop * > ret( "kdop_grid", _count * _count * _count );

  bvh::m::vec3d pos = _min;
  bvh::m::vec3d incr = ( _max - _min ) / static_cast< double >( _count );
  for ( std::size_t z = 0; z < _count; ++z )
  {
    for ( std::size_t y = 0; y < _count; ++y )
    {
      for ( std::size_t x = 0; x < _count; ++x )
      {
        const auto idx = z * _count * _count + y * _count + x;
        ret( idx ) = bvh::bphase_kdop::from_sphere( pos, _radius );
        pos.x() += incr.x();
      }
      pos.y() += incr.y();
      pos.x() = _min.x();
    }
    pos.z() += incr.z();
    pos.y() = _min.y();
  }

  return Kokkos::create_mirror_view_and_copy( bvh::default_execution_space{}, ret );
}

inline bvh::view< bvh::entity_snapshot * >
snapshots_from_kdops( bvh::view< const bvh::bphase_kdop * > _kdops )
{
  bvh::view< bvh::entity_snapshot * > ret( "entity snapshots", _kdops.extent( 0 ) );
  Kokkos::parallel_for( _kdops.extent( 0 ), KOKKOS_LAMBDA( int _i ){
    ret( _i ) = bvh::entity_snapshot( _i, _kdops( _i ), _kdops( _i ).centroid(), _i );
  } );

  return ret;
}

static_assert( std::is_same_v< bvh::m::epsilon_type_of_t< bvh::m::vec3d >, double > );
static_assert( std::is_same_v< bvh::m::epsilon_type_of_t< double >, double > );

template< typename T >
struct approx
{
  explicit approx( const T &_val, bvh::m::epsilon_type_of_t< std::remove_cv_t< std::remove_reference_t< T > > > _eps = bvh::m::epsilon_value< bvh::m::epsilon_type_of_t< T > > )
    : val( _val ), eps( _eps )
  {}

  T val;
  bvh::m::epsilon_type_of_t< T > eps;

  friend bool operator==( const T &_lhs, const approx< T > &_rhs )
  {
    using bvh::m::approx_equals;
    return approx_equals( _lhs, _rhs.val, _rhs.eps );
  }

  friend bool operator==( const approx< T > &_lhs, const T &_rhs )
  {
    using bvh::m::approx_equals;
    return approx_equals( _lhs.val, _rhs, _lhs.eps );
  }

  friend std::ostream &operator<<( std::ostream &_oss, const approx< T > &_val )
  {
    _oss << "approx( " << _val.val << " )";
    return _oss;
  }
};

#endif  // INC_TEST_COMMON_HPP
