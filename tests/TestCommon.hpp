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

#include <catch2/catch.hpp>

#include <bvh/math/vec.hpp>
#include <bvh/kdop.hpp>
#include <bvh/patch.hpp>
#include <bvh/util/container.hpp>
#include <numeric>
#include <random>


class Element
{
public:

  using kdop_type = bvh::bphase_kdop;

  Element( std::size_t _index = static_cast< std::size_t >( -1 ) ) : m_index( _index ) {};

  bvh::m::vec3d centroid() const
  {
    return std::accumulate( m_vertices.begin(), m_vertices.end(), bvh::m::vec3d::zeros() ) / static_cast< double >( m_vertices.size() );
  };

  void setVertices( std::initializer_list< bvh::m::vec3d > _verts )
  {
    setVertices( _verts.begin(), _verts.end() );
  }

  template< typename Iterator >
  void setVertices( Iterator _begin, Iterator _end )
  {
    m_vertices.reserve( static_cast< std::size_t >( std::distance( _begin, _end ) ) );
    std::copy( _begin, _end, std::back_inserter( m_vertices ) );
    m_bounds = kdop_type::from_vertices( m_vertices.begin(), m_vertices.end() );
  }

  template< typename Iterator >
  void setVerticesAndBounds( Iterator _begin, Iterator _end, const kdop_type &_bounds )
  {
    m_vertices.reserve( static_cast< std::size_t >( std::distance( _begin, _end ) ) );
    std::copy( _begin, _end, std::back_inserter( m_vertices ) );
    m_bounds = _bounds;
  }

  void setIndex( std::size_t _index ) { m_index = _index; }

  const bvh::dynarray< bvh::m::vec3d > &vertices() const { return m_vertices; }

  const kdop_type &kdop() const { return m_bounds; }
  std::size_t global_id() const { return m_index; }

  auto begin() { return m_vertices.begin(); }
  auto begin() const { return m_vertices.begin(); }

  auto end() { return m_vertices.end(); }
  auto end() const { return m_vertices.end(); }

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

  friend bool operator==( const Element &_lhs, const Element &_rhs )
  {
    if ( _lhs.m_index != _rhs.m_index )
      return false;

    if ( !std::equal(_lhs.m_vertices.begin(), _lhs.m_vertices.end(),
                     _rhs.m_vertices.begin(), _rhs.m_vertices.end() ) )
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
  bvh::dynarray< bvh::m::vec3d > m_vertices;
  kdop_type m_bounds;
};


inline auto
get_entity_kdop( const Element &_element )
{
  return _element.kdop();
}

inline auto
get_entity_global_id( const Element &_element )
{
  return _element.global_id();
}

inline auto
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
        el.setVertices( { bvh::m::vec3d{ x, y, z },
                                    { x + dx, y, z },
                                    { x + dx, y + dy, z },
                                    { x, y + dy, z },
                                    { x, y, z + dz },
                                    { x + dx, y, z + dz },
                                    { x + dx, y + dy, z + dz },
                                    { x, y + dy, z + dz } } );
        ret.emplace_back( std::move( el ) );
      }
    }
  }

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
        el.setVertices( { bvh::m::vec3d{ x, y, z },
                          { x + dx, y, z },
                          { x + dx, y + dy, z },
                          { x, y + dy, z },
                          { x, y, z + dz },
                          { x + dx, y, z + dz },
                          { x + dx, y + dy, z + dz },
                          { x, y + dy, z + dz } } );
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
    ret.back().setVertices( { bvh::m::vec3d{ x, y, z },
                      { x + dx, y, z },
                      { x + dx, y + dy, z },
                      { x, y + dy, z },
                      { x, y, z + dz },
                      { x + dx, y, z + dz },
                      { x + dx, y + dy, z + dz },
                      { x, y + dy, z + dz }} );
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
    e.setVertices( { bvh::m::vec3d{ x, y, z },
                     { x + dx, y, z },
                     { x + dx, y + dy, z },
                     { x, y + dy, z },
                     { x, y, z + dz },
                     { x + dx, y, z + dz },
                     { x + dx, y + dy, z + dz },
                     { x, y + dy, z + dz }} );

    ret.emplace_back( idx, bvh::span< const Element >( &e, &e + 1 ) );
  }

  return ret;
}


#ifdef BVH_ENABLE_KOKKOS

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
  Kokkos::parallel_for( Kokkos::RangePolicy< bvh::host_execution_space >( 0, sizeof...( _args ) ), [_arr, expected]( int i ){
    REQUIRE( _arr( i ) == expected[i] );
  } );
}

inline bvh::view< bvh::bphase_kdop * >
generate_random_kdops( std::default_random_engine &_eng,
                       std::size_t _count,
                       const bvh::m::vec3d &_min,
                       const bvh::m::vec3d &_max,
                       double _max_size )
{
  bvh::host_view< bvh::bphase_kdop * > ret( "random_kdops", _count );

  std::uniform_real_distribution xdist( _min.x() + _max_size, _max.x() - _max_size );
  std::uniform_real_distribution ydist( _min.y() + _max_size, _max.y() - _max_size );
  std::uniform_real_distribution zdist( _min.z() + _max_size, _max.z() - _max_size );
  std::uniform_real_distribution rdist( std::numeric_limits< double >::epsilon(), _max_size );

  for ( std::size_t i = 0; i < _count; ++i )
  {
    ret( i ) = bvh::bphase_kdop::from_sphere( bvh::m::vec3d( xdist( _eng ), ydist( _eng ), zdist( _eng ) ), rdist( _eng ) );
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

#endif // BVH_ENABLE_KOKKOS

#endif  // INC_TEST_COMMON_HPP
