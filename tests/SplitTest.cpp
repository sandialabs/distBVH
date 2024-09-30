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
#include <catch2/catch.hpp>

#include <array>
#include <iterator>
#include <vector>

#include <bvh/split/axis.hpp>
#include <bvh/split/split.hpp>
#include <bvh/split/mean.hpp>
#include <bvh/kdop.hpp>
#include <bvh/range.hpp>
#include <bvh/snapshot.hpp>
#include <bvh/util/span.hpp>
#include "TestCommon.hpp"

using kd_type = bvh::dop_26< double >;
using kd06_type = bvh::dop_6< double >;

namespace
{

  template< typename kdop_type = kd_type>
  struct element
  {
    using centroid_type = bvh::m::vec3d;

    element() = default;

    element( unsigned _gid, const bvh::m::vec3d &_center )
      : m_kdop( kdop_type::from_sphere( _center, 1.0 ) ),
        m_gid( _gid ),
        m_centroid( _center )
    {

    }

    element( const element &_other ) = default;
    element &operator=( const element &_other ) = default;
    element( element &&_other ) = default;
    element &operator=( element &&_other ) = default;

    kdop_type m_kdop;
    unsigned m_gid;
    bvh::m::vec3d m_centroid;

    const KOKKOS_INLINE_FUNCTION kdop_type &kdop() const noexcept{ return m_kdop; }
    const KOKKOS_INLINE_FUNCTION bvh::m::vec3d &centroid() const noexcept { return m_centroid; }
    KOKKOS_INLINE_FUNCTION unsigned global_id() const noexcept { return m_gid; }
  };
}

TEST_CASE("in place recursive mean splitting", "[split]")
{
  using traits_type = bvh::element_traits< element<kd_type> >;

  static constexpr std::size_t N = 8;
  std::array< element<kd_type>, N > elements;
  for ( std::size_t i = 0; i < N; ++i )
  {
    elements[i] = element<kd_type>( i, bvh::m::vec3d( static_cast< double >( i ), 0.0, 0.0 ) );
  }

  auto range = bvh::make_range( elements.begin(), elements.end() );
  auto kdops = bvh::transform_range( range, traits_type::get_kdop );

  SECTION( "longest axis" )
  {
    auto kdop = kd_type::from_kdops( kdops.begin(), kdops.end() );
    REQUIRE( bvh::axis::longest::axis( kdop ) == 0 );
  }

  SECTION( "single split" )
  {
    auto sp = bvh::split_in_place< bvh::split::mean >( range, 0 );

    REQUIRE( std::distance( range.begin(), sp ) == 4 );
    REQUIRE( std::distance( sp, range.end() ) == 4 );
  }

  SECTION( "in place recursive split depth 0" )
  {
    auto sps = bvh::split_in_place_recursive< bvh::split::mean, bvh::axis::longest >( range, 0 );
    REQUIRE( sps.size() == 2 );
  }

  SECTION( "in place recursive split depth 1" )
  {
    auto sps = bvh::split_in_place_recursive< bvh::split::mean, bvh::axis::longest >( range, 1 );
    REQUIRE( sps.size() == 3 );

    REQUIRE( std::distance( sps[0], sps[1] ) == 4 );
    REQUIRE( std::distance( sps[1], sps[2] ) == 4 );
  }

  SECTION( "in place recursive split depth 2" )
  {
    auto sps = bvh::split_in_place_recursive< bvh::split::mean, bvh::axis::longest >( range, 2 );
    REQUIRE( sps.size() == 5 );

    for ( std::size_t i = 0; i < sps.size() - 1; ++i )
      REQUIRE( std::distance( sps[i], sps[i + 1] ) == 2 );
  }

  SECTION( "in place recursive split depth 3" )
  {
    auto sps = bvh::split_in_place_recursive< bvh::split::mean, bvh::axis::longest >( range, 3 );
    REQUIRE( sps.size() == 9 );

    for ( std::size_t i = 0; i < sps.size() - 1; ++i )
      REQUIRE( std::distance( sps[i], sps[i + 1] ) == 1 );
  }

  SECTION( "in place recursive split depth 4" )
  {
    // Recursive split generates empty nodes
    auto sps = bvh::split_in_place_recursive< bvh::split::mean, bvh::axis::longest >( range, 4 );
    REQUIRE( sps.size() == 17 );

    for ( std::size_t i = 0; i < sps.size() - 1; ++i )
      REQUIRE( std::distance( sps[i], sps[i + 1] ) <= 1 );
  }
}

TEST_CASE("recursive mean splitting 1D elements", "[split]")
{
  using traits_type = bvh::element_traits< element<kd_type> >;

  static constexpr std::size_t N = 8;
  std::array< element<kd_type>, N > elements;
  for ( std::size_t i = 0; i < N; ++i )
  {
    elements[i] = element<kd_type>( i, bvh::m::vec3d( static_cast< double >( i ), 0.0, 0.0 ) );
  }

  bvh::element_permutations m_permutations;
  m_permutations.indices.resize( N );
  std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);

  auto range = bvh::make_range( elements.begin(), elements.end() );
  auto kdops = bvh::transform_range( range, traits_type::get_kdop );

  auto span_ele = bvh::span< const element<kd_type> >{elements.data(), N};
  auto range_perm = bvh::make_range( m_permutations.indices.begin(), m_permutations.indices.end() );

  SECTION( "longest axis" )
  {
    auto kdop = kd_type::from_kdops( kdops.begin(), kdops.end() );
    REQUIRE( bvh::axis::longest::axis( kdop ) == 0 );
  }

  SECTION( "single split" )
  {
    const int axis = 0;
    auto sp = bvh::detail::split_permutation< bvh::split::mean >( span_ele, range_perm, axis );

    REQUIRE( std::distance( range_perm.begin(), sp ) == 4 );
    REQUIRE( std::distance( sp, range_perm.end() ) == 4 );
  }

  SECTION( "recursive split depth 0" )
  {
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 0, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 0 );
  }

  SECTION( "recursive split depth 1" )
  {
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 1, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 1 );
    REQUIRE( m_permutations.splits[0] == N/2 );
  }

  SECTION( "recursive split depth 2" )
  {
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 2, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 3 );

    for ( std::size_t i = 0; i < m_permutations.splits.size() - 1; ++i ) {
      REQUIRE( m_permutations.splits[i] <= m_permutations.splits[i] );
      REQUIRE( ( m_permutations.splits[i+1] - m_permutations.splits[i] ) == 2 );
    }
  }

  SECTION( "recursive split depth 3" )
  {
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 3, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 7 );

    for ( std::size_t i = 0; i < m_permutations.splits.size() - 1; ++i ) {
      REQUIRE( m_permutations.splits[i] <= m_permutations.splits[i] );
      REQUIRE( ( m_permutations.splits[i+1] - m_permutations.splits[i] ) == 1 );
    }
  }

  SECTION( "recursive split depth 4" )
  {
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 4, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 15 );

    // Recursive split generates empty nodes
    for ( std::size_t i = 0; i < m_permutations.splits.size() - 1; ++i ) {
      REQUIRE( m_permutations.splits[i] <= m_permutations.splits[i] );
      REQUIRE( ( m_permutations.splits[i+1] - m_permutations.splits[i] ) <= 1 );
    }
  }

}

TEST_CASE("recursive mean splitting 2D elements", "[split]")
{
  using traits_type = bvh::element_traits< element< kd06_type > >;

  static constexpr std::size_t Nx = 4, Ny = 3;
  static constexpr std::size_t N = Nx * Ny;
  std::array< element< kd06_type >, N > elements;
  for ( std::size_t j = 0; j < Ny; ++j ) {
    for ( std::size_t i = 0; i < Nx; ++i ) {
      elements[i + j*Nx] = element< kd06_type >( i + j*Nx, bvh::m::vec3d( static_cast<double>(i), static_cast<double>(j), 0.0 ) );
    }
  }

  bvh::element_permutations m_permutations;

  auto range = bvh::make_range( elements.begin(), elements.end() );
  auto kdops = bvh::transform_range( range, traits_type::get_kdop );

  auto span_ele = bvh::span< const element<kd06_type> >{elements.data(), N};

  SECTION( "longest axis" )
  {
    auto kdop = kd06_type::from_kdops( kdops.begin(), kdops.end() );
    REQUIRE( bvh::axis::longest::axis( kdop ) == 0 );
  }

  SECTION( "single split" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    auto range_perm = bvh::make_range( m_permutations.indices.begin(), m_permutations.indices.end() );
    //
    const int axis = 0;
    auto sp = bvh::detail::split_permutation< bvh::split::mean >( span_ele, range_perm, axis );
    //
    REQUIRE( std::distance( range_perm.begin(), sp ) == N/2 );
    REQUIRE( std::distance( sp, range_perm.end() ) == N/2 );
  }

  SECTION( "recursive split depth 0" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 0, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 0 );
  }

  SECTION( "recursive split depth 1" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 1, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 1 );
    REQUIRE( m_permutations.splits[0] == N/2 );
    //
    for (size_t ii = m_permutations.splits[0]; ii < m_permutations.splits[1]; ++ii) {
      const auto point = elements[m_permutations.indices[ii]].centroid();
      REQUIRE( point[0] <= Nx / 2);
    }
  }

  SECTION( "recursive split depth 2" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations< bvh::split::mean, bvh::axis::longest >( span_ele, 2, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 3 );
    for ( std::size_t i = 0; i < m_permutations.splits.size() - 1; ++i ) {
      REQUIRE( m_permutations.splits[i] <= m_permutations.splits[i] );
      REQUIRE( ( m_permutations.splits[i+1] - m_permutations.splits[i] ) == 3 );
    }
    //
    // 2022-06-15 The axis for the 2nd split should be different
    //
    //----
    //
    // 2022-06-15 We should test the permutation indices
    //
  }

}

#ifndef BVH_ENABLE_CUDA
TEST_CASE("new recursive mean splitting 2D elements", "[split]")
{
  static constexpr std::size_t Nx = 4, Ny = 3;
  static constexpr std::size_t N = Nx * Ny;
  std::array< bvh::entity_snapshot, N > elements;
  for ( std::size_t j = 0; j < Ny; ++j ) {
    for ( std::size_t i = 0; i < Nx; ++i ) {
      bvh::m::vec3d tmp_vec( static_cast<double>(i), static_cast<double>(j), 0.0 );
      element< bvh::bphase_kdop > tmp_ele( i + j*Nx, tmp_vec);
      elements[i + j*Nx] = bvh::entity_snapshot{ i + j*Nx, tmp_ele.kdop(), tmp_vec, i + j*Nx};
    }
  }

  bvh::element_permutations m_permutations;

  auto span_ele = bvh::span< bvh::entity_snapshot >{elements.data(), N};

  SECTION( "single split" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    auto span_perm = bvh::span< size_t >{ m_permutations.indices };
    //
    std::vector< std::pair< bvh::entity_snapshot, size_t > > combi( N );
    for (size_t ii = 0; ii < N; ++ii)
      combi[ii] = std::make_pair( span_ele[ii], ii );
    //
    const int axis = 0;
    auto sp = bvh::detail::split_permutation_ml< bvh::split::mean >( span_ele, span_perm, axis, {combi.data(), N} );
    //
    REQUIRE( sp == N/2 );
  }

  SECTION( "recursive split depth 0" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations_ml< bvh::split::mean, bvh::axis::longest >( span_ele, 0, &m_permutations);
    //
    REQUIRE( m_permutations.splits.size() == 0 );
  }

  SECTION( "recursive split depth 1" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations_ml< bvh::split::mean, bvh::axis::longest >( span_ele, 1, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 1 );
    REQUIRE( m_permutations.splits[0] == N/2 );
    //
  }

  SECTION( "recursive split depth 2" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations_ml< bvh::split::mean, bvh::axis::longest >( span_ele, 2, &m_permutations);
    //
    REQUIRE( m_permutations.splits.size() == 3 );
    //
    REQUIRE( m_permutations.splits[1] - 0 == 6);
    REQUIRE( N - m_permutations.splits[1] == 6);
    //
    REQUIRE( m_permutations.splits[0] - 0 >= 2 );
    REQUIRE( m_permutations.splits[1] - m_permutations.splits[0] >= 2 );
    REQUIRE( m_permutations.splits[2] - m_permutations.splits[1] >= 2 );
    REQUIRE( N - m_permutations.splits[2] >= 2 );
    //
  }

  SECTION( "recursive split depth 3" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations_ml< bvh::split::mean, bvh::axis::longest >( span_ele, 3, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 7 );
    for ( std::size_t i = 0; i < m_permutations.splits.size() - 1; ++i ) {
      REQUIRE( m_permutations.splits[i] <= m_permutations.splits[i+1] );
      REQUIRE( ( m_permutations.splits[i+1] - m_permutations.splits[i] ) <= 4 );
    }
    //
  }

  SECTION( "recursive split depth 4" )
  {
    m_permutations.indices.resize( N );
    std::iota(m_permutations.indices.begin(), m_permutations.indices.end(), 0);
    //
    m_permutations.splits.clear();
    bvh::split_permutations_ml< bvh::split::mean, bvh::axis::longest >( span_ele, 4, &m_permutations);
    REQUIRE( m_permutations.splits.size() == 15 );
    for ( std::size_t i = 0; i < m_permutations.splits.size() - 1; ++i ) {
      REQUIRE( m_permutations.splits[i] <= m_permutations.splits[i + 1] );
      REQUIRE( ( m_permutations.splits[i+1] - m_permutations.splits[i] ) <= 2 );
    }
    //
  }
}
#endif
