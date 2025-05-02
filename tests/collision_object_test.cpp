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
#include "TestCommon.hpp"
#include "bvh/types.hpp"
#include <bvh/vt/helpers.hpp>
#include <bvh/collision_object.hpp>
#include <bvh/collision_world.hpp>
#include <bvh/util/epoch.hpp>
#include <bvh/vt/print.hpp>
#include <numeric>
#include <type_traits>
#include <vt/collective/collective_alg.h>
#include <vt/collective/reduce/operators/functors/plus_op.h>
#include <vt/context/context.h>
#include <vt/scheduler/scheduler.h>
#include <vt/termination/epoch_guard.h>


struct test_trees
{
  void operator()( const bvh::snapshot_tree &_tree )
  {
    auto nranks = ::vt::theContext()->getNumNodes();
    REQUIRE( _tree.count() == od_factor * nranks );
  }

  std::size_t od_factor;
};


struct test_sing_trees
{
  void operator()( const bvh::snapshot_tree &_tree )
  {
    auto nranks = ::vt::theContext()->getNumNodes();
    REQUIRE( _tree.count() == od_factor * nranks );
  }

  std::size_t od_factor;
};

struct test_empty_trees
{
  void operator()( const bvh::snapshot_tree &_tree )
  {
    ::vt::theContext()->getNumNodes();
    REQUIRE( _tree.count() == 0 );
  }

  std::size_t od_factor;
};

std::size_t test_od_factor = 0;

void verify_num_elements( std::size_t _count )
{
  bvh::vt::debug("{}: count: {}\n", ::vt::theContext()->getNode(), _count );
  // Cube test_od_factor because each dimension is multiplied...
  REQUIRE( _count == 12 * ::vt::theContext()->getNumNodes() * test_od_factor * test_od_factor * test_od_factor );
};

void
verify_empty_elements( std::size_t _count )
{
  bvh::vt::debug( "{}: count: {}\n", ::vt::theContext()->getNode(), _count );
  REQUIRE( _count == 0 );
};

TEST_CASE( "collision_object init", "[vt]")
{
  std::size_t od_factor = GENERATE( 1, 2, 4, 32, 64 );
  test_od_factor = od_factor;
  bvh::collision_world world( od_factor );

  auto &obj = world.create_collision_object();

  auto rank = ::vt::theContext()->getNode();
  auto elements = build_element_grid( 2 * od_factor, 3 * od_factor, 2 * od_factor, rank * 12 * od_factor );
  const std::array bound_vers{ bvh::m::vec3d{ 0.0, 0.0, 0.0 },
                               bvh::m::vec3d{ 0.0, 0.0, 1.0 },
                               bvh::m::vec3d{ 0.0, 1.0, 0.0 },
                               bvh::m::vec3d{ 0.0, 1.0, 1.0 },
                               bvh::m::vec3d{ 1.0, 0.0, 0.0 },
                               bvh::m::vec3d{ 1.0, 0.0, 1.0 },
                               bvh::m::vec3d{ 1.0, 1.0, 0.0 },
                               bvh::m::vec3d{ 1.0, 1.0, 1.0 } };
  const auto bounds = Element::kdop_type::from_vertices( bound_vers.begin(), bound_vers.end() );
  const std::array update_bounds_vers{ bvh::m::vec3d{ 10.0, 10.0, 10.0 },
                                       bvh::m::vec3d{ 10.0, 10.0, 11.0 },
                                       bvh::m::vec3d{ 10.0, 11.0, 10.0 },
                                       bvh::m::vec3d{ 10.0, 11.0, 11.0 },
                                       bvh::m::vec3d{ 11.0, 10.0, 10.0 },
                                       bvh::m::vec3d{ 11.0, 10.0, 11.0 },
                                       bvh::m::vec3d{ 11.0, 11.0, 10.0 },
                                       bvh::m::vec3d{ 11.0, 11.0, 11.0 } };
  const auto update_bounds = Element::kdop_type::from_vertices( update_bounds_vers.begin(), update_bounds_vers.end() );
  bvh::vt::debug( "{}: bounds: {}\n", ::vt::theContext()->getNode(), bounds );
  auto update_elements = build_element_grid( 2 * od_factor, 3 * od_factor, 2 * od_factor, rank * 12 * od_factor, 10.0 );

  auto split_method
    = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis, bvh::split_algorithm::clustering );

  bvh::vt::debug("{}: od_factor: {} split method: {}\n", ::vt::theContext()->getNode(), od_factor, static_cast< int >( split_method ) );

  // We should be able to set the data correctly
  SECTION( "set_data" )
  {
    ::vt::runInEpochCollective(
      "set_data", [&]() {
        vt::runInEpochCollective( "set_data.init", [&]() {
          obj.set_entity_data( elements, split_method );
          obj.init_broadphase();

          obj.for_each_tree( test_trees{ od_factor } );
        } );

        vt::runInEpochCollective( "set_data.init.check", [&]() {
          auto local_patches = obj.local_patches();
          REQUIRE( local_patches.size() == od_factor );
          std::size_t total_num_elements
            = std::transform_reduce( local_patches.begin(), local_patches.end(), 0UL, std::plus{},
                                     []( const auto &_patch ) { return _patch.size(); } );
          bvh::patch<>::kdop_type k;
          for ( auto &&p : local_patches )
          {
            bvh::vt::debug( "{}: patch {}: {} (centroid={})\n", ::vt::theContext()->getNode(), p.global_id(), p.kdop(),
                            p.centroid() );
            CHECK( !std::isnan( p.centroid().x() ) );
            CHECK( !std::isnan( p.centroid().y() ) );
            CHECK( !std::isnan( p.centroid().z() ) );
            k.union_with( p.kdop() );
          }

          REQUIRE( k == approx( bounds, 0.00001 ) );

          auto r = ::vt::theCollective()->global();
          r->reduce< verify_num_elements, ::vt::collective::PlusOp >( ::vt::Node{0}, total_num_elements );
        } );

        // Data should be updateable
        vt::runInEpochCollective( "set_data.update", [&]() {
          obj.set_entity_data( update_elements, split_method );
          obj.init_broadphase();

          obj.for_each_tree( test_sing_trees{ od_factor } );
        } );

        vt::runInEpochCollective( "set_data.update.check", [&]() {
          auto local_patches = obj.local_patches();
          std::size_t total_num_elements
            = std::transform_reduce( local_patches.begin(), local_patches.end(), 0UL, std::plus{},
                                     []( const auto &_patch ) { return _patch.size(); } );
          bvh::patch<>::kdop_type k;
          for ( auto &&p : local_patches )
          {
            bvh::vt::debug( "{}: patch {}: {} (centroid={})\n", ::vt::theContext()->getNode(), p.global_id(), p.kdop(),
                            p.centroid() );
            CHECK( !std::isnan( p.centroid().x() ) );
            CHECK( !std::isnan( p.centroid().y() ) );
            CHECK( !std::isnan( p.centroid().z() ) );
            k.union_with( p.kdop() );
          }

          REQUIRE( k == approx( update_bounds, 0.00001 ) );

          auto r = ::vt::theCollective()->global();
          r->reduce< verify_num_elements, ::vt::collective::PlusOp >( ::vt::Node{0}, total_num_elements );
        } );

        obj.end_phase();
      } );
  }

  bvh::view< Element * > empty_elements( "empty_elements", 0 );
  // Handle empt elements
  SECTION( "set_empty_data" )
  {
    ::vt::runInEpochCollective( "set_empty_data", [&]() {
      vt::runInEpochCollective( "set_empty_data.init", [&]() {
        obj.set_entity_data( empty_elements, split_method );
        obj.init_broadphase();

        obj.for_each_tree( test_empty_trees{ od_factor } );
      } );

      vt::runInEpochCollective( "set_data.init.check", [&]() {
        auto local_patches = obj.local_patches();
        REQUIRE( local_patches.size() == od_factor );
        std::size_t total_num_elements
          = std::transform_reduce( local_patches.begin(), local_patches.end(), 0UL, std::plus{},
                                   []( const auto &_patch ) { return _patch.size(); } );

        auto r = ::vt::theCollective()->global();
        r->reduce< verify_empty_elements, ::vt::collective::PlusOp >( ::vt::Node{ 0 }, total_num_elements );
      } );

      // Data should be updateable
      vt::runInEpochCollective( "set_data.update", [&]() {
        obj.set_entity_data( empty_elements, split_method );
        obj.init_broadphase();

        obj.for_each_tree( test_empty_trees{ od_factor } );
      } );

      vt::runInEpochCollective( "set_data.update.check", [&]() {
        auto local_patches = obj.local_patches();
        std::size_t total_num_elements
          = std::transform_reduce( local_patches.begin(), local_patches.end(), 0UL, std::plus{},
                                   []( const auto &_patch ) { return _patch.size(); } );

        auto r = ::vt::theCollective()->global();
        r->reduce< verify_empty_elements, ::vt::collective::PlusOp >( ::vt::Node{ 0 }, total_num_elements );
      } );

      obj.end_phase();
    } );
  }
}

TEST_CASE( "collision_object broadphase", "[vt]")
{
  auto split_method
    = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis, bvh::split_algorithm::clustering );
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( "collision_object.broadphase", [&]() {
    auto rank = ::vt::theContext()->getNode();

    auto elements = build_element_grid( 2, 3, 2, rank * 12 );
    obj.set_entity_data( elements, split_method );
    obj.init_broadphase();

    auto elements2 = build_element_grid( 1, 1, 1, rank );
    obj2.set_entity_data( elements2, split_method );
    obj2.init_broadphase();

    obj.broadphase( obj2 );

    obj.end_phase();
    obj2.end_phase();
  } );
}

TEST_CASE( "collision_object multiple broadphase", "[vt]")
{
  auto split_method
    = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis, bvh::split_algorithm::clustering );
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  auto rank = ::vt::theContext()->getNode();
  auto elements = build_element_grid( 2, 3, 2, rank * 12 );
  auto elements2 = build_element_grid( 1, 1, 1, rank );
  obj2.set_entity_data( elements2, split_method );

  ::vt::runInEpochCollective( "collision_object.multiple_broadphase", [&]() {
    obj.set_entity_data( elements, split_method );
    obj.init_broadphase();
    obj2.init_broadphase();

    obj.broadphase( obj2 );
    obj.broadphase( obj2 );

    obj.end_phase();
    obj2.end_phase();
  } );
}

struct narrowphase_result
{
  narrowphase_result( std::size_t _i )
      : idx( _i )
  {}

  std::size_t idx;
};

bool operator<( const narrowphase_result &_lhs, const narrowphase_result &_rhs )
{
  return _lhs.idx < _rhs.idx;
}

struct detailed_narrowphase_result
{
  std::size_t patch_p = 0;
  std::size_t element_p = 0;
  std::size_t patch_q = 0;
  std::size_t element_q = 0;
};

template<>
struct checkpoint::ByteCopyNonIntrusive< detailed_narrowphase_result >
{
  using isByteCopyable = std::true_type;
};

bool operator<( const detailed_narrowphase_result &_lhs, const detailed_narrowphase_result &_rhs )
{
  if ( _lhs.patch_p != _rhs.patch_p )
    return _lhs.patch_p < _rhs.patch_p;
  if ( _lhs.element_p != _rhs.element_p )
    return _lhs.element_p < _rhs.element_p;
  if ( _lhs.patch_q != _rhs.patch_q )
    return _lhs.patch_q < _rhs.patch_q;

  return _lhs.element_q < _rhs.element_q;
}

void verify_single_narrowphase(
  const bvh::vt::reducable_vector< detailed_narrowphase_result > &_res,
  const std::vector< std::size_t > &objSizes )
{
  using collision_pair_t = std::pair< std::size_t, std::size_t >;
  auto results = _res.vec;

  // Define the problem
  const std::size_t numObjs = objSizes.size();
  const bool isSelfContact = numObjs == 1;
  const std::size_t numNodes = ::vt::theContext()->getNumNodes();

  // Determine the expected number of collisions
  std::size_t expectedNumCollisions = 0;
  if ( isSelfContact ) expectedNumCollisions = numNodes * (objSizes[ 0 ] - numNodes);
  else {
    for ( std::size_t i = 0; i < numObjs; i++ ) {
      for ( std::size_t j = ( i + 1 ); j < numObjs; j++ ) {
        expectedNumCollisions += ( objSizes[ i ] * objSizes[ j ] );
      }
    }
  }

  // Confirm the correct number of collisions were found
  CHECK( results.size() == expectedNumCollisions );

  // Construct a vector of unordered pairs (min, max) representing the two colliding element global ids
  std::vector< collision_pair_t > collisionPairs;
  for ( std::size_t i = 0; i < results.size(); ++i ) {
    const auto& res = results[ i ];
    std::size_t id1 = res.element_p;
    std::size_t id2 = res.element_q;

    if ( id1 > id2 ) std::swap( id1, id2 );
      collisionPairs.push_back( { id1, id2 } );

    bvh::vt::debug( "Collision {}: ( patch {}, element {} ) x ( patch {}, element {} ) -> unordered pair = {{ {}, {} }}\n",
      i, res.patch_p, res.element_p, res.patch_q, res.element_q, id1, id2 );
  }

  // Generate the expected pairs -- all elements of all object(s) in contact
  std::vector< collision_pair_t > expectedPairs;
  if ( isSelfContact ) {
    for (std::size_t i = 0; i < numNodes; i++ ) {
      for ( std::size_t j = numNodes; j < objSizes[ 0 ]; j++ ) {
        expectedPairs.push_back( { i, j } );
      }
    }
  } else {
    for ( std::size_t i = 0; i < numObjs; i++ ) {
      for ( std::size_t j = ( i + 1 ); j < numObjs; j++ ) {
        for ( std::size_t e0 = 0; e0 < objSizes[ i ]; e0++ ) {
          for ( std::size_t e1 = 0; e1 < objSizes[ j ]; e1++ ) {
            expectedPairs.push_back( { e0, e1 } );
          }
        }
      }
    }
  }

  // Swap pairs so the lowest id comes first (so it matches with the results)
  for ( auto& p : expectedPairs ) {
    if ( p.second < p.first )
      std::swap( p.first, p.second );
  }

  // Sort both vectors (first by first element, then by second) to compare them regardless of order
  auto pairComparator = []( const collision_pair_t & a,
                            const collision_pair_t & b ) {
    return ( a.first < b.first ) || (( a.first == b.first ) && ( a.second < b.second ));
  };
  std::sort( collisionPairs.begin(), collisionPairs.end(), pairComparator );
  std::sort( expectedPairs.begin(), expectedPairs.end(), pairComparator );

  bvh::vt::debug( "Sorted collision pairs:\n" );
  for ( const auto& p : collisionPairs ) {
    bvh::vt::debug( "  {{ {}, {} }}\n", p.first, p.second );
  }
  bvh::vt::debug( "Expected collision pairs:\n" );
  for ( const auto& p : expectedPairs ) {
    bvh::vt::debug( "  {{ {}, {} }}\n", p.first, p.second );
  }

  // Assert all collisions are found correctly
  REQUIRE( collisionPairs.size() == expectedPairs.size() );
  for ( std::size_t i = 0; i < expectedPairs.size(); ++i ) {
    REQUIRE( collisionPairs[ i ].first == expectedPairs[ i ].first );
    REQUIRE( collisionPairs[ i ].second == expectedPairs[ i ].second );
  }
}

TEST_CASE( "collision_object narrowphase", "[vt]")
{
  auto split_method
    = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis, bvh::split_algorithm::clustering );

  bvh::vt::debug("{}: split method: {}\n", ::vt::theContext()->getNode(), static_cast< int >( split_method ) );

  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();
  bvh::vt::reducable_vector< detailed_narrowphase_result > results;

  ::vt::runInEpochCollective( "collision_object.narrowphase", [&]() {
    world.start_iteration();

    auto rank = ::vt::theContext()->getNode();

    auto elements = build_element_grid( 1, 1, 1, rank );
    obj.set_entity_data( elements, split_method );
    obj.init_broadphase();

    auto elements2 = build_element_grid( 2, 3, 2, rank * 12 );
    obj2.set_entity_data( elements2, split_method );
    obj2.init_broadphase();

    world.set_narrowphase_functor< Element >( []( const bvh::broadphase_collision< Element > &_a,
                                                  const bvh::broadphase_collision< Element > &_b ) {
      auto res = bvh::narrowphase_result_pair();
      res.a = bvh::narrowphase_result( sizeof( detailed_narrowphase_result ));
      res.b = bvh::narrowphase_result( sizeof( detailed_narrowphase_result ));
      auto &resa = static_cast< bvh::typed_narrowphase_result< detailed_narrowphase_result > & >( res.a );

      REQUIRE( _a.object.id() == 0 );
      REQUIRE( _b.object.id() == 1 );
      // First patch only has one element, ever
      REQUIRE( _a.elements.size() == 1 );
      // Second patch number of elements depends on the split algorithm, so not tested

      // Global id of the first patch should be the node from whence it came
      REQUIRE( _a.elements[0].global_id() < static_cast< std::size_t >( ::vt::theContext()->getNumNodes() ) );

      for ( auto &&e: _b.elements ) {
        REQUIRE( e.global_id() < static_cast< std::size_t >( ::vt::theContext()->getNumNodes() * 12 ) );
        resa.emplace_back( detailed_narrowphase_result{ _a.meta.global_id(), _a.elements[0].global_id(),
                                                        _b.meta.global_id(), e.global_id() } );
      }

      return res;
    } );

    obj.broadphase( obj2 );

    results.vec.clear();
    obj.for_each_result< detailed_narrowphase_result >( [&]( const detailed_narrowphase_result &_res ) {
      results.vec.emplace_back( _res );
    } );

    world.finish_iteration();
  } );

  static_assert( std::is_default_constructible_v< detailed_narrowphase_result > );

  ::vt::runInEpochCollective( "collision_object.narrowphase.verify", [&]() {
    auto r = ::vt::theCollective()->global();
    const std::vector<std::size_t> numEltsPerObj = { 1, 12 };
    r->reduce< verify_single_narrowphase, ::vt::collective::PlusOp >( ::vt::Node{ 0 }, results, numEltsPerObj );
  } );
}

TEST_CASE( "collision_object narrowphase three objects", "[vt]" ) {
  bvh::collision_world world( 2 );
  auto &obj0 = world.create_collision_object();
  auto &obj1 = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  auto split_method = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis );

  bvh::vt::reducable_vector< detailed_narrowphase_result > results;

  ::vt::runInEpochCollective( "collision_object.narrowphase", [&]() {
    world.start_iteration();
    auto rank = ::vt::theContext()->getNode();

    auto elements0 = build_element_grid( 1, 1, 1, rank, 0.0 );
    obj0.set_entity_data( elements0, split_method );
    obj0.init_broadphase();

    auto elements1 = build_element_grid( 1, 2, 1, rank * 2, 0.0 );
    obj1.set_entity_data( elements1, split_method );
    obj1.init_broadphase();

    auto elements2 = build_element_grid( 1, 1, 2, rank * 2, 0.0 );
    obj2.set_entity_data( elements2, split_method );
    obj2.init_broadphase();

    bvh::vt::debug( "Object 0 initialized with {} element(s):\n", elements0.extent( 0 ) );
    for ( std::size_t i = 0; i < elements0.extent( 0 ); i++ ) {
      bvh::vt::debug( "  Element {}: global_id = {}\n", i, elements0( i ).global_id() );
    }

    bvh::vt::debug( "Object 1 initialized with {} element(s):\n", elements1.extent( 0 ) );
    for ( std::size_t i = 0; i < elements1.extent( 0 ); i++ ) {
      bvh::vt::debug( "  Element {}: global_id = {}\n", i, elements1( i ).global_id() );
    }

    bvh::vt::debug( "Object 2 initialized with {} element(s):\n", elements2.extent( 0 ) );
    for ( std::size_t i = 0; i < elements2.extent( 0 ); i++ ) {
      bvh::vt::debug( "  Element {}: global_id = {}\n", i, elements2( i ).global_id() );
    }
    CHECK( elements0.extent( 0 ) == 1 );
    CHECK( elements1.extent( 0 ) == 2 );
    CHECK( elements2.extent( 0 ) == 2 );

    world.set_narrowphase_functor< Element >( []( const bvh::broadphase_collision< Element > &_a,
      const bvh::broadphase_collision< Element > &_b ) {
      auto res = bvh::narrowphase_result_pair();
      res.a = bvh::narrowphase_result( sizeof( detailed_narrowphase_result ));
      res.b = bvh::narrowphase_result( sizeof( detailed_narrowphase_result ));
      auto &resa = static_cast< bvh::typed_narrowphase_result< detailed_narrowphase_result > & >( res.a );

      for ( auto &&e: _b.elements ) {
        resa.emplace_back( detailed_narrowphase_result{ _a.meta.global_id(), _a.elements[0].global_id(),
                  _b.meta.global_id(), e.global_id() } );
      }

      return res;
    } );

    obj0.broadphase( obj1 );
    obj1.broadphase( obj2 );
    obj2.broadphase( obj0 );

    results.vec.clear();
    obj0.for_each_result< detailed_narrowphase_result >( [&]( const detailed_narrowphase_result &_res ) {
      results.vec.emplace_back( _res );
    } );

    obj1.for_each_result< detailed_narrowphase_result >( [&]( const detailed_narrowphase_result &_res ) {
      results.vec.emplace_back( _res );
    } );

    obj2.for_each_result< detailed_narrowphase_result >( [&]( const detailed_narrowphase_result &_res ) {
      results.vec.emplace_back( _res );
    } );

    world.finish_iteration();
  } );

  static_assert( std::is_default_constructible_v< detailed_narrowphase_result > );
  ::vt::runInEpochCollective( "collision_object.narrowphase.verify", [&]() {
    auto r = ::vt::theCollective()->global();
    const std::vector< std::size_t > numEltsPerObj = { 1, 2, 2 };
    r->reduce< verify_single_narrowphase, ::vt::collective::PlusOp >( ::vt::Node{ 0 }, results, numEltsPerObj );
  } );
}

TEST_CASE( "collision_object narrowphase self contact", "[vt]" ) {
  bvh::collision_world world( 2 );
  auto &obj0 = world.create_collision_object();

  auto split_method = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis );

  bvh::vt::reducable_vector< detailed_narrowphase_result > results;

  ::vt::runInEpochCollective( "collision_object.narrowphase", [&]() {
    world.start_iteration();
    auto rank = ::vt::theContext()->getNode();
    auto numNodes = ::vt::theContext()->getNumNodes();

    auto elements0 = build_element_grid( 1, 1, 1, rank, 0.0 );
    auto elements1 = build_element_grid( 1, 2, 1, numNodes + (rank * 2), 0.0 );
    CHECK( elements0.extent( 0 ) == 1 );
    CHECK( elements1.extent( 0 ) == 2 );

    auto combined = bvh::view< Element * >( "combined elements", elements0.size() + elements1.size() );
    CHECK( combined.extent( 0 ) == elements0.extent( 0 ) + elements1.extent( 0 ) );
    Kokkos::deep_copy( Kokkos::subview( combined, Kokkos::make_pair( size_t{ 0 }, elements0.size() ) ), elements0 );
    Kokkos::deep_copy(
      Kokkos::subview( combined, Kokkos::make_pair( elements0.size(), elements0.size() + elements1.size() ) ),
      elements1 );
    obj0.set_entity_data( combined, split_method );
    obj0.init_broadphase();

    bvh::vt::debug( "Object 0 initialized with {} element(s):\n", combined.extent( 0 ) );
    for ( std::size_t i = 0; i < combined.extent( 0 ); i++ ) {
      bvh::vt::debug( "  Element {}: global_id = {}\n", i, combined( i ).global_id() );
    }

    world.set_narrowphase_functor< Element >( []( const bvh::broadphase_collision< Element > &_a,
      const bvh::broadphase_collision< Element > &_b ) {
      auto res = bvh::narrowphase_result_pair();
      res.a = bvh::narrowphase_result( sizeof( detailed_narrowphase_result ));
      res.b = bvh::narrowphase_result( sizeof( detailed_narrowphase_result ));
      auto &resa = static_cast< bvh::typed_narrowphase_result< detailed_narrowphase_result > & >( res.a );

      auto narrowphase_filter = []( const Element& _e0, const Element& _e1 ) -> bool
      {
        auto get_element_grid_id = []( const Element& _e ) -> int
        {
          const std::size_t numNodes = ::vt::theContext()->getNumNodes();
          return _e.global_id() < numNodes ? 0 : 1;
        };
        const bool areOverlapping = overlap( _e0.kdop(), _e1.kdop() );
        const bool areOrdered = _e0.global_id() < _e1.global_id();
        const bool areOnDifferentGrids = get_element_grid_id( _e0 ) != get_element_grid_id( _e1 );
        return areOverlapping && areOrdered && areOnDifferentGrids;
      };

      REQUIRE(_a.object.id() == _b.object.id());

      for ( auto &&b_elt: _b.elements ) {
        for ( auto &&a_elt: _a.elements ) {
          if ( narrowphase_filter( a_elt, b_elt ) ) {
            resa.emplace_back( detailed_narrowphase_result{ _a.meta.global_id(), a_elt.global_id(),
                                                            _b.meta.global_id(), b_elt.global_id() } );
          }
        }
      }

      return res;
    } );

    obj0.broadphase( obj0 );

    results.vec.clear();
    obj0.for_each_result< detailed_narrowphase_result >( [&]( const detailed_narrowphase_result &_res ) {
      results.vec.emplace_back( _res );
    } );

    world.finish_iteration();
  } );

  static_assert( std::is_default_constructible_v< detailed_narrowphase_result > );
  ::vt::runInEpochCollective( "collision_object.narrowphase.verify", [&]() {
    auto r = ::vt::theCollective()->global();
    const std::vector< std::size_t > numEltsPerObj = { 3 };
    r->reduce< verify_single_narrowphase, ::vt::collective::PlusOp >( ::vt::Node{ 0 }, results, numEltsPerObj );
  } );
}

TEST_CASE( "collision_object narrowphase multi-iteration", "[vt]")
{
  auto split_method
    = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis, bvh::split_algorithm::clustering );

  bvh::vt::debug("{}: split method: {}\n", ::vt::theContext()->getNode(), static_cast< int >( split_method ) );

  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  std::vector< narrowphase_result > new_results;
  std::vector< narrowphase_result > new_results2;
  std::vector< narrowphase_result > old_results, old_results2;

  ::vt::runInEpochCollective( "collision_object.multiple_narrowphase", [&]() {
    for ( std::size_t i = 0; i < 8; ++i ) {
      world.start_iteration();

      auto rank = ::vt::theContext()->getNode();

      auto elements = build_element_grid( 1, 1, 1, rank );
      obj.set_entity_data( elements, split_method );
      obj.init_broadphase();

      auto elements2 = build_element_grid( 2, 3, 2, rank * 12 );
      obj2.set_entity_data( elements2, split_method );
      obj2.init_broadphase();

      world.set_narrowphase_functor< Element >(
        []( const bvh::broadphase_collision< Element > &_a, const bvh::broadphase_collision< Element > &_b ) {
        auto res = bvh::narrowphase_result_pair();
        res.a = bvh::narrowphase_result( sizeof( narrowphase_result ));
        res.b = bvh::narrowphase_result( sizeof( narrowphase_result ));
        auto &resa = static_cast< bvh::typed_narrowphase_result< narrowphase_result > & >( res.a );
        auto &resb = static_cast< bvh::typed_narrowphase_result< narrowphase_result > & >( res.b );

        REQUIRE( _a.object.id() == 0 );
        REQUIRE( _b.object.id() == 1 );
        // First patch only has one element, ever
        REQUIRE( _a.elements.size() == 1 );
        // Second patch number of elements depends on the split algorithm, so not tested

        // Global id of the first patch should be the node from whence it came
        REQUIRE( _a.elements[0].global_id() < static_cast< std::size_t >( ::vt::theContext()->getNumNodes() ) );
        bvh::vt::debug("{}: intersect patch ({}, {}) with ({}, {})\n",
                        ::vt::theContext()->getNode(),
                        _a.object.id(), _a.patch_id,
                        _b.object.id(), _b.patch_id );

        for ( auto &&e: _b.elements )
        {
          CHECK( e.global_id() < static_cast< std::size_t >( ::vt::theContext()->getNumNodes() * 12 ) );
          bvh::vt::debug("{}: intersect result ({}, {}, {}) with ({}, {}, {})\n",
                         ::vt::theContext()->getNode(),
                         _a.object.id(), _a.patch_id, _a.elements[0].global_id(),
                         _b.object.id(), _b.patch_id, e.global_id() );
          resa.emplace_back( e.global_id());
          resb.emplace_back( _a.elements[0].global_id());
        }

        return res;
      } );

      obj.broadphase( obj2 );

      // results for obj
      new_results.clear();
      obj.for_each_result< narrowphase_result >( [rank, &new_results]( const narrowphase_result &_res ) {
        new_results.emplace_back( _res );
        bvh::vt::debug("{}: got result {}\n", rank, _res.idx );
      } );

      // results for obj2
      new_results2.clear();
      obj2.for_each_result< narrowphase_result >( [&new_results2]( const narrowphase_result &_res ) {
        new_results2.emplace_back( _res );
      } );

      world.finish_iteration();

      // Sort results so we get them in a consistent order
      std::sort( new_results.begin(), new_results.end() );
      std::sort( new_results2.begin(), new_results2.end() );

      if ( i > 0 ) {
        REQUIRE( old_results.size() == new_results.size());
        for ( std::size_t j = 0; j < old_results.size(); ++j )
          REQUIRE( old_results.at( j ).idx == new_results.at( j ).idx );
      }

      old_results = new_results;

      if ( i > 0 ) {
        REQUIRE( old_results2.size() == new_results2.size());
        for ( std::size_t j = 0; j < old_results2.size(); ++j )
          REQUIRE( old_results2.at( j ).idx == new_results2.at( j ).idx );
      }

      old_results2 = new_results2;
    }
  } );
}

TEST_CASE( "collision_object narrowphase no overlap multi-iteration", "[vt]")
{
  auto split_method
    = GENERATE( bvh::split_algorithm::geom_axis, bvh::split_algorithm::ml_geom_axis, bvh::split_algorithm::clustering );
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( "collision_object.multiple_narrowphase_no_overlap", [&]() {
    std::vector< narrowphase_result > old_results, old_results2;

    for ( std::size_t i = 0; i < 8; ++i ) {
      world.start_iteration();

      auto rank = ::vt::theContext()->getNode();

      auto elements = build_element_grid( 1, 1, 1, rank );
      obj.set_entity_data( elements, split_method );
      obj.init_broadphase();

      auto elements2 = build_element_grid( 2, 3, 2, rank * 12 , 16.0);
      obj2.set_entity_data( elements2, split_method );
      obj2.init_broadphase();

      world.set_narrowphase_functor< Element >( []( const bvh::broadphase_collision< Element > &_a,
                                                    const bvh::broadphase_collision< Element > &_b ) {
        auto res = bvh::narrowphase_result_pair();
        res.a = bvh::narrowphase_result( sizeof( narrowphase_result ));
        res.b = bvh::narrowphase_result( sizeof( narrowphase_result ));
        auto &resa = static_cast< bvh::typed_narrowphase_result< narrowphase_result > & >( res.a );
        auto &resb = static_cast< bvh::typed_narrowphase_result< narrowphase_result > & >( res.b );

        REQUIRE( _a.object.id() == 0 );
        REQUIRE( _b.object.id() == 1 );
        // First patch only has one element, ever
        REQUIRE( _a.elements.size() == 1 );
        // Second patch number of elements depends on the split algorithm, so not tested

        // Global id of the first patch should be the node from whence it came
        REQUIRE( _a.elements[0].global_id() < static_cast< std::size_t >( ::vt::theContext()->getNumNodes() ) );

        for ( auto &&e: _b.elements ) {
          CHECK( e.global_id() < static_cast< std::size_t >( ::vt::theContext()->getNumNodes() * 12 ) );
          resa.emplace_back( e.global_id());
          resb.emplace_back( _a.elements[0].global_id());
        }

        return res;
      } );

      obj.broadphase( obj2 );

      std::vector< narrowphase_result > new_results;
      obj.for_each_result< narrowphase_result >(
        [&new_results]( const narrowphase_result &_res ) { new_results.emplace_back( _res ); } );

      std::vector< narrowphase_result > new_results2;
      obj2.for_each_result< narrowphase_result >( [&new_results2]( const narrowphase_result &_res ) {
        new_results2.emplace_back( _res );
      } );

      world.finish_iteration();

      if ( i > 0 ) {
        REQUIRE( old_results.size() == new_results.size());
        for ( std::size_t j = 0; j < old_results.size(); ++j )
          REQUIRE( old_results.at( j ).idx == new_results.at( j ).idx );
      }

      old_results = new_results;

      if ( i > 0 ) {
        REQUIRE( old_results2.size() == new_results2.size());
        for ( std::size_t j = 0; j < old_results2.size(); ++j )
          REQUIRE( old_results2.at( j ).idx == new_results2.at( j ).idx );
      }

      old_results2 = new_results2;

    }
  } );
}

TEST_CASE( "set entity data benchmark", "[vt][!benchmark]")
{
  std::size_t od_factor = GENERATE(1, 2, 4, 8, 16, 32, 64, 128, 256);
  CAPTURE(od_factor);

  bvh::collision_world world( od_factor );

  auto &obj = world.create_collision_object();

  auto rank = ::vt::theContext()->getNode();
  auto elements = build_element_grid( 256, 64, 64, rank );

  SECTION( "geom_axis" )
  {
    BENCHMARK( "geom_axis" )
    {
      ::vt::runInEpochCollective( "set_entity_data_benchmark.splitting", [&]() {
        world.start_iteration();

        obj.set_entity_data( elements, bvh::split_algorithm::geom_axis );
        obj.init_broadphase();

        world.finish_iteration();
      } );
    };
  }

  SECTION( "ml_geom_axis" )
  {
    BENCHMARK( "ml_geom_axis" )
    {
      ::vt::runInEpochCollective( "set_entity_data_benchmark.splitting", [&]() {
        world.start_iteration();

        obj.set_entity_data( elements, bvh::split_algorithm::ml_geom_axis );
        obj.init_broadphase();

        world.finish_iteration();
      } );
    };
  }

  SECTION("clustering")
  {

    BENCHMARK("clustering")
    {
      ::vt::runInEpochCollective( "set_entity_data_benchmark.clustering", [&]() {
        world.start_iteration();

        obj.set_entity_data( elements, bvh::split_algorithm::clustering );
        obj.init_broadphase();

        world.finish_iteration();
      } );
    };
  }
}
