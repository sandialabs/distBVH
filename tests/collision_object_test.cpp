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
#include <bvh/collision_object.hpp>
#include <bvh/collision_world.hpp>
#include <bvh/util/epoch.hpp>
#include <bvh/vt/print.hpp>
#include <vt/termination/epoch_guard.h>

void test_trees( const bvh::snapshot_tree &_tree )
{
  auto nranks = ::vt::theContext()->getNumNodes();
  REQUIRE( _tree.count() == 2 * nranks );
}

void test_sing_trees( const bvh::snapshot_tree &_tree )
{
  auto nranks = ::vt::theContext()->getNumNodes();
  REQUIRE( _tree.count() == 2 * nranks );

  for ( auto &&l : _tree.leafs() )
  {
    REQUIRE( l.kdop().centroid() == bvh::m::vec3d::zeros() );
  }
}

TEST_CASE( "collision_object init", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();

  ::vt::runInEpochCollective( [&]() {
    auto rank = ::vt::theContext()->getNode();

    // We should be able to set the data correctly
    SECTION( "set_data" )
    {
      vt::runInEpochCollective( [&]() {
        auto vec = buildElementGrid( 2, 3, 2, rank * 12 );

        obj.set_entity_data( bvh::make_const_span( vec ));
        obj.init_broadphase();

        obj.for_each_tree( &test_trees );
      } );
    }

      // Data should be updateable
    SECTION( "update_data" )
    {
      vt::runInEpochCollective( [&]() {
        auto sing_vec = std::vector< Element >{};
        sing_vec.reserve( 12 );
        for ( std::size_t i = 0; i < 12; ++i ) {
          sing_vec.emplace_back( i );
          sing_vec.back().setVertices( { bvh::m::vec3d::zeros() } );
        }

        obj.set_entity_data( bvh::make_const_span( sing_vec ));
        obj.init_broadphase();

        obj.for_each_tree( &test_sing_trees );
      } );
    }

    obj.end_phase();
  } );
}

TEST_CASE( "collision_object broadphase", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( [&]() {
    auto rank = ::vt::theContext()->getNode();

    auto vec = buildElementGrid( 2, 3, 2, rank * 12 );
    obj.set_entity_data( bvh::make_const_span( vec ));
    obj.init_broadphase();

    auto vec2 = buildElementGrid( 1, 1, 1, rank );
    obj2.set_entity_data( bvh::make_const_span( vec2 ));
    obj2.init_broadphase();

    obj.broadphase( obj2 );

    obj.end_phase();
    obj2.end_phase();
  } );
}

TEST_CASE( "collision_object multiple broadphase", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( [&]() {
    auto rank = ::vt::theContext()->getNode();

    auto vec = buildElementGrid( 2, 3, 2, rank * 12 );
    obj.set_entity_data( bvh::make_const_span( vec ));
    obj.init_broadphase();

    auto vec2 = buildElementGrid( 1, 1, 1, rank );
    obj2.set_entity_data( bvh::make_const_span( vec2 ));
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

TEST_CASE( "collision_object narrowphase", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( [&]() {
    world.start_iteration();

    auto rank = ::vt::theContext()->getNode();

    auto vec = buildElementGrid( 1, 1, 1, rank );
    obj.set_entity_data( bvh::make_const_span( vec ));
    obj.init_broadphase();

    auto vec2 = buildElementGrid( 2, 3, 2, rank * 12 );
    obj2.set_entity_data( bvh::make_const_span( vec2 ));
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
      REQUIRE( _a.elements[0].global_id() < ::vt::theContext()->getNumNodes());

      for ( auto &&e: _b.elements ) {
        REQUIRE( e.global_id() < ::vt::theContext()->getNumNodes() * 12 );
        resa.emplace_back( e.global_id());
        resb.emplace_back( _a.elements[0].global_id());
      }

      return res;
    } );

    obj.broadphase( obj2 );

    obj.for_each_result< narrowphase_result >( []( const narrowphase_result &_res ) {
      //std::cout << "collision with " << _res.idx << '\n';
    } );

    world.finish_iteration();
  } );
}

TEST_CASE( "collision_object narrowphase multi-iteration", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( [&]() {
    std::vector< narrowphase_result > old_results, old_results2;

    for ( std::size_t i = 0; i < 1000; ++i ) {
      world.start_iteration();

      auto rank = ::vt::theContext()->getNode();

      auto vec = buildElementGrid( 1, 1, 1, rank );
      obj.set_entity_data( bvh::make_const_span( vec ));
      obj.init_broadphase();

      auto vec2 = buildElementGrid( 2, 3, 2, rank * 12 );
      obj2.set_entity_data( bvh::make_const_span( vec2 ));
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
        REQUIRE( _a.elements[0].global_id() < ::vt::theContext()->getNumNodes());

        for ( auto &&e: _b.elements ) {
          CHECK( e.global_id() < ::vt::theContext()->getNumNodes() * 12 );
          resa.emplace_back( e.global_id());
          resb.emplace_back( _a.elements[0].global_id());
        }

        return res;
      } );

      obj.broadphase( obj2 );

      std::vector< narrowphase_result > new_results;
      obj.for_each_result< narrowphase_result >( [rank, i, &new_results]( const narrowphase_result &_res ) {
        new_results.emplace_back( _res );
        bvh::vt::debug("{}: got result {}\n", rank, _res.idx );
      } );

      std::vector< narrowphase_result > new_results2;
      obj2.for_each_result< narrowphase_result >( [i, &new_results2]( const narrowphase_result &_res ) {
        std::cout << "got a result!!!\n";
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

      std::cout << "new results size " << new_results.size() << '\n';
      old_results = new_results;

      if ( i > 0 ) {
        REQUIRE( old_results2.size() == new_results2.size());
        for ( std::size_t j = 0; j < old_results2.size(); ++j )
          REQUIRE( old_results2.at( j ).idx == new_results2.at( j ).idx );
      }

      std::cout << "new results2 size " << new_results2.size() << '\n';
      old_results2 = new_results2;

    }
  } );
}

TEST_CASE( "collision_object narrowphase no overlap multi-iteration", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();
  auto &obj2 = world.create_collision_object();

  ::vt::runInEpochCollective( [&]() {
    std::vector< narrowphase_result > old_results, old_results2;

    for ( std::size_t i = 0; i < 8; ++i ) {
      world.start_iteration();

      auto rank = ::vt::theContext()->getNode();

      auto vec = buildElementGrid( 1, 1, 1, rank );
      obj.set_entity_data( bvh::make_const_span( vec ));
      obj.init_broadphase();

      auto vec2 = buildElementGrid( 2, 3, 2, rank * 12 , 16.0);
      obj2.set_entity_data( bvh::make_const_span( vec2 ));
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
        REQUIRE( _a.elements[0].global_id() < ::vt::theContext()->getNumNodes());

        for ( auto &&e: _b.elements ) {
          CHECK( e.global_id() < ::vt::theContext()->getNumNodes() * 12 );
          resa.emplace_back( e.global_id());
          resb.emplace_back( _a.elements[0].global_id());
        }

        return res;
      } );

      obj.broadphase( obj2 );

      std::vector< narrowphase_result > new_results;
      obj.for_each_result< narrowphase_result >( [&old_results, i, &new_results]( const narrowphase_result &_res ) {
        new_results.emplace_back( _res );
      } );

      std::vector< narrowphase_result > new_results2;
      obj2.for_each_result< narrowphase_result >( [&old_results2, i, &new_results2]( const narrowphase_result &_res ) {
        std::cout << "got a result!!!\n";
        new_results2.emplace_back( _res );
      } );

      world.finish_iteration();

      if ( i > 0 ) {
        REQUIRE( old_results.size() == new_results.size());
        for ( std::size_t j = 0; j < old_results.size(); ++j )
          REQUIRE( old_results.at( j ).idx == new_results.at( j ).idx );
      }

      std::cout << "new results size " << new_results.size() << '\n';
      old_results = new_results;

      if ( i > 0 ) {
        REQUIRE( old_results2.size() == new_results2.size());
        for ( std::size_t j = 0; j < old_results2.size(); ++j )
          REQUIRE( old_results2.at( j ).idx == new_results2.at( j ).idx );
      }

      std::cout << "new results2 size " << new_results2.size() << '\n';
      old_results2 = new_results2;

    }
  } );
}

TEST_CASE( "set entity data benchmark", "[vt]")
{
  bvh::collision_world world( 2 );

  auto &obj = world.create_collision_object();

  auto rank = ::vt::theContext()->getNode();
  auto vec = buildElementGrid( 128, 128, 128, rank );

  BENCHMARK("bench")
  {
    ::vt::runInEpochCollective( [&]() {
      world.start_iteration();

      obj.set_entity_data( bvh::make_const_span( vec ));
      obj.init_broadphase();

      world.finish_iteration();
    } );
  };
}

