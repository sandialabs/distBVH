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

#include <string>
#include <bvh/math/vec.hpp>
#include <bvh/kdop.hpp>
#include <bvh/node.hpp>
#include <bvh/tree.hpp>
#include <bvh/types.hpp>
#include <bvh/serialization/bvh_serialize.hpp>
#include <bvh/tree_build.hpp>
#include <bvh/collision_object/narrowphase.hpp>
#include <bvh/collision_object/types.hpp>
#include "TestCommon.hpp"

TEMPLATE_TEST_CASE("a single value can be serialized", "[serializer]", int, double, float, std::size_t )
{
  using T = TestType;

  T a{ 5 };

  auto serialized = checkpoint::serialize< T >( a );

  REQUIRE( serialized->getSize() == sizeof( T ) + 12 );

  auto b = *checkpoint::deserialize< T >( serialized->getBuffer() );

  REQUIRE( b == T{5} );
}

TEMPLATE_TEST_CASE("vector serialization", "[serializer][math][vec]", float, double )
{
  using T = TestType;
  bvh::m::vec3< T > a{ T{ 1 }, T{ 5 }, T{ 17 } };

  auto serialized = checkpoint::serialize< bvh::m::vec3< T > >( a );
  REQUIRE( serialized->getSize() == sizeof( T ) * 4 + 12 ); // rounds up to nearest 256 bit

  auto b = *checkpoint::deserialize< bvh::m::vec3< T > >( serialized->getBuffer() );

  REQUIRE( b[0] == T{ 1 } );
  REQUIRE( b[1] == T{ 5 } );
  REQUIRE( b[2] == T{ 17 } );
}

TEST_CASE("kdop serialization", "[serializer][kdop]")
{

}

TEST_CASE("patch serialization", "[serializer][patch]" )
{
  auto k = bvh::bphase_kdop::from_sphere( bvh::m::vec3d{ 17.53, 21.9, 36.0 }, 2.7 );
  bvh::patch<> p( 13, 4096, k, bvh::m::vec3d{ 17.3, 20.6, 33.31 } );

  auto serialized = checkpoint::serialize< bvh::patch<> >( p );
  //REQUIRE( serialized->getSize() == 12 + 6 * sizeof( double ) + 2 * sizeof( std::size_t ) + 4 * sizeof( double ) );

  auto pd = *checkpoint::deserialize< bvh::patch<> >( serialized->getBuffer() );

  REQUIRE( pd.global_id() == p.global_id() );
  REQUIRE( pd.kdop() == p.kdop() );
  REQUIRE( pd.size() == p.size() );
  REQUIRE( pd.centroid() == p.centroid() );
}

namespace
{
  void check_ghost_msg( bvh::collision_object_impl::ghost_msg *_msg )
  {
    // checks the lock is carried through serialization
    REQUIRE( ::vt::messaging::envelopeIsLocked( _msg->env ) );

    auto k = bvh::bphase_kdop::from_sphere( bvh::m::vec3d{ 17.53, 21.9, 36.0 }, 2.7 );
    bvh::patch<> p( 13, 4096, k, bvh::m::vec3d{ 17.3, 20.6, 33.31 } );
    REQUIRE( _msg->meta.global_id() == p.global_id() );
    REQUIRE( _msg->meta.kdop() == p.kdop() );
    REQUIRE( _msg->meta.size() == p.size() );
    REQUIRE( _msg->meta.centroid() == p.centroid() );
    REQUIRE( _msg->origin_node == 13 );
    REQUIRE( _msg->patch_data.size() == 4096 * sizeof( double ) );
    std::vector< double > vals( 4096 );
    std::memcpy( vals.data(), _msg->patch_data.data(), 4096 * sizeof( double ) );

    for ( std::size_t i = 0; i < 4096; ++i )
      REQUIRE( vals[i] == static_cast< double >( i + 1 ) );
  }
}

TEST_CASE("ghost_msg serialization", "[serializer][collision_object][narrowphase]" )
{
  if ( ::vt::theContext()->getNumNodes() > 1 )
  {
    auto ep = ::vt::theTerm()->makeEpochCollective( "ghost_msg_test" );
    ::vt::theMsg()->pushEpoch( ep );
    if ( ::vt::theContext()->getNode() == 0 )
    {
      for ( std::size_t i = 0; i < 10000; ++i )
      {
        auto msg = ::vt::makeMessage< bvh::collision_object_impl::ghost_msg >();
        auto k = bvh::bphase_kdop::from_sphere( bvh::m::vec3d{ 17.53, 21.9, 36.0 }, 2.7 );
        bvh::patch<> p( 13, 4096, k, bvh::m::vec3d{ 17.3, 20.6, 33.31 } );
        msg->meta = p;

        std::vector< double > vals;
        vals.reserve( 4096 );
        for ( std::size_t i = 0; i < 4096; ++i )
          vals.push_back( static_cast< double >( i + 1 ) );
        msg->origin_node = 13;
        Kokkos::resize( msg->patch_data, vals.size() * sizeof( double ) );
        std::memcpy( msg->patch_data.data(), vals.data(), msg->patch_data.size() );

        auto han = ::vt::auto_registry::makeAutoHandler< bvh::collision_object_impl::ghost_msg, check_ghost_msg >();
        ::vt::SerializedMessenger::sendSerialMsg< bvh::collision_object_impl::ghost_msg >( 1, msg.get(), han );
      }
    }
    ::vt::theMsg()->popEpoch();
    ::vt::theTerm()->finishedEpoch( ep );
    ::vt::runSchedulerThrough( ep );

  }
}

namespace
{
  struct initiate_migrate_msg : ::vt::CollectionMessage< bvh::collision_object_impl::narrowphase_patch_collection_type >
  {};
  void migrate_narrowphase_patch( bvh::collision_object_impl::narrowphase_patch_collection_type *_coll, initiate_migrate_msg * )
  {
    auto curr_node = ::vt::theContext()->getNode();
    auto dest = ( curr_node + 1 ) % ::vt::theContext()->getNumNodes();

    _coll->migrate( dest );
  }

  [[maybe_unused]] void narrowphase_patch_check( bvh::collision_object_impl::narrowphase_patch_collection_type *_coll )
  {
    auto k = bvh::bphase_kdop::from_sphere( bvh::m::vec3d{ 17.53, 21.9, 36.0 }, 2.7 );
    bvh::patch<> p( 13, 4096, k, bvh::m::vec3d{ 17.3, 20.6, 33.31 } );
    REQUIRE( _coll->patch_meta.global_id() == p.global_id() );
    REQUIRE( _coll->patch_meta.kdop() == p.kdop() );
    REQUIRE( _coll->patch_meta.size() == p.size() );
    REQUIRE( _coll->patch_meta.centroid() == p.centroid() );
    REQUIRE( _coll->origin_node == 17 );
    REQUIRE( _coll->bytes.size() == 4096 * sizeof( double ) );
    std::vector< double > vals( 4096 );
    std::memcpy( vals.data(), _coll->bytes.data(), 4096 * sizeof( double ) );

    for ( std::size_t i = 0; i < 4096; ++i )
      REQUIRE( vals[i] == static_cast< double >( i + 1 ) );
  }
}

TEST_CASE("narrowphase_patches collection serialization", "[serializer][collision_object][narrowphase][collection]" )
{
  using vt_index = bvh::collision_object_impl::vt_index;
  using narrowphase_patch_collection_type = bvh::collision_object_impl::narrowphase_patch_collection_type;
  if ( ::vt::theContext()->getNumNodes() > 1 )
  {
    auto coll_size = vt_index{ static_cast< std::size_t >( 4 * ::vt::theContext()->getNumNodes() ) };
    auto ep = ::vt::theTerm()->makeEpochCollective( "narrowphase_patch_test" );
    ::vt::theMsg()->pushEpoch( ep );

    auto collection = ::vt::theCollection()->constructCollective< narrowphase_patch_collection_type >(
      coll_size, []( vt_index _idx ) {
      auto ret = std::make_unique< narrowphase_patch_collection_type >();
      auto k = bvh::bphase_kdop::from_sphere( bvh::m::vec3d{ 17.53, 21.9, 36.0 }, 2.7 );
      bvh::patch<> p( 13, 4096, k, bvh::m::vec3d{ 17.3, 20.6, 33.31 } );
      ret->patch_meta = p;
      ret->origin_node = 17;
      std::vector< double > vals;
      vals.reserve( 4096 );
      for ( std::size_t i = 0; i < 4096; ++i )
        vals.push_back( static_cast< double >( i + 1 ) );
      Kokkos::resize( ret->bytes, 4096 * sizeof( double ) );
      std::memcpy( ret->bytes.data(), vals.data(), ret->bytes.size() );
      return ret;
    } );

    for ( std::size_t i = 0; i < 1000; ++i )
    {
      auto ep2 = ::vt::theTerm()->makeEpochCollective( "narrowphase_patch_test.iter" );
      ::vt::theMsg()->pushEpoch( ep2 );
      if ( ::vt::theContext()->getNode() == 0 )
      {
        auto msg = ::vt::makeMessage< initiate_migrate_msg >();
        collection.broadcastMsg< initiate_migrate_msg, migrate_narrowphase_patch >( msg );
      }
      ::vt::theMsg()->popEpoch();
      ::vt::theTerm()->finishedEpoch( ep2 );
      ::vt::runSchedulerThrough( ep2 );
    }
    ::vt::theMsg()->popEpoch();
    ::vt::theTerm()->finishedEpoch( ep );
    ::vt::runSchedulerThrough( ep );
  }
}

#if 0
TEMPLATE_TEST_CASE("multiple values can be serialized", "[serializer]", int, double, float, std::size_t )
{
  bvh::serializer s;
  using T = TestType;

  T a, b, c;

  s << T{ 17 } << static_cast< T >( 21.3 ) << static_cast< T >( 3.9 );
  s >> a >> b >> c;

  REQUIRE( a == T{ 17 } );
  REQUIRE( b == static_cast< T >( 21.3 ) );
  REQUIRE( c == static_cast< T >( 3.9 ) );
}

TEST_CASE("multiple differently-typed values can be serialized", "[serializer]")
{
  bvh::serializer s;
  s << 5 << 3.7 << 1.1f;

  int a;
  double b;
  float c;

  s >> a >> b >> c;

  REQUIRE( a == 5 );
  REQUIRE( b == 3.7 );
  REQUIRE( c == 1.1f );
}

TEST_CASE("strings can be serialized", "[serializer]")
{
  using namespace std::string_literals;

  bvh::serializer s;

  s << "hello, "s << "world"s;

  std::string a, b;

  s >> a >> b;

  REQUIRE( a == "hello, "s );
  REQUIRE( b == "world"s );
}


TEST_CASE("char array literals can be serialized and deserialized as strings", "[serializer]")
{
  using namespace std::string_literals;

  bvh::serializer s;

  s << "hello, " << "world";

  std::string a, b;

  s >> a >> b;

  REQUIRE( a == "hello, \0"s );
  REQUIRE( b == "world\0"s );
}


TEST_CASE("a simple vector can be serialized", "[serializer]")
{
  using namespace std::string_literals;

  bvh::serializer s;

  auto v = std::vector< int >{ 1, 2, 3 };

  s << v;

  std::vector< int > r;

  s >> r;

  REQUIRE( r.size() == v.size() );

  for ( std::size_t i = 0; i < r.size(); ++i )
    REQUIRE( r[i] == v[i] );
}

struct NonTrivial
{
  NonTrivial() = default;
  NonTrivial( int _m ) : m( _m ) {}

  int m;

  virtual ~NonTrivial() {}
};

template< typename Serializer >
bvh::serializer_interface< Serializer > &
operator<<( bvh::serializer_interface< Serializer > &_serializer, const NonTrivial &_nt )
{
  _serializer << _nt.m;

  return _serializer;
}

template< typename Serializer >
bvh::serializer_interface< Serializer > &
operator>>( bvh::serializer_interface< Serializer > &_serializer, NonTrivial &_nt )
{
  _serializer >> _nt.m;

  return _serializer;
}

TEST_CASE("vectors of custom serialized types can be serialized", "[serializer]")
{
  using namespace std::string_literals;

  bvh::serializer s;

  auto v = std::vector< NonTrivial >{ 1, 2, 3 };

  s << v;

  std::vector< NonTrivial > r;

  s >> r;

  REQUIRE( r.size() == v.size() );

  for ( std::size_t i = 0; i < r.size(); ++i )
    REQUIRE( r[i].m == v[i].m );
}

TEST_CASE("a nontrivial vector that wraps an int can be identically deserialized as a vector of ints", "[serializer]")
{
  using namespace std::string_literals;

  bvh::serializer s;

  auto v = std::vector< NonTrivial >{ 1, 2, 3 };

  s << v;

  std::vector< int > r;

  s >> r;

  REQUIRE( r.size() == v.size() );

  for ( std::size_t i = 0; i < r.size(); ++i )
    REQUIRE( r[i] == v[i].m );
}

TEST_CASE("a vector of ints can be identically deserialized as nontrivial wrapper type", "[serializer]")
{
  using namespace std::string_literals;

  bvh::serializer s;

  auto v = std::vector< int >{ 1, 2, 3 };

  s << v;

  std::vector< NonTrivial > r;

  s >> r;

  REQUIRE( r.size() == v.size() );

  for ( std::size_t i = 0; i < r.size(); ++i )
    REQUIRE( r[i].m == v[i] );
}

TEST_CASE("a dop26d can be serialized", "[serializer]")
{
  auto kdop = bvh::dop_26d::from_sphere( 0.0, 0.0, 0.0, 1.0 );

  bvh::serializer s;

  s << kdop;

  bvh::dop_26d kdop2;

  s >> kdop2;

  REQUIRE( kdop == kdop2 );
}

TEST_CASE("an empty tree can be serialized", "[serializer]")
{
  auto tree = bvh::snapshot_tree_26d< Element >{};

  bvh::serializer s;

  s << tree;

  auto tree2 = bvh::snapshot_tree_26d< Element >{};

  s >> tree2;

  REQUIRE( tree2.debug_validate() );

  REQUIRE( tree2.count() == 0 );
  REQUIRE( tree.count() == tree2.count() );

  REQUIRE( tree == tree2 );
}

TEST_CASE("a tree can be serialized", "[serializer]")
{
  auto elements = buildElementGrid( 2, 2, 2 );
  auto tree = bvh::build_snapshot_tree_top_down(elements );

  bvh::serializer s;

  s << tree;

  auto tree2 = bvh::snapshot_tree_26d< Element >{};

  s >> tree2;

  REQUIRE_NOTHROW( tree2.debug_validate() );

  REQUIRE( tree2.count() == elements.size() );
  REQUIRE( tree.count() == tree2.count() );

  REQUIRE( tree == tree2 );
}

TEST_CASE("a tree with a single element can be serialized", "[serializer]")
{
  auto elements = buildElementGrid( 1, 1, 1 );
  auto tree = bvh::build_snapshot_tree_top_down(elements );

  bvh::serializer s;

  s << tree;

  auto tree2 = bvh::snapshot_tree_26d< Element >{};

  s >> tree2;

  REQUIRE_NOTHROW( tree2.debug_validate() );

  REQUIRE( tree2.count() == elements.size() );
  REQUIRE( tree.count() == tree2.count() );

  REQUIRE( tree == tree2 );
}
#endif
