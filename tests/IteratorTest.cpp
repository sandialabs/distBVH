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

#include <bvh/iterators/zip_iterator.hpp>
#include <bvh/range.hpp>
#include <bvh/iterators/level_iterator.hpp>
#include <bvh/iterators/offset_iterator.hpp>
#include <bvh/tree.hpp>
#include "TestCommon.hpp"
#include <algorithm>
#include <bvh/tree_build.hpp>


TEST_CASE("the zip iterator can be constructed from two ranges of differing type", "[iterator]")
{
  std::vector< char > a{ 'a', 'b', 'c' };
  std::vector< int > b{ 1, 2, 3 };
  
  auto zip_begin = bvh::make_zip_iterator( a.begin(), b.begin() );
  auto zip_end = bvh::make_zip_iterator( a.end(), b.end() );
  
  std::size_t idx = 0;
  for ( auto &&t : bvh::make_range( zip_begin, zip_end ) )
  {
    REQUIRE( bvh::detail::get< 0 >( t ) == a[idx] );
    REQUIRE( bvh::detail::get< 1 >( t ) == b[idx] );
    ++idx;
  }
}

template< int N >
using node_data_array = std::array< const bvh::snapshot_tree::node_type *, N >;


TEST_CASE("level iterators iterate through each level of a tree", "[iterator]")
{
  auto elements = buildElementGrid( 2, 2, 2 );
  
  auto tree = bvh::build_snapshot_tree_top_down< Element >( elements );
  REQUIRE( tree.depth() == 3 );
  
  const auto level0 = node_data_array< 1 >{ tree.root() };
  auto level0iter = bvh::level_traverse( tree, 0 );
  REQUIRE( level0.size() == level0iter.size() );
  REQUIRE( std::equal( level0.begin(), level0.end(),
                           level0iter.begin(), level0iter.end(),
                           []( auto *_a, const auto &_b ) { return _a == &_b; }) );
  
  const auto level1 = node_data_array< 2 >{ tree.root()->left(), tree.root()->right() };
  auto level1iter = bvh::level_traverse( tree, 1 );
  REQUIRE( level1.size() == level1iter.size() );
  
  REQUIRE( std::equal( level1.begin(), level1.end(),
                           level1iter.begin(), level1iter.end(),
                           []( auto *_a, const auto &_b ) { return _a == &_b; }) );
  
  const auto level2 = node_data_array< 4 >{ tree.root()->left()->left(),
                                            tree.root()->left()->right(),
                                            tree.root()->right()->left(),
                                            tree.root()->right()->right()};
  auto level2iter = bvh::level_traverse( tree, 2 );
  REQUIRE( level2.size() == level2iter.size() );
  
  REQUIRE( std::equal( level2.begin(), level2.end(),
                           level2iter.begin(), level2iter.end(),
                           []( auto *_a, const auto &_b ) { return _a == &_b; }) );
  
  const auto level3 = node_data_array< 8 >{ tree.root()->left()->left()->left(),
                                            tree.root()->left()->left()->right(),
                                            tree.root()->left()->right()->left(),
                                            tree.root()->left()->right()->right(),
                                            tree.root()->right()->left()->left(),
                                            tree.root()->right()->left()->right(),
                                            tree.root()->right()->right()->left(),
                                            tree.root()->right()->right()->right() };
  auto level3iter = bvh::level_traverse( tree, 3 );
  REQUIRE( level3.size() == level3iter.size() );
  
  REQUIRE( std::equal( level3.begin(), level3.end(),
                           level3iter.begin(), level3iter.end(),
                           []( auto *_a, const auto &_b ) { return _a == &_b; }) );
}

#if 0
TEST_CASE("offset iterators point inside a container correctly even if the container is reallocated", "[iterator]")
{
  std::vector< int > container{ 1, 5, 8 };
  
  bvh::offset_iter< std::vector< int > > iter( container, 0 );
  
  container.reserve( 100 );
  
  REQUIRE( *iter++ == 1 );
  REQUIRE( *iter++ == 5 );
  REQUIRE( *iter++ == 8 );
  REQUIRE( iter == bvh::offset_iter< std::vector< int > >( container ) );
}
#endif
