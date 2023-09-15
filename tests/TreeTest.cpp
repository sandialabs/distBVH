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

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

#include <bvh/common.hpp>
#include <bvh/tree.hpp>
#include <bvh/split/mean.hpp>
#include <bvh/tree_iterator.hpp>
#include <bvh/collision_query.hpp>
#include "TestCommon.hpp"
#include <bvh/tree_build.hpp>


TEST_CASE("a tree constructed from an empty range has a depth of 0", "[tree]")
{
  bvh::dynarray< Element > elements;
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  REQUIRE( 0 == tree.depth() ); // NOLINT
}

TEST_CASE("a tree constructed from N elements exists and has that many nodes", "[tree]")
{
  auto elements = buildElementGrid( 16, 16, 16 );
  auto tree = bvh::build_snapshot_tree_top_down( elements );
  
  REQUIRE( tree.root() != nullptr );
  
  REQUIRE( 4096ULL == tree.count() );
}

TEST_CASE("a tree constructed from overlapping elements correctly splits the elements", "[tree]")
{
  auto elements = buildElementGrid( 2, 2, 2 );
  auto elements2 = buildElementGrid( 1, 1, 1 );
  elements.insert( elements.end(), elements2.begin(), elements2.end() );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  REQUIRE( tree.root()->left()->count() <= 5 );
  REQUIRE( tree.root()->right()->count() <= 5 );
}


TEST_CASE("a preorder traversal of a full tree yields N * 2 - 1 nodes in total", "[tree]")
{
  auto elements = buildElementGrid( 4, 4, 4 );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  int i = 0;
  
  for ( BVH_MAYBE_UNUSED auto &&data : bvh::make_preorder_traverse( tree ) )
  {
    ++i;
  }
  
  REQUIRE( i == 127 );
}

TEST_CASE("a leaf traversal of a full tree yields N nodes", "[tree]")
{
  auto elements = buildElementGrid( 4, 4, 4 );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  int i = 0;
  std::size_t count = 0;
  
  for ( auto &&node : bvh::leaf_traverse( tree ) )
  {
    count += node.num_patch_elements();
    ++i;
  }
  
  REQUIRE( count == 64ULL );
  REQUIRE( count == tree.count() );
  REQUIRE( i == 64 );
}

TEST_CASE("a tree copy is identical", "[tree]")
{
  auto elements = buildElementGrid( 4, 4, 4 );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  auto tree2 = tree;
  
  REQUIRE( tree2.debug_validate() );
  
  REQUIRE( tree == tree2 );
  REQUIRE( tree.count() == tree2.count() );
}

TEST_CASE("a tree copy of a single node tree is identical", "[tree]")
{
  auto elements = buildElementGrid( 1, 1, 1 );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  auto tree2 = tree;
  
  REQUIRE( tree2.debug_validate() );
  
  REQUIRE( tree == tree2 );
  REQUIRE( tree.count() == tree2.count() );
  REQUIRE( tree.count() == 1 );
}

TEST_CASE("a tree copy of an empty tree is identical and empty", "[tree]")
{
  bvh::snapshot_tree tree{};
  
  auto tree2 = tree;
  
  REQUIRE( tree2.debug_validate() );
  
  REQUIRE( tree == tree2 );
  REQUIRE( tree.count() == tree2.count() );
  REQUIRE( tree.count() == 0 );
}


TEST_CASE("a moved tree is valid and containts the original number of elements", "[tree]")
{
  auto elements = buildElementGrid( 4, 4, 4 );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  auto tree2 = std::move( tree );
  
  REQUIRE( tree2.debug_validate() );
  
  REQUIRE( elements.size() == tree2.count() );
}

TEST_CASE("a moved tree with one element is valid and contains only one element", "[tree]")
{
  auto elements = buildElementGrid( 1, 1, 1 );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  auto tree2 = std::move( tree );
  
  REQUIRE( tree2.debug_validate() );
  
  REQUIRE( tree2.count() == 1 );
}

TEST_CASE("a moved empty tree is valid and empty", "[tree]")
{
  bvh::snapshot_tree tree{};
  
  auto tree2 = std::move( tree );
  
  REQUIRE( tree2.debug_validate() );
  
  REQUIRE( tree == tree2 );
  REQUIRE( tree.count() == tree2.count() );
  REQUIRE( tree.count() == 0 );
}

TEST_CASE("empty patches in tree build", "[tree]")
{
  auto elements = buildElementPatchGrid( 1, 1, 1, 0 );
  // Push back empty
  elements.push_back( bvh::patch<>{} );
  
  auto tree = bvh::build_snapshot_tree_top_down(elements );
  
  auto obj = buildElementPatchGrid( 1, 1, 1, 0 )[0];
  
  REQUIRE( tree.root() != nullptr );
  
  bvh::snapshot_tree::collision_query_result_type res;

  bvh::query_tree( tree, obj, [&res]( auto _p, auto _q ){
    res.pairs.emplace_back( _p, _q );
  } );
  
  REQUIRE( res.pairs.size() == 1 );
}


TEST_CASE("bottom up build", "[tree]")
{
  bvh::dynarray< Element > elements;
  auto tree = bvh::build_snapshot_tree_bottom_up_serial( elements );
  
  
  
  REQUIRE( 0 == tree.depth() ); // NOLINT
}

TEST_CASE("bottom up single build", "[tree]")
{
  bvh::dynarray< Element > elements = buildElementGrid( 2, 1, 1 );
  auto tree = bvh::build_snapshot_tree_bottom_up_serial( elements );
}

TEST_CASE("bottom up multi build", "[tree]")
{
  bvh::dynarray< Element > elements = buildElementGrid( 4, 4, 4 );
  auto tree = bvh::build_snapshot_tree_bottom_up_serial( elements );
  
  bvh::dump_tree( std::cout, tree );
}
