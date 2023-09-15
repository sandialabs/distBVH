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

#include "TestCommon.hpp"

#include <bvh/common.hpp>
#include <bvh/tree.hpp>
#include <bvh/collision_query.hpp>
#include <bvh/tree_build.hpp>


TEST_CASE("two fully overlapping element groups all collide", "[collision]")
{
  auto elements = buildElementGrid( 4, 4, 4 );
  auto elements2 = buildElementGrid( 1, 1, 1, elements.size() );
  
  auto tree = bvh::build_snapshot_tree_top_down< Element >( elements );
  auto tree2 = bvh::build_snapshot_tree_top_down< Element >( elements2 );
  
  auto pc = bvh::potential_collision_set( tree, tree2 );
  
  // All elements in tree 1 should collide with tree 2
  REQUIRE( pc.size() == elements.size() ); // NOLINT
}


TEST_CASE("a non-overlapping element group does not self-collide", "[collision]")
{
  auto elements = buildElementGrid( 2, 2, 1 );
  
  auto tree = bvh::build_snapshot_tree_top_down< Element >( elements );
  
  auto pc = bvh::self_collision_set( tree );
  
  REQUIRE( pc.size() == 0ULL ); // NOLINT
}


TEST_CASE("an overlapping element group self-collides", "[collision]")
{
  auto elements = buildElementGrid( 2, 2, 2 );
  auto elements2 = buildElementGrid( 1, 1, 1, elements.size() );
  elements.insert( elements.end(), elements2.begin(), elements2.end() );
  
  auto tree = bvh::build_snapshot_tree_top_down< Element >( elements );
  
  auto pc = bvh::self_collision_set( tree );
  
  REQUIRE( pc.size() == 8ULL ); // NOLINT
}
