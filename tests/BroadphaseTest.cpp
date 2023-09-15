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
#include <bvh/narrowphase.hpp>
/*
TEST_CASE("broadphase works serially", "[broadphase]")
{
  auto obj = buildElementGrid( 1, 1, 1 );
  
  auto tempobj = buildElementGrid( 2, 3, 2, 1ULL );
  
  auto partition = obj.insert( obj.end(), tempobj.begin(), tempobj.end() );
  
  
  bvh::bvh_tree_26d tree1{ obj.begin(), partition };
  bvh::bvh_tree_26d tree2{ partition, obj.end() };
  
  auto res = bvh::broadphase< bvh::serial_execution_model >( obj, 0, tree2, bvh::narrowphase_single< Element > );
  
  // obj1 should collide with all of obj2 (12 elements)
  REQUIRE( res.size() == 12 );
  
  // No reverse collision
  for ( std::size_t i = 1; i < obj.size(); ++i )
  {
    res = bvh::broadphase< bvh::serial_execution_model >( obj, i, tree1, bvh::narrowphase_single< Element > );
    REQUIRE( res.size() == 0 );
  }
}*/
