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
#ifndef INC_BVH_TREE_BUILD_HPP
#define INC_BVH_TREE_BUILD_HPP

#include <vector>
#include "tree.hpp"
#include "util/span.hpp"
#include "patch.hpp"
#include "split/mean.hpp"
#include "bvh_build.hpp"
#include "traits.hpp"

namespace bvh
{
  /**
   * Rebuild the tree using a specified tree-building policy with the given elements. The type stored in the tree,
   * `T`, must be constructable from `Element`
   * \tparam TreeBuildPolicy    the policy to use for building a tree
   * \tparam Element            the element-type conforming to `ContactEntity`
   * \param _tree               the tree to rebuild
   * \param _elements           the span of elements to rebuild from
   */
  template< typename TreeBuildPolicy, typename T, typename KDop, typename NodeData, typename Element >
  void rebuild_tree( bvh_tree< T, KDop, NodeData > &_tree, span< const Element > _elements )
  {
    auto &leafs = _tree.m_leafs;
    auto &nodes = _tree.m_nodes;
    
    leafs.clear();
    nodes.clear();
    
    if ( !_elements.empty() )
    {
      // Store indices to collision objects
      leafs.assign( _elements.begin(), _elements.end() );
    
      auto b = typename TreeBuildPolicy::template builder< T, KDop, NodeData >( span< const T >( leafs.data(), leafs.size() ), nodes );
      b.build( 0, leafs.data(), 0, leafs.size(), 1 );
    }
  }

  /**
   * Build a new tree with the given span of elements using the specified tree-building policy.
   *
   * \tparam TreeBuildPolicy    the policy to use to build the tree
   * \tparam Tree               the type of the tree
   * \tparam Element            the type of element in the span
   * \param _elements           the span of elements
   * \return                    a new tree where each element is constructed from an element in the span
   */
  template< typename TreeBuildPolicy, typename Tree, typename Element >
  auto build_tree( span< const Element > _elements )
  {
    Tree ret;
    rebuild_tree< TreeBuildPolicy >( ret, _elements );
    
    return ret;
  }

  /**
   * Build a tree with the specified span of elements using a top-down method with the given splitting heuristic.
   * \f$O\left(n\right)\f$ time.
   *
   * \see \ref bvh::top_down_builder
   *
   * \tparam Tree               the type of the tree
   * \tparam SplittingMethod    the splitting heuristic to use -- defaults to \ref bvh::split::mean
   * \tparam Element            the type of element to construct the tree from
   * \param _elements           the span of elements
   * \return                    a new tree constructed with the top-down building algorithm
   */
  template< typename Tree, typename SplittingMethod = split::mean, typename Element >
  auto build_tree_top_down( span< const Element > _elements )
  {
    return build_tree< top_down_builder< SplittingMethod >, Tree >( _elements );
  }

  /**
   * Build a tree with the specified span of elements using a bottom-up method. This is the shared-memory implementation
   * of bottom-up tree building and will be done serially. \f$O\left(n\right)\f$ time.
   *
   * \tparam Tree           the type of tree
   * \tparam Element        the type of element to construct the tree from
   * \param _elements       the span of elements
   * \return                a new tree constructed with the bottom-up building algorithm
   */
  template< typename Tree, typename Element >
  auto build_tree_bottom_up_serial( span< const Element > _elements )
  {
    return build_tree< bottom_up_serial_builder, Tree >( _elements );
  }

  /**
   * \overload
   *
   * \tparam Container  the container to use in construction
   * \param _elements   the container of elements
   */
  template< typename Tree, typename SplittingMethod = split::mean, typename Container >
  auto build_tree_top_down( const Container &_elements )
  {
    return build_tree_top_down< Tree, SplittingMethod >( span< const typename Container::value_type >( _elements ) );
  }

  /**
   * \overload
   *
   * \tparam Container  the container to use in construction
   * \param _elements   the container of elements
   */
  template< typename Tree, typename Container >
  auto build_tree_bottom_up_serial( const Container &_elements )
  {
    return build_tree_bottom_up_serial< Tree >( span< const typename Container::value_type >( _elements ) );
  }

  using snapshot_tree = bvh_tree< entity_snapshot, bphase_kdop, void >;

  /**
   * Construct a snapshot tree using the top-down algorithm with mean splitting.
   *
   * \see \ref bvh::entity_snapshot
   *
   * \tparam Element    the type of element to create snapshots of
   * \param _elements   the span of elements to use for tree building
   * \return            a snapshot tree constructed in a top-down manner
   */
  template< typename Element >
  auto build_snapshot_tree_top_down( span< Element > _elements )
  {
    std::vector< entity_snapshot > snaps;
    snaps.reserve( _elements.size() );
    for ( std::size_t i = 0; i < _elements.size(); ++i )
      snaps.emplace_back( make_snapshot( _elements[i], i ) );

    return build_tree_top_down< snapshot_tree >( snaps );
  }

  /**
   * \overload
   *
   * \tparam Container  the container to use in construction
   * \param _elements   the container of elements
   */
  template< typename Container >
  auto build_snapshot_tree_top_down( const Container &_elements )
  {
    std::vector< entity_snapshot > snaps;
    snaps.reserve( _elements.size() );
    for ( std::size_t i = 0; i < _elements.size(); ++i )
      snaps.emplace_back( make_snapshot( _elements[i], i ) );

    return build_tree_top_down< snapshot_tree >( snaps );
  }

  /**
   * Construct a snapshot tree using the serial bottom-up build algorithm
   *
   * \see \ref bvh::entity_snapshot
   *
   * \tparam Element    the type of element to create snapshots of
   * \param _elements   the span of elements to use for tree building
   * \return            a snapshot tree constructed in a bottom-up manner
   */
  template< typename Container >
  auto build_snapshot_tree_bottom_up_serial( const Container &_elements )
  {
    std::vector< entity_snapshot > snaps;
    snaps.reserve( _elements.size() );
    for ( std::size_t i = 0; i < _elements.size(); ++i )
      snaps.emplace_back( make_snapshot( _elements[i], i ) );

    return build_tree_bottom_up_serial< snapshot_tree >( snaps );
  }
}

# endif // INC_BVH_TREE_BUILD_HPP
