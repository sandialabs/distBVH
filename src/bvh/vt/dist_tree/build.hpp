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
#ifndef INC_BVH_VT_DIST_TREE_HPP
#define INC_BVH_VT_DIST_TREE_HPP

#include "../../util/span.hpp"
#include "../../hash.hpp"
#include "../context.hpp"
#include "../objgroup.hpp"
#include <vector>
#include "../../debug/assert.hpp"
#include "../../util/bits.hpp"
#include "../collection.hpp"
#include "../../bvh_build.hpp"
#include "../../util/optional.hpp"
#include "../tree_build.hpp"

namespace bvh
{
  namespace vt
  {
    namespace detail
    {
      ::vt::Index1D
      index_mapping( std::uint64_t _m64, std::size_t _num_indices )
      {
        // Morton hashes are 63 bit numbers
        // Max possible morton code is 2^63 - 1
        const std::size_t denom = ( 0x1UL << ( 63 - bit_log2( _num_indices ) ) );
        const std::size_t index = _m64 / denom;
        return ::vt::Index1D( static_cast< int >( index ) );
      }
      
      template< typename Patch >
      struct dist_tree_builder
      {
        using snapshot_type = entity_snapshot;
        using kdop_type = typename element_traits< Patch >::kdop_type;
        using builder_type = bottom_up_serial_builder::builder< snapshot_type, kdop_type, void >;
        using treelet_type = typename builder_type::treelet_type;

        kdop_type bounds;
        dynarray< treelet_type > treelets;
        std::size_t build_height = 0;

        std::vector< optional< treelet_type > > pending_merges;

        template< typename Serializer >
        friend void serialize( Serializer &_s, const dist_tree_builder &_tree_builder )
        {
          _s | _tree_builder.bounds | _tree_builder.treelets;
        }
      };
      
      template< typename Patch >
      void add_treelet( collection_element< dist_tree_builder< Patch >, ::vt::Index1D > _builder,
        const typename dist_tree_builder< Patch >::treelet_type &_treelet )
      {
        debug( "adding treelet hash {} to index {}\n", _treelet.m64, _builder.index() );
        _builder->treelets.emplace_back( _treelet );
      }
      
      template< typename Patch >
      void bin_patch( collection_element< Patch, ::vt::Index1D > _patch,
          collection< dist_tree_builder< Patch >, ::vt::Index1D > _builders,
          typename dist_tree_builder< Patch >::kdop_type _bounds )
      {
        using snapshot_type = entity_snapshot;
        using kdop_type = typename dist_tree_builder< Patch >::kdop_type ;
        using centroid_type = typename element_traits< Patch >::centroid_type;

        auto discretize_coord = [_bounds]( const centroid_type &_centroid ) {
          auto frac = _bounds.get_normalized_coords( _centroid );

          return static_cast< m::vec3< std::uint64_t > >( m::floor( frac * 0x1fffff ) );
        };

        auto centroid = element_traits< Patch >::get_centroid( *_patch );
        auto coord = discretize_coord( centroid );
        auto m64 = morton( coord.x(), coord.y(), coord.z() );
        debug( "converting coord {} -> {} -> morton {}\n", centroid, coord, m64 );
        
        using builder_type = bottom_up_serial_builder::builder< snapshot_type, kdop_type, void >;
        using treelet_type = typename builder_type::treelet_type;
        
        auto snapshot = make_snapshot( *_patch );
        auto treelet = treelet_type{ m64, element_traits< snapshot_type >::get_kdop( snapshot ), 0, snapshot };
        
        _builders.send( index_mapping( m64, _builders.range().x() ), BVH_FH( add_treelet< Patch > ), treelet );
      }

      template< typename Patch >
      void start_merge( collection_element< dist_tree_builder< Patch >, ::vt::Index1D > _builder, std::size_t _height );

      template< typename Patch >
      void merge_up( collection_element< dist_tree_builder< Patch >, ::vt::Index1D > _builder,
          typename dist_tree_builder< Patch >::treelet_type _treelet,
          std::size_t _height )
      {
        using builder_type = typename dist_tree_builder< Patch >::builder_type;

        _builder->pending_merges[_height] = std::move( _treelet );
        // We got a message too soon, buffer
        if ( _height > _builder->build_height )
        {
          auto &tr = *_builder->pending_merges[_height];
          debug( "{} buffering treelet {} (n: {}, e:{}) at height {}\n",
                 _builder.index().x(),
                 tr.m64, tr.nodes.size(), tr.leafs.size(),
                 _height );
          return;
        } else if ( _height < _builder->build_height ) {
          abort( "somehow got a build message less than the build height" );
        }

        auto &treelet = _builder->treelets[0];
        auto &h = _builder->build_height;

        while ( _builder->pending_merges[h] && h < _builder->pending_merges.size() )
        {
          auto &right_treelet = *_builder->pending_merges[h];
          debug( "{} merging treelet {} (n: {}, e: {}) with treelet {} (n: {}, e: {}) at height {}\n",
                 _builder.index().x(),
                 treelet.m64, treelet.nodes.size(), treelet.leafs.size(),
                 right_treelet.m64, right_treelet.nodes.size(), right_treelet.leafs.size(),
                 h );

          if ( !right_treelet.empty() )
          {
            if ( !treelet.empty() )
              right_treelet.merge_left( treelet );

            treelet = std::move( right_treelet );
          }

          start_merge< Patch >( _builder, h + 1 );

          ++h;
        }
      }

      template< typename Patch >
      void start_merge( collection_element< dist_tree_builder< Patch >, ::vt::Index1D > _builder, std::size_t _height )
      {
        using builder_type = typename dist_tree_builder< Patch >::builder_type;
        const std::uint64_t mask = static_cast< std::uint64_t >( -2 ) << _height;

        if ( ( _builder.index().x() & mask ) == _builder.index().x() )
          return;

        // Figure out merge height with left neighbor
        std::uint64_t this_m64 = static_cast< std::uint64_t >( _builder.index().x() );
        std::uint64_t target_m64 = this_m64 & mask;

        debug( "{} will merge treelet {} (n: {}, e: {}) with {} at height {} ({} treelets active)\n", this_m64,
            _builder->treelets[0].m64,
            _builder->treelets[0].nodes.size(),
            _builder->treelets[0].leafs.size(),
            target_m64, _height, _builder->treelets.size() );

        auto dest_index = ::vt::Index1D{ static_cast< int >( target_m64 ) };

        _builder.get_collection().send( dest_index, BVH_FH( merge_up< Patch > ), _builder->treelets[0], _height );
      }
      
      template< typename Patch >
      void step( collection_element< dist_tree_builder< Patch >, ::vt::Index1D > _builder, std::size_t _height )
      {
        using builder_type = typename dist_tree_builder< Patch >::builder_type;

        // Do local merge
        // TODO: deal with duplicates
        auto &treelets = _builder->treelets;

        if ( treelets.empty() )
          treelets.emplace_back();

        std::sort( treelets.begin(), treelets.end(), []( const auto &_left, const auto &_right ){ return _left.m64 < _right.m64; } );

        builder_type::merge_duplicates( treelets );

        std::size_t min_height = 0;
        for ( std::size_t i = 0; i < treelets.size() - 1; ++i )
        {
          treelets[i].next_merge_height = builder_type::merge_height( treelets[i].m64, treelets[i + 1].m64 );
          min_height = std::min( min_height, treelets[i].next_merge_height );
        }

        const std::size_t num_index_bits = bit_log2( static_cast< std::size_t >( _builder.get_collection().range().x() ) );
        const std::size_t max_height = 64 - num_index_bits;
        _builder->pending_merges.assign( num_index_bits, {} );
        debug( "distributed tree up to height {}\n", num_index_bits );

        builder_type::build_treelets( treelets, min_height, max_height );

        debug( "{} merged into {} treelets\n", _builder.index(), treelets.size() );

        if ( treelets.size() > 1 )
        {
          warn( "\033[1;93mWARNING: {} has {} treelets after merge\033[0m\n", _builder.index(), treelets.size() );
          for ( auto &&tr : treelets )
          {
            warn( "\033[1;93m{}\t{}\033[0m\n", _builder.index(), tr.m64 );
          }
        }

        start_merge< Patch >( _builder, 0 );
      }
    }
    
    namespace detail
    {
      template< typename KDop >
      struct bounds_merge
      {
        using vec_type = m::vec3< typename KDop::arithmetic_type >;

        bounds_merge() = default;

        bounds_merge( const KDop &_bounds, const vec_type &_centroid )
          : bounds( _bounds ), centroid( _centroid )
        {
        
        }
        
        bounds_merge &operator+=( const bounds_merge &_other )
        {
          //bounds = merge( bounds, _other.bounds );
          bounds.expand( _other.centroid );
          
          return *this;
        }
        
        friend bounds_merge operator+( bounds_merge _lhs, const bounds_merge &_rhs )
        {
          return _lhs += _rhs;
        }

        template< typename Serializer >
        void serialize( Serializer &_s )
        {
          _s | bounds | centroid;
        }
        
        KDop bounds;
        m::vec3< typename KDop::arithmetic_type > centroid;
      };

      template< typename Patch >
      void bounds_reduce( const collection< Patch, ::vt::Index1D > &_patches,
                          detail::bounds_merge< typename Patch::kdop_type > _merge,
                          collection< detail::dist_tree_builder< Patch >, ::vt::Index1D > _builders )
      {

        debug( "setting builder bounds to {}\n", _merge.bounds );
        _patches.broadcast( BVH_FH( bin_patch< Patch > ), _builders, _merge.bounds );
      }

      template< typename TreeType, typename Patch >
      void broadcast_trees( collection_element< dist_tree_builder< Patch >, ::vt::Index1D > _builder,
          collection< TreeType, ::vt::Index1D > _trees )
      {
        auto &tr = _builder->treelets[0];

        TreeType tree{ std::move( tr )};

        debug("{}: broadcasting tree with {} nodes ({} leafs)\n",
            ::vt::theContext()->getNode(),
            tree.nodes().size(),
            tree.leafs().size() );
        _builder->treelets.clear();

        _trees.broadcast( BVH_FH( set_trees< TreeType > ), tree );
      }
    }
    
    template< typename Patch >
    void start_bounds_reduce( const collection_element< Patch, ::vt::Index1D > &_patch,
                              collection< detail::dist_tree_builder< Patch >, ::vt::Index1D > _builders )
    {
      auto centroid = element_traits< Patch >::get_centroid( *_patch );
      auto reducer = detail::bounds_merge< typename Patch::kdop_type >( Patch::kdop_type::from_sphere( centroid, 0 ), centroid );
  
      _patch.get_collection().reduce( std::move( reducer ), BVH_FH( detail::bounds_reduce< Patch > ), _builders );
    }

    template< typename Patch >
    auto make_dist_tree()
    {
      const auto nranks = bvh::vt::context::current()->num_ranks();
      auto col_size = ::vt::Index1D{ bvh::next_pow2( nranks * 4 ) };
      return make_collection< detail::dist_tree_builder< Patch > >( col_size );
    }

    template< typename TreeType, typename Patch >
    auto build_dist_tree( collection< Patch, ::vt::Index1D > _patches,
                          collection< detail::dist_tree_builder< Patch >, ::vt::Index1D > _builders,
                          collection< TreeType, ::vt::Index1D > _trees )
    {
      auto ep = ::vt::theTerm()->makeEpochCollective();
      auto step_epoch = ::vt::theTerm()->makeEpochCollective();
      auto bcast_epoch = ::vt::theTerm()->makeEpochCollective();
      promise< collection< TreeType, ::vt::Index1D > > p;

      _patches.broadcast_epoch( ep, BVH_FH( start_bounds_reduce< Patch > ), _builders );

      ::vt::theTerm()->addAction( ep, [_builders, p, step_epoch, _trees, bcast_epoch]() mutable {
        _builders.broadcast_epoch( step_epoch, BVH_FH( detail::step< Patch > ), 0 );

        ::vt::theTerm()->addAction( step_epoch, [_builders, p, _trees, bcast_epoch]() mutable {

          if ( ::vt::theContext()->getNode() == 0 )
          {
            auto dest_index = ::vt::Index1D{ static_cast< int >( 0 ) };
            _builders.send_epoch( bcast_epoch, dest_index, BVH_FH( detail::broadcast_trees< TreeType, Patch > ), _trees );
          }

          ::vt::theTerm()->addAction( bcast_epoch, [p, _trees]() mutable {
            p.set_value( _trees );
          } );

          ::vt::theTerm()->finishedEpoch( bcast_epoch );
        } );

        ::vt::theTerm()->finishedEpoch( step_epoch );
      } );

      ::vt::theTerm()->finishedEpoch( ep );

      return p.get_future();
    }
  }
}

#endif  // INC_BVH_VT_DIST_TREE_HPP
