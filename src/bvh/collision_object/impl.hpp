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
#ifndef INC_BVH_COLLISION_OBJECT_IMPL_HPP
#define INC_BVH_COLLISION_OBJECT_IMPL_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <optional>
#include <Kokkos_Core.hpp>
#include "../collision_object.hpp"
#include "types.hpp"
#include "../collision_world/impl.hpp"
#include "../split/element_permutations.hpp"

#include <vt/transport.h>
#include <vt/messaging/collection_chain_set.h>

namespace bvh
{
  namespace collision_object_impl
  {
    inline void
    narrowphase_patch_copy( collision_object_impl::narrowphase_patch_collection_type *_patch,
                            narrowphase_patch_msg *_msg )
    {
      const auto &obj = *_patch->collision_object.get()->self;
      auto &logger = obj.narrowphase_logger();
      auto idx = _patch->getIndex();
      logger.debug( "late initializing narrowphase patch {} with {} bytes", idx.x(), _msg->data_size );
      _patch->ghost_destinations.clear();
      _patch->patch_meta = _msg->patch_meta;
      Kokkos::resize( Kokkos::WithoutInitializing, _patch->bytes, _msg->data_size );
      Kokkos::deep_copy( _patch->bytes, _msg->user_data() );
      //std::memcpy( _patch->bytes.data(), _msg->user_data(), _msg->data_size );
      _patch->origin_node = _msg->origin_node;
    }
  }

  struct collision_object::impl
  {
    explicit impl( collision_world &_world, std::size_t _idx );

    static impl &get_impl( collision_object &_obj ) { return *_obj.m_impl; }
    static const impl &get_impl( const collision_object &_obj ) { return *_obj.m_impl; }

    using broadphase_patch_type = collision_object_impl::broadphase_patch_type;
    using tree_type = collision_object_impl::tree_type;
    using vt_index = collision_object_impl::vt_index;
    using broadphase_patch_collection_type = collision_object_impl::broadphase_patch_collection_type;
    using narrowphase_patch_collection_type = collision_object_impl::narrowphase_patch_collection_type;

    using narrowphase_collection_type = collision_object_impl::narrowphase_collection_type;
    using ghost_table_index = collision_object_impl::narrowphase_index;

    /**
     * @brief Copy the local data pointed to by m_entity_ptr at the offset corresponding to the
     * permutation for the given local element index
     *
     * @param _idx the local element index
     * @param _rank the current rank, passed in to avoid an extra function call
     * @return the message containing the narrowphase data
     */
    ::vt::MsgPtr< collision_object_impl::narrowphase_patch_msg >
    prepare_local_patch_for_sending( std::size_t _local_idx, int _rank )
    {
      auto &logger = *broadphase_logger;

      using narrowphase_patch_msg = collision_object_impl::narrowphase_patch_msg;

      const auto idx = _local_idx;
      const auto sbeg = ( idx == 0 ) ? 0 : splits_h( idx - 1 );
      const auto send = ( idx == num_splits ) ? split_indices_h.extent( 0 ) : splits_h( idx );
      const std::size_t nelements = send - sbeg;
      const std::size_t chunk_data_size = nelements * m_user_data->element_size();
      const int rank = _rank;
      debug_assert( chunk_data_size > 0, "chunk_data_size size must be > 0" );

      auto send_msg = ::vt::makeMessageSz< narrowphase_patch_msg >( chunk_data_size );
      send_msg->data_size = chunk_data_size;

      logger.debug( "obj={} sending narrowphase patch {} with {} num elements",
                    collision_idx, vt_index{ _local_idx + rank * overdecomposition }, nelements );
      
      
      m_user_data->scatter_to_byte_buffer( send_msg->user_data(), sbeg, send, split_indices_h );

      send_msg->origin_node = rank;
      send_msg->patch_meta = local_patches[idx];

      return send_msg;
    }



    collision_world *world;

    /// \brief Object index for the `collision_world`
    std::size_t collision_idx;

    std::vector< broadphase_patch_type > local_patches;
    std::vector< broadphase_patch_type > last_step_local_patches;
    std::vector< ::vt::MsgPtr< collision_object_impl::narrowphase_patch_msg > > narrowphase_patch_messages;
    std::vector< std::size_t > local_data_indices;

    // Not a collection because we want this to always live on a per-node basis
    std::vector< narrowphase_result > local_results;

    ::vt::messaging::CollectionChainSet< vt_index > chainset;
    std::size_t overdecomposition = 1;
    bool build_trees = true;

    // 1D collection of patch metadata, each index in the collection corresponds to the same index in narrowphase_patch_collection_proxy
    broadphase_patch_collection_type::CollectionProxyType broadphase_patch_collection_proxy;
    // 1D collection of patch element data, each index in the collection corresponds to the same index in broadphase_patch_collection_proxy
    narrowphase_patch_collection_type::CollectionProxyType narrowphase_patch_collection_proxy = ::vt::no_vrt_proxy;

    // 3D collection of contact hits
    // -1st (nonexistant) index is the id of the collision object -- in NimbleSM this is 0 for primary and 1 for secondary
    // 0th index is the patch id (see above) of the patch in contact
    // 1st index is the id of the contacting collision object
    // 2nd index is the patch id (see above) of the patch in the contacting collision object
    narrowphase_collection_type::CollectionProxyType narrowphase_collection_proxy = ::vt::no_vrt_proxy;
    std::optional< ::vt::vrt::collection::ModifierToken > narrowphase_modification_token;

    collision_object_impl::collision_object_proxy_type objgroup;
    tree_type tree;

    std::vector< collision_object_impl::narrowphase_index > active_narrowphase_indices;
    std::unordered_set< size_t > active_narrowphase_local_index;

    std::unique_ptr< detail::user_element_storage_base > m_user_data;
    element_permutations m_latest_permutations;

    struct narrowphase_patch_cache_entry
    {
      patch<> meta;
      view< std::byte * > patch_data;
      ::vt::NodeType origin_node;
    };

    std::unordered_map< vt_index, narrowphase_patch_cache_entry > narrowphase_patch_cache;

    // Split and clustering views
    view< bvh::entity_snapshot * > snapshots;
    view< bvh::entity_snapshot * >::host_mirror_type snapshots_h;
    view< std::size_t * > split_indices;  ///< Mapping from original element indices to the reordered indices
    view< std::size_t * >::host_mirror_type split_indices_h;
    view< std::size_t * > splits; ///< bounds of each split
    view< std::size_t * >::host_mirror_type splits_h;
    std::size_t num_splits = 0; ///< The number of actual splits -- may be les than splits.extent( 0 )

    // Loggers
    std::shared_ptr< spdlog::logger > logger;
    std::shared_ptr< spdlog::logger > broadphase_logger;
    std::shared_ptr< spdlog::logger > narrowphase_logger;
  };

  namespace collision_object_impl
  {
    void activate_narrowphase( collision_object_impl::narrowphase_collection_type *_narrow, activate_narrowphase_msg *_msg );
    void start_ghosting( collision_object_impl::narrowphase_collection_type *_narrow, start_ghosting_msg *_msg );
    void start_narrowphase( collision_object_impl::narrowphase_collection_type *_narrow, start_narrowphase_msg *_msg );
    void clear_narrowphase( collision_object_impl::narrowphase_collection_type *_narrow, clear_narrowphase_msg *_msg );
  } // namespace collision_object_impl

} // namespace bvh

#endif  // INC_BVH_COLLISION_OBJECT_IMPL_HPP
