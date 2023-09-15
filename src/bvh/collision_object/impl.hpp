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

#include <vector>
#include <optional>
#include "../collision_object.hpp"
#include "types.hpp"
#include "../collision_world/impl.hpp"
#include "../split/element_permutations.hpp"

#include <vt/transport.h>
#include <vt/messaging/collection_chain_set.h>

namespace bvh
{
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
    using ghost_table_index = collision_object_impl::narrowphase_index ;

    collision_world *world;

    /// \brief Object index for the `collision_world`
    std::size_t collision_idx;

    std::vector< broadphase_patch_type > local_patches;
    std::vector< ::vt::MsgPtr< collision_object_impl::narrowphase_patch_msg > > narrowphase_patch_messages;
    std::vector< std::size_t > local_data_indices;

    // Not a collection because we want this to always live on a per-node basis
    std::vector< narrowphase_result > local_results;

    ::vt::messaging::CollectionChainSet< vt_index > chainset;
    int overdecomposition = 1;
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

    const unsigned char *m_entity_ptr;
    std::size_t m_entity_unit_size = 0;
    element_permutations m_latest_permutations;

    struct narrowphase_patch_cache_entry
    {
      patch<> meta;
      std::vector< unsigned char > patch_data;
      ::vt::NodeType origin_node;
    };

    std::unordered_map< vt_index, narrowphase_patch_cache_entry > narrowphase_patch_cache;
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
