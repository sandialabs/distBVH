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
#include "broadphase.hpp"
#include "../collision_object.hpp"
#include "impl.hpp"
#include "../collision_query.hpp"
#include "../vt/print.hpp"
#include "../collision_world/impl.hpp"

namespace bvh
{
  namespace collision_object_impl
  {
    namespace
    {
      struct flag_active_narrowpatch_msg : ::vt::CollectionMessage< broadphase_patch_collection_type >
      {
        collision_object_proxy_type patch_obj;
      };

      void flag_active_narrowpatch( broadphase_patch_collection_type *_patch, flag_active_narrowpatch_msg *_msg )
      {
        auto &patch_obj = _msg->patch_obj.get()->self;
        auto activate_narrowphase_index_msg = ::vt::makeMessage< active_narrowphase_local_index_msg  >();
        activate_narrowphase_index_msg->idx = _patch->local_idx;
        patch_obj->get_impl().objgroup[_patch->origin_node].sendMsg<active_narrowphase_local_index_msg, &collision_object_impl::collision_object_holder::insert_active_narrow_local_index >( activate_narrowphase_index_msg );
      }

      struct start_broadphase_msg : ::vt::CollectionMessage< broadphase_patch_collection_type >
      {
        collision_object_proxy_type patch_obj;
        collision_object_proxy_type tree_obj;
        vt_index patch_index;
        vt_index local_idx;
        ::vt::NodeType origin_node;
      };

      void start_broadphase( broadphase_patch_collection_type *_patch, start_broadphase_msg *_msg )
      {
        auto &patch = _patch->patch;
        auto &patch_obj = _msg->patch_obj.get()->self;
        auto &tok = *patch_obj->get_impl().narrowphase_modification_token;
        collision_object_impl::narrowphase_index tmp_idx( 0, static_cast< int >( patch_obj->get_impl().collision_idx ), 0 );
        patch_obj->get_impl().narrowphase_collection_proxy[tmp_idx].insert( tok );

        debug_assert( patch.global_id() != static_cast< broadphase_patch_type::index_type >( -1 ), "patch wasn't initialized" );

        //--- Quick exit for empty patch
        if (patch.size() == 0)
          return;
        //
        auto &tree_obj = _msg->tree_obj.get()->self;
        auto &tree = tree_obj->get_impl().tree;
        int origin_node = _msg->origin_node;
        auto local_idx = _msg->local_idx;
        //
        tmp_idx[1] = static_cast<int>( tree_obj->get_impl().collision_idx );
        //
        query_tree( tree, patch, [&_msg, local_idx, origin_node, &patch_obj, &tree_obj, &tok]( std::size_t _p, std::size_t _q ){
          collision_object_impl::narrowphase_index idx( static_cast< int >( _p ),
                                                        static_cast<int>( tree_obj->get_impl().collision_idx ),
                                                        static_cast< int >( _q ) );
          patch_obj->get_impl().narrowphase_collection_proxy[idx].insert( tok );
          patch_obj->get_impl().active_narrowphase_indices.emplace_back( idx );
          //
          auto activate_narrowphase_index_msg = ::vt::makeMessage< active_narrowphase_local_index_msg  >();
          activate_narrowphase_index_msg->idx = local_idx;
          patch_obj->get_impl().objgroup[origin_node].sendMsg< active_narrowphase_local_index_msg, &collision_object_impl::collision_object_holder::insert_active_narrow_local_index >( activate_narrowphase_index_msg );
          //
          // Note that the global index `_q` may not be managed by VT on this rank
          // So we need to send a message to the rank that VT is using to manage `_q`.
          //
          auto tree_msg = ::vt::makeMessage< flag_active_narrowpatch_msg  >();
          tree_msg->patch_obj = _msg->tree_obj;
          tree_obj->get_impl().broadphase_patch_collection_proxy[_q].sendMsg< flag_active_narrowpatch_msg, &flag_active_narrowpatch >( tree_msg );
        } );
      }
    }

    pending_send broadphase( vt_index _global_idx, vt_index _local_idx, int _origin_node,
                             broadphase_patch_collection_proxy _this_patches,
                             collision_object_proxy_type _this_obj,
                             collision_object_proxy_type _other_obj )
    {
      auto msg = ::vt::makeMessage< start_broadphase_msg >();
      msg->patch_obj = _this_obj;
      msg->tree_obj = _other_obj;
      msg->patch_index = _global_idx;
      msg->local_idx = _local_idx;
      msg->origin_node = _origin_node;
      return _this_patches[_global_idx].sendMsg< start_broadphase_msg, &start_broadphase >( msg );
    }
  }
}
