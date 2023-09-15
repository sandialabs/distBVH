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
#include "top_down.hpp"
#include "../vt/helpers.hpp"
#include "../tree_build.hpp"
#include "impl.hpp"

namespace bvh
{
  namespace collision_object_impl
  {
    namespace
    {
      struct tree_build_broadcast_msg : ::vt::CollectionMessage< broadphase_patch_collection_type >
      {
        collision_object_proxy_type coll_obj;
      };

      using reduce_vec = vt::reducable_vector< entity_snapshot >;

      struct tree_reduce_msg : ::vt::collective::ReduceTMsg< reduce_vec >
      {
        vt_msg_serialize_required();

        tree_reduce_msg() = default;
        tree_reduce_msg( entity_snapshot _snap, collision_object_proxy_type _coll_obj )
          : coll_obj( _coll_obj )
        {
          this->getVal() = reduce_vec{ _snap };
        }

        template< typename Serializer >
        void serialize( Serializer &_s )
        {
          ::vt::collective::ReduceTMsg< reduce_vec >::serialize( _s );
          _s | coll_obj;
        }

        collision_object_proxy_type coll_obj;
      };


      void set_broadphase_trees( collision_object *_coll_obj, broadphase_tree_msg *_msg )
      {
        _coll_obj->get_impl().tree = _msg->tree;
      }

      struct tree_build_reduce
      {
        void operator()( tree_reduce_msg *_msg )
        {
          // Build the tree
          auto msg = ::vt::makeMessage< broadphase_tree_msg >();
          msg->tree = build_tree_top_down< tree_type >( _msg->getConstVal().vec );

          // Broadcast to every element of the collision object objgroup
          _msg->coll_obj.broadcastMsg< broadphase_tree_msg, &collision_object_holder::delegate< broadphase_tree_msg, &set_broadphase_trees > >( msg );
        }
      };

      void tree_build_broadcast( broadphase_patch_collection_type *_patch, tree_build_broadcast_msg *_msg )
      {
        auto msg = ::vt::makeMessage< tree_reduce_msg >( make_snapshot( _patch->patch, static_cast< std::size_t >( _patch->getIndex().x() ) ),
                                                            _msg->coll_obj );

        using ::vt::collective::reduce::makeStamp;
        using ::vt::collective::reduce::StrongUserID;
        auto stamp = makeStamp<StrongUserID>(static_cast<uint64_t>(::vt::thePhase()->getCurrentPhase()));

        _patch->getCollectionProxy().reduce< reduce_vec::opt, tree_build_reduce, tree_reduce_msg >( msg.get(), stamp );
      }
    }

    pending_send
    build_trees_top_down( vt_index _idx, collision_object_proxy_type _col_obj, broadphase_patch_collection_proxy _patches )
    {
      auto msg = ::vt::makeMessage< tree_build_broadcast_msg >();
      msg->coll_obj = _col_obj;

      return _patches[_idx].sendMsg< tree_build_broadcast_msg, &tree_build_broadcast >( msg );
    }
  }
}
