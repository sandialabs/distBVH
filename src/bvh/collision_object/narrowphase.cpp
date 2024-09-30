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
#include "narrowphase.hpp"
#include "../collision_object.hpp"
#include "impl.hpp"

#include <vt/trace/trace_user.h>

namespace bvh
{
  namespace collision_object_impl
  {
    pending_send activate_narrowphase( [[maybe_unused]] vt_index _local_idx, collision_object_proxy_type _this_obj )
    {
      auto msg = ::vt::makeMessage< start_activate_narrowphase_msg >();
      return _this_obj[::vt::theContext()->getNode()].sendMsg< start_activate_narrowphase_msg, &collision_object_impl::collision_object_holder::activate_narrowphase >( msg );
    }

    pending_send clear_narrowphase( [[maybe_unused]] vt_index _local_idx, collision_object_proxy_type _this_obj )
    {
      auto msg = ::vt::makeMessage< clear_narrowphase_msg >();
      return _this_obj[::vt::theContext()->getNode()].sendMsg< clear_narrowphase_msg, &collision_object_impl::collision_object_holder::clear_narrowphase >( msg );
    }

    pending_send narrowphase( [[maybe_unused]] vt_index _local_idx, collision_object_proxy_type _this_obj )
    {
      auto msg = ::vt::makeMessage< start_narrowphase_msg >();
      return _this_obj[::vt::theContext()->getNode()].sendMsg< start_narrowphase_msg, &collision_object_impl::collision_object_holder::start_narrowphase >( msg );
    }

    namespace details
    {
      void
      copy_narrowphase_patch( collision_object_impl::narrowphase_patch_collection_type *_patch,
                              narrowphase_patch_msg *_msg )
      {
        _patch->ghost_destinations.clear();
        _patch->patch_meta = _msg->patch_meta;
        Kokkos::resize( Kokkos::WithoutInitializing, _patch->bytes, _msg->data_size );
        std::memcpy( _patch->bytes.data(), _msg->user_data(), _msg->data_size );
        _patch->origin_node = _msg->origin_node;
      }
    } // namespace details

    pending_send narrowphase_patch_copy( vt_index _global_idx, vt_index _local_idx,
                        int _rank,
                        collision_object_proxy_type _obj,
                        narrowphase_patch_collection_type::CollectionProxyType _patches )
    {
      auto &this_impl = _obj.get()->self->get_impl();
      auto &this_msg = this_impl.narrowphase_patch_messages[_local_idx.x()];
      if ((this_msg == nullptr) || (this_msg->data_size == 0)) {
        return pending_send{ nullptr };
      }
      this_msg->origin_node = _rank;
      return _patches[_global_idx].sendMsg< narrowphase_patch_msg, &details::copy_narrowphase_patch >( this_msg );
    }

    pending_send request_ghosts( [[maybe_unused]] vt_index _local_idx, collision_object_proxy_type _this_obj,
                                 collision_object_proxy_type _other_obj )
    {
      auto &this_obj = _this_obj.get()->self;
      auto &other_obj = _other_obj.get()->self;
      auto &logger = this_obj->narrowphase_logger();
      auto msg = ::vt::makeMessage< start_ghosting_msg >();
      msg->this_obj = this_obj->get_impl().objgroup;
      msg->other_obj = other_obj->get_impl().objgroup;
      logger.debug( "<send=local> requesting ghosts for objects {} and {} in potential collision", this_obj->id(), other_obj->id() );
      return _this_obj[::vt::theContext()->getNode()].sendMsg< start_ghosting_msg, &collision_object_impl::collision_object_holder::request_ghosts >( msg );
    }

    namespace detail
    {
      struct send_ghost_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_patch_collection_type >
      {
        collision_object_proxy_type obj;
      };

      void
      send_ghost( collision_object_impl::narrowphase_patch_collection_type *_patch, send_ghost_msg *_msg )
      {
        const auto &obj = *_patch->collision_object.get()->self;
        auto &logger = obj.narrowphase_logger();
        logger.debug( "obj={} index {} has {} destinations", obj.id(), _patch->getIndex(), _patch->ghost_destinations.size() );
        for ( auto &&d : _patch->ghost_destinations )
        {
          logger.debug( "<send={}> obj={} sending ghost for idx {}", d, obj.id(), _patch->getIndex() );
          auto msg = ::vt::makeMessage< ghost_msg >();
          msg->meta = _patch->patch_meta;
          msg->patch_data = _patch->bytes;
          msg->origin_node = _patch->origin_node;
          msg->idx = _patch->getIndex();

          _msg->obj[d].sendMsg< ghost_msg, &collision_object_impl::collision_object_holder::cache_patch >( msg );
        }
      }
    } // namespace detail

    pending_send ghost( vt_index _global_idx,
                        collision_object_proxy_type _obj,
                        narrowphase_patch_collection_type::CollectionProxyType _patches )
    {
      auto msg = ::vt::makeMessage< detail::send_ghost_msg >();
      msg->obj = _obj;
      return _patches[_global_idx].sendMsg< detail::send_ghost_msg, &detail::send_ghost >( msg );
    }

  }
}
