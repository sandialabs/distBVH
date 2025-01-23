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
#ifndef INC_BVH_PARALLEL_BUILD_HPP
#define INC_BVH_PARALLEL_BUILD_HPP

#include "../tree.hpp"
#include "../serialization/bvh_serialize.hpp"
#include "../patch.hpp"
#include <vt/configs/types/types_type.h>
#include <vt/transport.h>
#include <array>
#include <unordered_set>
#include "../collision_world.hpp"

namespace bvh
{
  class collision_object;

  namespace collision_object_impl
  {
    using kdop_type = bphase_kdop;
    using centroid_type = m::vec3d;
    using broadphase_patch_type = patch<>;
    using tree_type = snapshot_tree;
    using vt_index = ::vt::index::Index1D< std::size_t >;
    using pending_send = ::vt::messaging::PendingSend;

    struct broadphase_patch_collection_type : ::vt::Collection< broadphase_patch_collection_type, vt_index >
    {
      using MessageParentType = ::vt::Collection< broadphase_patch_collection_type, vt_index >;
      vt_msg_serialize_required();

      broadphase_patch_type patch;
      ::vt::NodeType origin_node = {};
      vt_index local_idx = {};

      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );
        _s | patch | origin_node | local_idx;
      }
    };

    using broadphase_patch_collection_proxy = broadphase_patch_collection_type::CollectionProxyType;

    struct broadphase_patch_msg : ::vt::CollectionMessage< broadphase_patch_collection_type >
    {
      using MessageParentType = ::vt::CollectionMessage< broadphase_patch_collection_type >;
      vt_msg_serialize_required();

      broadphase_patch_type patch;
      ::vt::NodeType origin_node;
      vt_index local_idx;

      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );
        _s | patch | origin_node | local_idx;
      }
    };

    struct broadphase_tree_msg : ::vt::Message
    {
      using MessageParentType = ::vt::Message;
      vt_msg_serialize_required();

      tree_type tree;

      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );
        _s | tree;
      }
    };

    struct result_msg : ::vt::Message
    {
      using MessageParentType = ::vt::Message;
      vt_msg_serialize_required();

      result_msg() = default;

      narrowphase_result result;

      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );

        if ( _s.isUnpacking() )
        {
          std::size_t stride;
          // FIXME_CUDA: replace vector with a View (?)
          std::vector< std::byte > bbuffer;

          _s | stride | bbuffer;

          result = narrowphase_result( stride );
          result.set_data( bbuffer.data(), bbuffer.size() / stride );
        } else {
          auto stride = result.stride();
          // FIXME_CUDA: replace vector with a View (?)
          std::vector< std::byte > bbuffer = result.byte_buffer();
          _s | stride | bbuffer;
        }
      }
    };

    struct start_activate_narrowphase_msg;
    struct setup_narrowphase_msg;
    struct start_ghosting_msg;
    struct start_narrowphase_msg;
    struct clear_narrowphase_msg;

    namespace messages
    {
      struct modify_msg : ::vt::Message {};
    }

    struct active_narrowphase_local_index_msg;
    struct ghost_msg;

    struct collision_object_holder
    {
      collision_object *self;

      template< typename Message >
      using function_type = void( collision_object *, Message * );

      template< typename Message, function_type< Message > *Fun >
      void delegate( Message *_msg )
      {
        ( *Fun )( self, _msg );
      }

      void activate_narrowphase( start_activate_narrowphase_msg *_msg );
      void setup_narrowphase( setup_narrowphase_msg *_msg );
      void request_ghosts( start_ghosting_msg *_msg );
      void start_narrowphase( start_narrowphase_msg *_msg );
      void clear_narrowphase( clear_narrowphase_msg *_msg );

      void insert_active_narrow_local_index( active_narrowphase_local_index_msg *_msg );

      void cache_patch( ghost_msg *_msg );

      void set_result( result_msg *_msg );

      void begin_narrowphase_modification( messages::modify_msg * );
      void finish_narrowphase_modification( messages::modify_msg * );
    };

    using collision_object_proxy_type = ::vt::objgroup::ObjGroupManager::ProxyType< collision_object_holder >;

    struct narrowphase_patch_collection_type : ::vt::Collection< narrowphase_patch_collection_type, vt_index >
    {
      using MessageParentType = ::vt::Collection< narrowphase_patch_collection_type, vt_index >;
      vt_msg_serialize_required();

      narrowphase_patch_collection_type() = default;
      explicit narrowphase_patch_collection_type( collision_object_proxy_type _coll_obj )
        : collision_object( _coll_obj )
      {}

      patch<> patch_meta;
      view< std::byte * > bytes;
      ::vt::NodeType origin_node = ::vt::uninitialized_destination;
      std::unordered_set< ::vt::NodeType > ghost_destinations;
      collision_object_proxy_type collision_object;

      template< typename Serializer > void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );
        _s | patch_meta | bytes | origin_node | ghost_destinations | collision_object;
      }
    };

    // Byte serializable
    struct narrowphase_patch_msg : ::vt::CollectionMessage< narrowphase_patch_collection_type >
    {
      patch<> patch_meta;
      ::vt::NodeType origin_node = ::vt::uninitialized_destination;
      std::size_t data_size = 0;

      // Used with makeMessageSz, invalid otherwise!
      auto
      user_data()
      {
        return Kokkos::View< std::byte *, Kokkos::LayoutLeft, bvh::host_execution_space, Kokkos::MemoryTraits< Kokkos::Unmanaged > >(
          reinterpret_cast< std::byte * >( this ) + sizeof( narrowphase_patch_msg ),
          data_size
        );
      }

      Kokkos::View< const std::byte *, Kokkos::LayoutLeft, bvh::host_execution_space, Kokkos::MemoryTraits< Kokkos::Unmanaged > >
      user_data() const
      {
        return Kokkos::View< const std::byte *, bvh::host_execution_space, Kokkos::MemoryTraits< Kokkos::Unmanaged > >(
          reinterpret_cast< const std::byte * >( this ) + sizeof( narrowphase_patch_msg ),
          data_size
        );
      }
    };

    /**
     * 3D index -- first dimension references the patch id,
     * the second index references the object id, of the colliding obj,
     * the third index represents the colliding patch id
     */
    using narrowphase_index = ::vt::DenseIndex< int, 3 >;
    struct narrowphase_collection_type : ::vt::InsertableCollection< narrowphase_collection_type, narrowphase_index >
    {
      using MessageParentType = ::vt::InsertableCollection< narrowphase_collection_type, narrowphase_index >;
      vt_msg_serialize_required();

      narrowphase_collection_type()
          : active( false )
      {
      }

      bool active = false;
      collision_object_proxy_type this_proxy;
      collision_object_proxy_type other_proxy;

      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );
        _s | active | this_proxy | other_proxy;
      }
    };

    struct start_activate_narrowphase_msg : ::vt::Message
    { };

    struct setup_narrowphase_msg : ::vt::Message
    { };

    struct activate_narrowphase_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >
    {
      collision_object_proxy_type this_obj;
    };

    struct start_ghosting_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >
    {
      collision_object_proxy_type this_obj;
      collision_object_proxy_type other_obj;
    };

    struct start_narrowphase_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >
    { };

    struct clear_narrowphase_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >
    { };

    struct ghost_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >
    {
      using MessageParentType = ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >;
      vt_msg_serialize_required();

      patch<> meta;
      view< std::byte * > patch_data;
      ::vt::NodeType origin_node;
      vt_index idx;

      template< typename Serializer >
      std::ptrdiff_t getOffset( Serializer &_s ) {
        if (_s.getBuffer() != nullptr) {
          return _s.getSpotIncrement(0) - _s.getBuffer();
        } else {
          // sizing
          return 0;
        }
      }

      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        MessageParentType::serialize( _s );
        _s | meta | patch_data | origin_node | idx;
      }
    };

    struct active_narrowphase_local_index_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_collection_type >
    {
      vt_index idx;
    };

  } // namespace collision_object_impl

} // namespace bvh

#endif  // INC_BVH_PARALLEL_BUILD_HPP
