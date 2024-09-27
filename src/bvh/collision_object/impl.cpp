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
#include "impl.hpp"
#include "narrowphase.hpp"
#include <vt/messaging/envelope/envelope_extended_util.h>

namespace bvh
{
  collision_object::impl::impl( collision_world &_world, std::size_t _idx )
    : world( &_world ),
      collision_idx( _idx ),
      snapshots( fmt::format( "contact entity {} snapshot", _idx ), 0 ),
      snapshots_h( fmt::format( "contact entity host {} snapshot", _idx ), 0 ),
      split_indices( fmt::format( "contact entity {} split indices", _idx ), 0 ),
      splits( fmt::format( "contact entity {} splits", _idx ), 0 ),
      splits_h( fmt::format( "contact entity host {} splits", _idx ), 0 ),
      logger( _world.collision_object_logger() ),
      broadphase_logger( _world.collision_object_broadphase_logger() ),
      narrowphase_logger( _world.collision_object_narrowphase_logger() )
  {}

  namespace collision_object_impl
  {

    //
    // Define member functions for the class 'collision_object_impl::collision_object_holder'
    // declared in `types.hpp`
    //

    void collision_object_holder::insert_active_narrow_local_index( active_narrowphase_local_index_msg *_msg )
    {
      self->get_impl().active_narrowphase_local_index.insert( _msg->idx.x() );
    }

    void collision_object_holder::setup_narrowphase( [[maybe_unused]] setup_narrowphase_msg *_msg )
    {
      auto &logger = self->broadphase_logger();
      auto &impl = self->get_impl();
      auto rank = static_cast< int >( ::vt::theContext()->getNode() );
      const auto od_factor = impl.overdecomposition;
      const std::size_t od_offset = rank * od_factor;
      auto &patches = impl.narrowphase_patch_collection_proxy;

      logger.debug( "obj={}, setting up {} narrowphase patches marked as ready to activate", self->id(),
                    impl.active_narrowphase_indices.size() );
      for ( const auto idx : impl.active_narrowphase_local_index )
      {
        auto send_msg = impl.prepare_local_patch_for_sending( idx, rank );
        logger.trace( "<send=idx({})> obj={} narrowphase_patch_copy", od_offset + idx, self->id() );
        patches[od_offset + idx].sendMsg< narrowphase_patch_msg, &collision_object_impl::narrowphase_patch_copy >(
          send_msg );
      }
    }

    void collision_object_holder::activate_narrowphase( [[maybe_unused]] start_activate_narrowphase_msg *_msg )
    {
      auto &logger = self->narrowphase_logger();
      auto &impl = self->get_impl();

      logger.debug( "obj={}, activating {} narrowphase patches", self->id(), impl.active_narrowphase_indices.size() );
      for ( auto &&idx : impl.active_narrowphase_indices )
      {
        auto msg = ::vt::makeMessage< activate_narrowphase_msg >();
        msg->this_obj = self->get_impl().objgroup;
        logger.trace( "<send={}> obj={} activate_narrowphase", idx, self->id() );
        impl.narrowphase_collection_proxy[idx]
          .sendMsg< activate_narrowphase_msg, &collision_object_impl::activate_narrowphase >( msg.get() );
      }
    }

    void collision_object_holder::request_ghosts( start_ghosting_msg *_msg )
    {
      auto &logger = self->narrowphase_logger();
      auto &impl = self->get_impl();
      for ( auto &&idx : impl.active_narrowphase_indices )
      {
        auto msg = ::vt::makeMessage< start_ghosting_msg >();
        msg->this_obj = _msg->this_obj;
        msg->other_obj = _msg->other_obj;
        logger.trace( "<send={}> objp={}, objq={} start_ghosting", idx, _msg->this_obj.get()->self->id(),
                      _msg->other_obj.get()->self->id() );
        impl.narrowphase_collection_proxy[idx].sendMsg< start_ghosting_msg, &collision_object_impl::start_ghosting >(
          msg.get() );
      }
    }

    void collision_object_holder::start_narrowphase( [[maybe_unused]] start_narrowphase_msg *_msg )
    {
      auto &impl = self->get_impl();
      for ( auto &&idx : impl.active_narrowphase_indices )
      {
        auto msg = ::vt::makeMessage< start_narrowphase_msg >();
        impl.narrowphase_collection_proxy[idx]
          .sendMsg< start_narrowphase_msg, &collision_object_impl::start_narrowphase >( msg.get() );
      }
    }

    void collision_object_holder::clear_narrowphase( [[maybe_unused]] clear_narrowphase_msg *_msg )
    {
      auto &impl = self->get_impl();
      for ( auto &&idx : impl.active_narrowphase_indices )
      {
        auto msg = ::vt::makeMessage< clear_narrowphase_msg >();
        impl.narrowphase_collection_proxy[idx]
          .sendMsg< clear_narrowphase_msg, &collision_object_impl::clear_narrowphase >( msg.get() );
      }
    }

    void collision_object_holder::begin_narrowphase_modification( messages::modify_msg * )
    {
      ::vt::theMsg()->pushEpoch( ::vt::term::any_epoch_sentinel );
      self->get_impl().narrowphase_modification_token
        = self->get_impl().narrowphase_collection_proxy.beginModification( "broadphase contact insertion" );
      ::vt::theMsg()->popEpoch( ::vt::term::any_epoch_sentinel );
    }

    void collision_object_holder::finish_narrowphase_modification( messages::modify_msg * )
    {
      self->get_impl().narrowphase_collection_proxy.finishModification(
        std::move( *self->get_impl().narrowphase_modification_token ) );
      self->get_impl().narrowphase_modification_token = {};
    }

    void collision_object_holder::cache_patch( ghost_msg *_msg )
    {
      auto &impl = self->get_impl();
      auto &logger = self->narrowphase_logger();
      logger.debug( "obj={} caching patch idx {}", impl.collision_idx, _msg->idx );

      auto &ent = impl.narrowphase_patch_cache[_msg->idx];

      ent.meta = _msg->meta;
      ent.origin_node = _msg->origin_node;
      ent.patch_data = _msg->patch_data;
    }

    void collision_object_holder::set_result( result_msg *_msg )
    {
      self->get_impl().local_results.emplace_back( _msg->result );
    }

    void activate_narrowphase( collision_object_impl::narrowphase_collection_type *_narrow, activate_narrowphase_msg *_msg )
    {
      const auto &this_obj = *_msg->this_obj.get()->self;
      auto &logger = this_obj.narrowphase_logger();
      auto idx = _narrow->getIndex();
      logger.trace( "marking <{}, {}, {}, {}> as active (epoch={:x})", this_obj.id(), idx[0], idx[1], idx[2], ::vt::envelopeGetEpoch( _msg->env ) );
      _narrow->active = true;
    }

    namespace detail
    {

      struct ghost_request_msg : ::vt::CollectionMessage< collision_object_impl::narrowphase_patch_collection_type >
      {
        collision_object_impl::narrowphase_collection_type::CollectionProxyType proxy;
        collision_object_impl::narrowphase_index idx;
        ::vt::NodeType dest_node;
      };

      void request_ghost( collision_object_impl::narrowphase_patch_collection_type *_patch, ghost_request_msg *_msg )
      {
        const auto &obj = _patch->collision_object.get()->self;
        auto &logger = obj->narrowphase_logger();
        // Find destination node for the narrowphase collection element
        auto dst = _msg->dest_node;
        logger.debug( "obj={} requesting ghost for index {} to node {}", obj->id(), _patch->getIndex(), dst );

        // Build up ghost_destination list/group. In ghosting step,
        // this element will be transferred to every node in ghost_destination.
        // Then it will be cached when narrowphase is executed
        _patch->ghost_destinations.emplace( dst );
      }

    }  // namespace detail

    void start_ghosting( collision_object_impl::narrowphase_collection_type *_narrow, start_ghosting_msg *_msg )
    {
      auto idx = _narrow->getIndex();
      auto &this_obj = _msg->this_obj.get()->self;
      auto &other_obj = _msg->other_obj.get()->self;
      auto &logger = this_obj->narrowphase_logger();

      // Only do anything if this is active to prev ent residual elements
      // from earlier iterations
      if ( !_narrow->active )
      {
        logger.trace( "skipping <{}, {}, {}, {}> -- not active", this_obj->id(), idx[0], idx[1], idx[2] );
        return;
      }

      // Only run if we are looking at the right "other obj"
      if ( other_obj->get_impl().collision_idx != static_cast< std::size_t >( idx.y() ) )
      {
        logger.trace( "skipping <{}, {}, {}, {}> -- mismatched index", this_obj->id(), idx[0], idx[1], idx[2] );
        return;
      }

      // Ignore self collisions (this will usually be caught by the above condition)
      if ( this_obj->get_impl().collision_idx == static_cast< std::size_t >( idx.y() ) )
      {
        logger.trace( "{}: skipping <{}, {}, {}, {}> -- self collision", this_obj->id(), idx[0], idx[1], idx[2] );
        return;
      }

      _narrow->this_proxy = _msg->this_obj;
      _narrow->other_proxy = _msg->other_obj;

      logger.debug( "start ghosting <{}, {}, {}, {}>", this_obj->id(), idx[0], idx[1], idx[2] );

      auto rank = ::vt::theContext()->getNode();
      // Send ghost request to this obj
      auto msg = ::vt::makeMessage< detail::ghost_request_msg >();
      msg->idx = idx;
      msg->proxy = _narrow->getCollectionProxy();
      // msg->ordering = 0;
      msg->dest_node = rank;
      auto this_idx = collision_object_impl::vt_index{ static_cast< std::size_t >( idx[0] ) };
      logger.trace( "<send={}> obj={} requesting primary patch {} from object {}", this_idx, this_obj->id(), this_idx.x(),
                    this_obj->id() );
      this_obj->get_impl()
        .narrowphase_patch_collection_proxy[this_idx]
        .sendMsg< detail::ghost_request_msg, &detail::request_ghost >( msg.get() );

      // Send ghost request to other obj
      auto other_msg = ::vt::makeMessage< detail::ghost_request_msg >();
      other_msg->idx = idx;
      other_msg->proxy = _narrow->getCollectionProxy();
      other_msg->dest_node = rank;
      // other_msg->ordering = 1;
      auto other_idx = collision_object_impl::vt_index{ static_cast< std::size_t >( idx[2] ) };
      logger.trace( "<send={}> obj={} requesting secondary patch {} from object {}", other_idx, this_obj->id(), other_idx.x(),
                    other_obj->id() );
      other_obj->get_impl()
        .narrowphase_patch_collection_proxy[other_idx]
        .sendMsg< detail::ghost_request_msg, &detail::request_ghost >( other_msg.get() );
    }

    void start_narrowphase( narrowphase_collection_type *_narrow, start_narrowphase_msg * )
    {
      auto idx = _narrow->getIndex();

      auto &this_obj = *_narrow->this_proxy.get()->self;
      auto &other_obj = *_narrow->other_proxy.get()->self;
      auto &this_impl = this_obj.get_impl();
      auto &other_impl = other_obj.get_impl();

      auto &logger = this_obj.narrowphase_logger();
      logger.debug( "executing narrowphase <{}, {}, {}, {}> in epoch={}", this_obj.id(), idx[0], idx[1],
                    idx[2], ::vt::theMsg()->getEpoch() );

      // Run actual narrowphase functor
      auto &world = *this_obj.get_impl().world;
      auto &world_impl = get_impl( world );

      auto this_index = vt_index{ static_cast< std::size_t >( idx[0] ) };
      auto other_index = vt_index{ static_cast< std::size_t >( idx[2] ) };

      // Only run if we are looking at the right "other obj"
      if ( other_obj.get_impl().collision_idx != static_cast< std::size_t >( idx.y() ) )
      {
        logger.trace( "skipping <{}, {}, {}, {}> -- mismatched index",
                      this_obj.id(), idx[0], idx[1], idx[2] );
        return;
      }

      // Ignore self collisions (this will usually be caught by the above condition)
      if ( this_obj.get_impl().collision_idx == static_cast< std::size_t >( idx.y() ) )
      {
        logger.trace( "skipping <{}, {}, {}, {}> -- self collision",
                      this_obj.id(), idx[0], idx[1], idx[2] );
        return;
      }

      BVH_ASSERT_ALWAYS( this_impl.narrowphase_patch_cache.find( this_index ) != this_impl.narrowphase_patch_cache.end(),
                         logger,
                         "this_index={} - not present in `narrowphase_patch_cache`",
                         this_index );
      const auto &this_cache = this_impl.narrowphase_patch_cache.at( this_index );

      BVH_ASSERT_ALWAYS( other_impl.narrowphase_patch_cache.find( other_index ) != other_impl.narrowphase_patch_cache.end(),
                         logger,
                         "other_index={} - not present in `narrowphase_patch_cache`",
                         other_index );
      const auto &other_cache = other_impl.narrowphase_patch_cache.at( other_index );

      ::vt::NodeType left_node = this_cache.origin_node;
      ::vt::NodeType right_node = other_cache.origin_node;

      if ( world_impl.functor )
      {
        ::vt::trace::TraceScopedEvent scope( world_impl.bvh_impl_functor_ );
        auto r = world_impl.functor( this_obj, this_cache.meta, static_cast< std::size_t >( idx[0] ),
                                     this_cache.patch_data, other_obj, other_cache.meta,
                                     static_cast< std::size_t >( idx[2] ), other_cache.patch_data );

        if ( r.a.size() > 0 )
        {
          auto lmsg = ::vt::makeMessage< result_msg >();
          lmsg->result = std::move( r.a );
          logger.trace( "<send={}> result from <{}, {}, {}, {}>",
                        left_node, this_obj.id(), idx[0], idx[1], idx[2] );
          this_obj.get_impl()
            .objgroup[left_node]
            .sendMsg< result_msg, &collision_object_impl::collision_object_holder::set_result >( lmsg );
        }

        if ( r.b.size() > 0 )
        {
          auto rmsg = ::vt::makeMessage< result_msg >();
          rmsg->result = std::move( r.b );
          logger.trace( "<send={}> result from <{}, {}, {}, {}>",
                        right_node, this_obj.id(), idx[0], idx[1], idx[2] );
          this_obj.get_impl()
            .objgroup[right_node]
            .sendMsg< result_msg, &collision_object_impl::collision_object_holder::set_result >( rmsg );
        }
      }
    }

    void clear_narrowphase( collision_object_impl::narrowphase_collection_type *_narrow, clear_narrowphase_msg * )
    {
      auto &this_obj = *_narrow->this_proxy.get()->self;
      auto &logger = this_obj.narrowphase_logger();
      const auto idx = _narrow->getIndex();
      logger.trace( "clearing narrowphase index <{}, {}, {}, {}>",
                    this_obj.id(), idx[0], idx[1], idx[2] );
      _narrow->active = false;
    }

  }  // namespace collision_object_impl
}  // namespace bvh
