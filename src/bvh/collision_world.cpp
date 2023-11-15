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
#include "collision_world.hpp"
#include "collision_object.hpp"
#include "collision_world/impl.hpp"
#include "logging.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <vt/transport.h>
#include <vt/trace/trace_lite.h>

namespace bvh
{

  collision_world::collision_world( std::size_t _overdecomposition_factor )
    : m_impl( std::make_unique< impl >() )
  {
    auto stdout_sink = std::make_shared< spdlog::sinks::stdout_color_sink_st >();
    stdout_sink->set_level( spdlog::level::trace );
    m_impl->collision_world_logger = logging::make_logger( "collision_world", stdout_sink );
    m_impl->collision_world_logger->trace( "Initialized collision world logger" );
    m_impl->collision_object_logger = logging::make_logger( "collision_object", stdout_sink );
    m_impl->collision_world_logger->trace( "Initialized collision object logger" );
    m_impl->collision_object_broadphase_logger = logging::make_logger( "collision_object.broadphase", stdout_sink );
    m_impl->collision_world_logger->trace( "Initialized collision object broadphase logger" );

    m_impl->overdecomposition = _overdecomposition_factor;
    auto user_event_name = "bvh_impl_functor_";
    m_impl->bvh_impl_functor_ = ::vt::theTrace()->registerUserEventColl( user_event_name);
    m_impl->collision_world_logger->trace( "registered user tracing event {}", user_event_name );

    m_impl->collision_world_logger->info( "Initialized collision world with overdecomposition factor {}", _overdecomposition_factor );
  }

  collision_world::~collision_world() = default;

  collision_world::collision_world( collision_world &&_other ) noexcept = default;
  collision_world &collision_world::operator=( collision_world &&_other ) noexcept = default;

  collision_object &
  collision_world::create_collision_object()
  {
    std::size_t idx = m_impl->collision_objects.size();
    // Use new allocator here because of private collision_object constructor
    m_impl->collision_objects.emplace_back( new collision_object( *this, idx, m_impl->overdecomposition ) );

    return *m_impl->collision_objects.back();
  }

  std::size_t
  collision_world::num_collision_objects() const noexcept
  {
    return m_impl->collision_objects.size();
  }

  std::size_t
  collision_world::overdecomposition_factor() const noexcept
  {
    return m_impl->overdecomposition;
  }

  void
  collision_world::set_narrowphase_functor_impl( internal_narrowphase_functor &&_fun )
  {
    m_impl->functor = std::move( _fun );
  }

  void
  collision_world::start_iteration()
  {
    m_impl->epoch = ::vt::theTerm()->makeEpochCollective( "iteration" );

    ::vt::theMsg()->pushEpoch( m_impl->epoch );
  }

  void
  collision_world::finish_iteration()
  {
    for ( auto &&obj : m_impl->collision_objects )
      obj->end_phase();

    ::vt::theMsg()->popEpoch( m_impl->epoch );


    ::vt::theTerm()->finishedEpoch( m_impl->epoch );
    ::vt::runSchedulerThrough( m_impl->epoch );

    // --vt_lb enables lb
    // --vt_lb_name="" RotateLB, RandomLB
    ::vt::thePhase()->nextPhaseCollective();

    m_impl->epoch = ::vt::no_epoch;
  }

  std::shared_ptr< spdlog::logger >
  collision_world::collision_object_logger() const
  {
    return m_impl->collision_object_logger;
  }

  std::shared_ptr< spdlog::logger >
  collision_world::collision_object_broadphase_logger() const
  {
    return m_impl->collision_object_broadphase_logger;
  }
}
