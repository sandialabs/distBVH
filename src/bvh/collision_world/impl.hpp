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
#ifndef INC_BVH_COLLISION_WORLD_IMPL_HPP
#define INC_BVH_COLLISION_WORLD_IMPL_HPP

#include "../collision_world.hpp"
#include <vt/transport.h>
#include <vt/trace/trace_common.h>

#define SPDLOG_HEADER_ONLY
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace bvh
{

  namespace collision_world_impl
  {
  }

  inline collision_world::impl &get_impl( collision_world &_world )
  {
    return *_world.m_impl;
  }

  struct collision_world::impl
  {
    std::vector< std::unique_ptr< collision_object > > collision_objects;

    collision_world::internal_narrowphase_functor functor;

    std::size_t overdecomposition = 2;
    ::vt::EpochType epoch;

    ::vt::trace::UserEventIDType bvh_impl_functor_ = ::vt::trace::no_user_event_id;

    std::shared_ptr< spdlog::logger > collision_world_logger;
    std::shared_ptr< spdlog::logger > collision_object_logger;
    std::shared_ptr< spdlog::logger > collision_object_broadphase_logger;
    std::shared_ptr< spdlog::logger > collision_object_narrowphase_logger;
  };
}

#endif // INC_BVH_COLLISION_WORLD_IMPL_HPP
