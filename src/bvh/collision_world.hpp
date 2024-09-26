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
#ifndef INC_BVH_COLLISION_WORLD_HPP
#define INC_BVH_COLLISION_WORLD_HPP

#include <vector>
#include <memory>
#include "collision_query.hpp"
#include "snapshot.hpp"
#include "util/functional.hpp"
#include "tree_build.hpp"
#include <spdlog/spdlog.h>

namespace bvh
{
  class collision_object;

  struct world_config
  {
    spdlog::level::level_enum log_levels = spdlog::level::trace;
    spdlog::level::level_enum flush_level = spdlog::level::trace;
  };

  class collision_world
  {
  public:

    template< typename T >
    using narrowphase_functor = std::function< narrowphase_result_pair( const broadphase_collision< T > &, const broadphase_collision< T > & ) >;

    explicit collision_world( std::size_t _overdecomposition_factor, const world_config &_cfg = {} );
    ~collision_world();

    collision_world( const collision_world & ) = delete;
    collision_world( collision_world &&_other ) noexcept;

    collision_world &operator=( const collision_world & ) = delete;
    collision_world &operator=( collision_world &&_other ) noexcept;

    collision_object &create_collision_object();

    std::size_t num_collision_objects() const noexcept;
    std::size_t overdecomposition_factor() const noexcept;

    template< typename T >
    void set_narrowphase_functor( narrowphase_functor< T > _fun )
    {
      // FIXME_CUDA: static casts from const void * need to be changed into the correct view type with element type T
      //             (do the `internal_narrowphase_functor` change first)
      set_narrowphase_functor_impl( [_fun]( collision_object &_first, const patch<> &_ma, std::size_t _first_patch_id, const void *_first_patch, std::size_t _first_patch_size,
                                           collision_object &_second, const patch<> &_mb,  std::size_t _second_patch_id, const void *_second_patch, std::size_t _second_patch_size ) {
        const T *first_elms = static_cast< const T * >( _first_patch );
        const T *second_elms = static_cast< const T * >( _second_patch );

        assert( _first_patch_id == _ma.global_id() );
        assert( _second_patch_id == _mb.global_id() );

        broadphase_collision< T > first{ _first, _ma, _first_patch_id,
              span< const T >( first_elms, _first_patch_size / sizeof( T ) ) };

        broadphase_collision< T > second{ _second, _mb, _second_patch_id,
              span< const T >( second_elms, _second_patch_size / sizeof( T ) ) };

        return _fun( first, second );
      } );
    }

    void start_iteration();
    void finish_iteration();

    std::shared_ptr< spdlog::logger > collision_object_logger() const;
    std::shared_ptr< spdlog::logger > collision_object_broadphase_logger() const;
    std::shared_ptr< spdlog::logger > collision_object_narrowphase_logger() const;

  private:

    struct impl;

    friend impl &get_impl( collision_world &_world );

    // FIXME_CUDA: replace void* with a View
    using internal_narrowphase_functor
        = std::function< narrowphase_result_pair( collision_object &, const patch<> &, std::size_t, const void *, std::size_t, collision_object &, const patch<> &, std::size_t, const void *, std::size_t ) >;
    void set_narrowphase_functor_impl( internal_narrowphase_functor &&_fun );

    std::unique_ptr< impl > m_impl;
  };
}

#endif  // INC_BVH_COLLISION_WORLD_HPP
