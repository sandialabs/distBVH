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

#include <memory>

#include <Kokkos_Core.hpp>
#include <spdlog/spdlog.h>

#include "collision_query.hpp"
#include "tree_build.hpp"

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
      set_narrowphase_functor_impl( [_fun]( collision_object &_first, const patch<> &_ma, std::size_t _first_patch_id,
                                            bvh::view< std::byte * > _first_patch, collision_object &_second,
                                            const patch<> &_mb, std::size_t _second_patch_id,
                                            bvh::view< std::byte * > _second_patch ) {
        bvh::unmanaged_view< T * > u_first_elms( reinterpret_cast< T * >( _first_patch.data() ),
                                                 _first_patch.size() / sizeof( T ) );
        bvh::host_view< T * > first_elms( "first_elements", u_first_elms.size() );
        Kokkos::deep_copy( first_elms, u_first_elms );

        bvh::unmanaged_view< T * > u_second_elms( reinterpret_cast< T * >( _second_patch.data() ),
                                                  _second_patch.size() / sizeof( T ) );
        bvh::host_view< T * > second_elms( "second_elements", u_second_elms.size() );
        Kokkos::deep_copy( second_elms, u_second_elms );

        assert( _first_patch_id == _ma.global_id() );
        assert( _second_patch_id == _mb.global_id() );

        broadphase_collision< T > first{ _first, _ma, _first_patch_id, first_elms };
        broadphase_collision< T > second{ _second, _mb, _second_patch_id, second_elms };

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

    using internal_narrowphase_functor = std::function< narrowphase_result_pair(
      collision_object &, const patch<> &, std::size_t, bvh::view< std::byte * >, collision_object &,
      const patch<> &, std::size_t, bvh::view< std::byte * > ) >;
    void set_narrowphase_functor_impl( internal_narrowphase_functor &&_fun );

    std::unique_ptr< impl > m_impl;
  };
}

#endif  // INC_BVH_COLLISION_WORLD_HPP
