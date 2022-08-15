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
#ifndef INC_BVH_COLLISION_OBJECT_HPP
#define INC_BVH_COLLISION_OBJECT_HPP

#include <cstddef>
#include <memory>
#include <functional>

#include "snapshot.hpp"
#include "split/split.hpp"
#include "split/mean.hpp"
#include "split/axis.hpp"
#include "tree_build.hpp"
#include "types.hpp"
#include "util/span.hpp"
#include "split/cluster.hpp"
#include "contact_entity.hpp"

namespace bvh
{
  class collision_world;

  class collision_object
  {
  public:

    using tree_function = std::function< void( const snapshot_tree & ) >;

    collision_object( const collision_object & ) = delete;
    collision_object( collision_object && ) noexcept;

    collision_object &operator=( const collision_object & ) = delete;
    collision_object &operator=( collision_object && ) noexcept;

    ~collision_object();

    template< typename T, typename = std::enable_if_t< !std::is_same< entity_snapshot, T >::value > >
    void set_entity_data( span< const T > _data, const element_permutations &splits, ::vt::trace::TraceScopedEvent &&_trace )
    {
      Kokkos::resize( m_split_indices, splits.indices.size() );
      Kokkos::resize( m_split_indices_h, splits.indices.size() );

      Kokkos::resize( m_splits, splits.splits.size() );
      Kokkos::resize( m_splits_h, splits.splits.size() );

      Kokkos::View< const std::size_t *, bvh::host_execution_space, Kokkos::MemoryTraits< Kokkos::Unmanaged > > indices_view( splits.indices.data(), splits.indices.size() );
      Kokkos::View< const std::size_t *, bvh::host_execution_space, Kokkos::MemoryTraits< Kokkos::Unmanaged > > splits_view( splits.splits.data(), splits.splits.size() );

      Kokkos::resize( m_snapshots, splits.indices.size() );

      // TODO: For now just assuming host space until everything gets moved over to views...
      Kokkos::parallel_for( splits.indices.size(), KOKKOS_LAMBDA( int _i ){
        m_snapshots( _i ) = make_snapshot( _data[splits.indices[_i]], splits.indices[_i] );
      } );

      Kokkos::deep_copy( m_splits_h, splits_view );
      Kokkos::deep_copy( m_split_indices_h, indices_view );

      set_entity_data_impl( _data.data(), sizeof( T ), splits.splits.size() );
      std::move( _trace ).end();
    }

    template< typename T, typename = std::enable_if_t< !std::is_same< entity_snapshot, T >::value > >
    void set_entity_data( const std::vector< T > &_data , const element_permutations &splits )
    {
      span< const T > _data_span{ _data.data(), _data.size() };
      this->set_entity_data< T >( _data_span, splits );
    }

    template< typename T, typename = std::enable_if_t< !std::is_same< entity_snapshot, T >::value > >
    void set_entity_data( span< const T > _data, bvh::split_algorithm _algorithm = geom_axis)
    {
      //
      // Split data by overdecomposition factor
      //
      const auto od_factor = this->overdecomposition_factor();
      //
      int depth = bit_log2( od_factor );
      //
      if (_algorithm == geom_axis) {
        ::vt::trace::TraceScopedEvent scope(this->bvh_splitting_geom_axis_);
        split_permutations< split::mean, axis::longest >( _data, depth, &m_last_permutations );
        set_entity_data<T>( _data, m_last_permutations, std::move( scope ) );
      }
      else {
        m_snapshots.clear();
        m_snapshots.reserve( _data.size() );
        for (size_t ii = 0; ii < _data.size(); ++ii) {
          m_snapshots.emplace_back( make_snapshot( _data[ii], ii ) );
        }
        {
          ::vt::trace::TraceScopedEvent scope(this->bvh_splitting_ml_);
          split_permutations_ml< split::mean, axis::longest, bvh::entity_snapshot >(m_snapshots, depth, &m_last_permutations);
        }
        {
          ::vt::trace::TraceScopedEvent scope(this->bvh_set_entity_data_impl_);
          set_entity_data_impl(m_snapshots, _data.data(), sizeof( T ), m_last_permutations);
        }
      }
    }

    template< typename T, typename = std::enable_if_t< !std::is_same< entity_snapshot, T >::value > >
    void set_entity_data( const std::vector< T > &_data, bvh::split_algorithm _algorithm = geom_axis)
    {
      span< const T > _data_span{ _data.data(), _data.size() };
      this->set_entity_data< T >( _data_span, _algorithm );
    }

    /// \brief Set up data for the broadphase (including the tree)
    void init_broadphase() const;

    template< typename T >
    void
    set_entity_data( view< T * > _data_view )
    {
      set_entity_data( view< const T * >( std::move( _data_view ) ) );
    }
    template< typename T >
    void
    set_entity_data( view< const T * > _data_view )
    {
      static const auto n = _data_view.extent( 0 );
      if ( n != m_clusterer.size() )
      {
        m_clusterer = morton_cluster( n );
        Kokkos::resize( m_split_indices, n );
        Kokkos::resize( m_splits, n );
        Kokkos::resize( m_split_indices_h, n );
        Kokkos::resize( m_splits_h, n );
      }

      const auto od_factor = this->overdecomposition_factor();
      const int depth = bit_log2( od_factor );
      std::size_t num_splits = 0;

      m_clusterer( _data_view, m_split_indices, m_splits, depth, num_splits );

      update_snapshots( _data_view );

      // Maybe we can do better at some point...
      auto hview = Kokkos::create_mirror_view_and_copy( bvh::default_execution_space{}, _data_view );

      Kokkos::deep_copy( m_splits_h, m_splits );
      Kokkos::deep_copy( m_split_indices_h, m_split_indices );

      set_entity_data_impl( hview.data(), sizeof( T ), num_splits );
    }

    template< typename F >
    void for_each_tree( F &&_fun )
    {
      for_each_tree_impl( tree_function{ std::forward< F >( _fun ) } );
    }

    void broadphase( collision_object &_other );

    void end_phase();

    int overdecomposition_factor() const noexcept;

    std::size_t id() const noexcept;

    template< typename ResultType, typename F >
    void for_each_result( F &&_fun )
    {
      for_each_result_impl( [&_fun]( const narrowphase_result &_res ) {
        for ( std::size_t i = 0; i < _res.size(); ++i )
        {
          std::forward< F >( _fun )( *reinterpret_cast< const ResultType * >( _res.at( i ) ) );
        }
      } );
    }

  private:

    friend class collision_world;

    template< typename T >
    void
    update_snapshots( view< const T * > _data_view )
    {
      // No-op if the view is the same size, which is typically the case
      Kokkos::resize( Kokkos::WithoutInitializing, m_snapshots, _data_view.extent( 0 ) );
      Kokkos::parallel_for( _data_view.extent( 0 ), KOKKOS_LAMBDA( int _idx ){
        m_snapshots( _idx ) = make_snapshot( _data_view( _idx ) );
      } );
    }

    collision_object( collision_world &_world, std::size_t _idx, std::size_t _overdecomposition );

    /// \brief Implementation for setting the container of data
    ///
    /// \param[in] _user
    /// \param[in] _element_size
    /// \param[in] _num_splits the actual number of splits
    void set_entity_data_impl( const void *_user, std::size_t _element_size, std::size_t _num_splits );

    void set_all_narrow_patches();
    void set_active_narrow_patches();
    void narrowphase(collision_object &_other );

    struct impl;

  public:

    impl &get_impl() noexcept { return *m_impl; }
    const impl &get_impl() const noexcept { return *m_impl; }

  private:

    void for_each_tree_impl( tree_function &&_fun );
    void for_each_result_impl( std::function< void(const narrowphase_result &) > &&_fun );

    std::unique_ptr< impl > m_impl;

    element_permutations m_last_permutations;

    ::vt::trace::UserEventIDType bvh_splitting_geom_axis_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_splitting_ml_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_set_entity_data_impl_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_build_trees_ = ::vt::trace::no_user_event_id;

    view< bvh::entity_snapshot * > m_snapshots;
    view< std::size_t * > m_split_indices;
    view< std::size_t * > m_splits;
    host_view< std::size_t * > m_split_indices_h;
    host_view< std::size_t * > m_splits_h;

    morton_cluster m_clusterer;
  };
}

#endif  // INC_BVH_COLLISION_OBJECT_HPP
