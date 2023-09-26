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
#include <optional>
#include <vt/context/context.h>

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

    template< typename T, typename... ViewProp, typename = std::enable_if_t< !std::is_same< entity_snapshot, T >::value > >
    void set_entity_data( Kokkos::View< T *, ViewProp... > _data_view, split_algorithm _algorithm )
    {
      set_entity_data( view< const T * >( std::move( _data_view ) ), _algorithm );
    }

    template< typename T, typename... ViewProp, typename = std::enable_if_t< !std::is_same< entity_snapshot, T >::value > >
    void set_entity_data( Kokkos::View< const T *, ViewProp... > _data, split_algorithm _algorithm )
    {
      switch ( _algorithm )
      {
        case split_algorithm::geom_axis: set_entity_data_geom_axis( _data ); break;
        case split_algorithm::ml_geom_axis: set_entity_data_ml_geom_axis( _data ); break;
        case split_algorithm::clustering: set_entity_data_clustering( _data ); break;
      }
    }

    /// \brief Set up data for the broadphase (including the tree)
    void init_broadphase() const;

    template< typename T, typename... ViewProp >
    void
    set_entity_data_clustering( Kokkos::View< const T *, ViewProp... > _data_view )
    {
      {
        ::vt::trace::TraceScopedEvent scope( this->bvh_clustering_ );
        const auto n = _data_view.extent( 0 );

        const auto od_factor = this->overdecomposition_factor();
        const auto num_splits = od_factor - 1;

        ::bvh::vt::debug( "{}: clustering {} elements\n", ::vt::theContext()->getNode(), n  );
        if ( !m_clusterer || ( n != m_clusterer->size() ) )
        {
          m_clusterer = morton_cluster( n );
          Kokkos::resize( get_split_indices(), n );
          Kokkos::resize( get_split_indices_h(), n );
        }

        Kokkos::resize( get_splits(), num_splits );
        Kokkos::resize( get_splits_h(), num_splits );

        // Initialize our indices
        Kokkos::parallel_for(
          n, KOKKOS_LAMBDA( int _i ) { get_split_indices()( _i ) = _i; } );

        ( *m_clusterer )( _data_view, get_split_indices(), get_splits() );

        Kokkos::deep_copy( get_splits_h(), get_splits() );
        Kokkos::deep_copy( get_split_indices_h(), get_split_indices() );

        // Now split_indices/_h is reordered according to the morton encoding
        // It provides a mapping from original indices to the new reordered elements that
        // are clustered by locality

        update_snapshots( _data_view );
      }
      // This assumes _data_view is on host for now... at the moment we can't do much better
      {
        ::vt::trace::TraceScopedEvent scope( this->bvh_set_entity_data_impl_ );
        set_entity_data_impl( _data_view.data(), sizeof( T ) );
      }
    }

    template< typename T, typename... ViewProp >
    void set_entity_data_ml_geom_axis( Kokkos::View< const T *, ViewProp... > _data )
    {
      // Split data by overdecomposition factor
      const auto od_factor = this->overdecomposition_factor();
      int depth = bit_log2( od_factor );
      update_snapshots_without_permuting( _data );
      {
        ::vt::trace::TraceScopedEvent scope( this->bvh_splitting_ml_ );
        Kokkos::fence();  // snapshots need to finish updating
        split_permutations_ml< split::mean, axis::longest, bvh::entity_snapshot >( get_snapshots(), depth,
                                                                                    &m_last_permutations );
        initialize_split_indices( m_last_permutations );
      }
      {
        ::vt::trace::TraceScopedEvent scope( this->bvh_set_entity_data_impl_ );
        set_entity_data_impl( _data.data(), sizeof( T ) );
      }
    }

    template< typename T, typename... ViewProp >
    void set_entity_data_geom_axis( Kokkos::View< const T *, ViewProp... > _data )
    {
      // Split data by overdecomposition factor
      const auto od_factor = this->overdecomposition_factor();
      int depth = bit_log2( od_factor );
      ::vt::trace::TraceScopedEvent scope( this->bvh_splitting_geom_axis_ );
      split_permutations< split::mean, axis::longest, T >( _data, depth, &m_last_permutations );
      set_entity_data_with_permutations( _data, m_last_permutations, std::move( scope ) );
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

    span< const patch<> > local_patches() const noexcept;

  private:

    friend class collision_world;

    template< typename T, typename...ViewProp >
    void set_entity_data_with_permutations( Kokkos::View< const T *, ViewProp... > _data, const element_permutations &_splits, ::vt::trace::TraceScopedEvent &&_trace )
    {
      always_assert( _splits.indices.size() == _data.extent( 0 ), "must have a split index per data element!" );

      initialize_split_indices( _splits );

      update_snapshots( _data );

      set_entity_data_impl( _data.data(), sizeof( T ) );
      std::move( _trace ).end();
    }

    collision_object( collision_world &_world, std::size_t _idx, std::size_t _overdecomposition );

    /// \brief Implementation for setting the container of data
    ///
    /// \param[in] _data
    /// \param[in] _element_size
    void set_entity_data_impl( const void *_data, std::size_t _element_size );

    void set_all_narrow_patches();
    void set_active_narrow_patches();
    void narrowphase(collision_object &_other );

    struct impl;

  public:

    impl &get_impl() noexcept { return *m_impl; }
    const impl &get_impl() const noexcept { return *m_impl; }

  private:

    template< typename T, typename... ViewProp >
    void
    update_snapshots( Kokkos::View< const T *, ViewProp... > _data_view )
    {
      // No-op if the view is the same size, which is typically the case
      auto &snap = get_snapshots();
      Kokkos::resize( Kokkos::WithoutInitializing, snap, _data_view.extent( 0 ) );
      auto &ind = get_split_indices_h();
      Kokkos::parallel_for(
        ind.extent( 0 ), KOKKOS_LAMBDA( int _idx ) {
          snap( ind( _idx ) ) = make_snapshot( _data_view( _idx ), static_cast< std::size_t >( _idx ) );
        } );
    }

    template< typename T, typename... ViewProp >
    void
    update_snapshots_without_permuting( Kokkos::View< const T *, ViewProp... > _data_view )
    {
      // No-op if the view is the same size, which is typically the case
      auto &snap = get_snapshots();
      Kokkos::resize( Kokkos::WithoutInitializing, snap, _data_view.extent( 0 ) );
      Kokkos::parallel_for(
        _data_view.extent( 0 ), KOKKOS_LAMBDA( int _idx ) {
          snap( _idx ) = make_snapshot( _data_view( _idx ), static_cast< std::size_t >( _idx ) );
        } );
    }

    void for_each_tree_impl( tree_function &&_fun );
    void for_each_result_impl( std::function< void(const narrowphase_result &) > &&_fun );

    view< bvh::entity_snapshot * > &get_snapshots();
    view< std::size_t * > &get_split_indices();
    view< std::size_t * > &get_splits();
    host_view< std::size_t * > &get_split_indices_h();
    host_view< std::size_t * > &get_splits_h();

    void initialize_split_indices( const element_permutations &_splits );

    std::unique_ptr< impl > m_impl;

    // Only used as scratch space for CPU based splitting
    element_permutations m_last_permutations;

    ::vt::trace::UserEventIDType bvh_splitting_geom_axis_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_splitting_ml_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_set_entity_data_impl_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_clustering_ = ::vt::trace::no_user_event_id;
    ::vt::trace::UserEventIDType bvh_build_trees_ = ::vt::trace::no_user_event_id;

    std::optional< morton_cluster > m_clusterer; // lazily initialized
  };
}

#endif  // INC_BVH_COLLISION_OBJECT_HPP
