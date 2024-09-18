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
#ifndef INC_BVH_SPLIT_HPP
#define INC_BVH_SPLIT_HPP

#include "../debug/assert.hpp"
#include "../range.hpp"
#include "../traits.hpp"
#include "../util/span.hpp"
#include "../util/kokkos.hpp"

#include "../snapshot.hpp"

#include "element_permutations.hpp"

#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <iterator>
#include <numeric>
#include <algorithm>
#include <vector>


namespace bvh
{
  template< typename SplittingMethod, typename Iterator >
  auto split_in_place( range< Iterator > _range, int _axis )
  {
    using traits_type = element_traits< typename std::iterator_traits< Iterator >::value_type >;
    using kdop_type = typename traits_type::kdop_type;

    auto split_point = SplittingMethod::split_point( _range, _axis );

    auto split_iter = std::partition( _range.begin(), _range.end(),
      [split_point, _axis]( const auto &e ) {
        return kdop_type::project( traits_type::get_centroid( e ), _axis ) < split_point;
    } );

    // If one partition is size 0, just take the middle element
    if ( ( _range.begin() == split_iter ) || ( _range.end() == split_iter ) )
    {
      split_iter = _range.begin();
      std::advance( split_iter, _range.distance() / 2 );
    }

    return split_iter;
  }

  namespace detail
  {
    template< typename SplittingMethod, typename AxisSelector, typename Iterator >
    void
    split_in_place_recursive_impl( range< Iterator > _range, int _depth, std::vector< Iterator > &_splits )
    {
      using traits_type = element_traits< typename std::iterator_traits< Iterator >::value_type >;
      using kdop_type = typename traits_type::kdop_type;

      auto kdops = bvh::transform_range( _range, traits_type::get_kdop );
      auto kdop = kdop_type::from_kdops( kdops.begin(), kdops.end() );

      int axis = AxisSelector::axis( kdop );

      auto sp = split_in_place< SplittingMethod >( _range, axis );

      if ( _depth > 0 )
        split_in_place_recursive_impl< SplittingMethod, AxisSelector >( make_range( _range.begin(), sp ), _depth - 1,
                                                                        _splits );

      _splits.emplace_back( sp );

      // Check if range begin is sp because that means we are only at one element
      if ( _depth > 0 )
        split_in_place_recursive_impl< SplittingMethod, AxisSelector >( make_range( sp, _range.end()), _depth - 1,
                                                                        _splits );
    }

    using permute_range = range< std::vector< std::size_t >::iterator >;

    /// \brief Function to split entities along an axis and
    ///        to gather entities on each side of the axis
    ///
    /// \tparam        SplittingMethod
    /// \param[in,out] _elements
    /// \param[in,out] _perm
    /// \param[in]     _axis   Coordinate index of splitting axis
    /// \param[in,out] combi
    ///
    /// \return   Offset to separating entity
    ///
    template< typename SplittingMethod >
    long int
    split_permutation_ml( span< bvh::entity_snapshot > _elements,
                           span< size_t > _perm, int _axis,
                          span< std::pair< bvh::entity_snapshot, size_t > > combi )
    {
      if (_elements.size() < 2) {
        return 0;
      }

      using traits_type = element_traits< bvh::entity_snapshot >;
      using kdop_type = typename traits_type::kdop_type;

      auto split_point = SplittingMethod::split_point( _elements, _axis );

      auto split_combi = std::partition( combi.begin(), combi.end(),
                                [split_point, _axis]( std::pair< bvh::entity_snapshot, size_t > &entry ) {
                                      const auto &e = std::get<0>( entry );
                                      const auto &c = e.centroid();
                                      return kdop_type::project( c, _axis ) < split_point;
                                        } );

      Kokkos::View< size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_perm(_perm.data(), _perm.size());
      Kokkos::View< bvh::entity_snapshot*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_elements(_elements.data(), _elements.size());
      Kokkos::parallel_for("CopyLoop", _elements.size(), KOKKOS_LAMBDA (const int& i) {
                       auto tmp_pair = combi[i];
                       h_elements[i] = std::get<0>( tmp_pair );
                       h_perm(i) = std::get<1>( tmp_pair );
                     } );

      auto delta = std::distance( combi.begin(), split_combi );
      if ( ( combi.begin() == split_combi ) || ( combi.end() == split_combi ) ) {
        delta = _elements.size() / 2;
      }

      return delta;
    }

    template< typename SplittingMethod, typename Element >
    auto split_permutation( span< const Element > _elements,
                           permute_range _perm, int _axis )
    {
      using traits_type = element_traits< Element >;
      using kdop_type = typename traits_type::kdop_type;

      auto split_point = SplittingMethod::split_point( make_range( _elements.begin(), _elements.end() ), _axis );

      auto split_iter = std::partition( _perm.begin(), _perm.end(),
                                        [split_point, _axis, _elements]( std::size_t _i ) {
                                          auto &e = _elements[_i];
                                          return kdop_type::project( traits_type::get_centroid( e ), _axis ) < split_point;
                                        } );

      // If one partition is size 0, just take the middle element
      if ( ( _perm.begin() == split_iter ) || ( _perm.end() == split_iter ) )
      {
        split_iter = _perm.begin();
        std::advance( split_iter, _perm.distance() / 2 );
      }

      return split_iter;
    }

    template< typename SplittingMethod, typename AxisSelector, typename Element = bvh::entity_snapshot >
    void
    split_permutations_recursive_impl_ml( span< Element > _elements, int _depth,
                                       const std::vector< std::size_t >::iterator _start,
                                       span< size_t > _perm, std::vector< std::size_t > &_splits,
                                       span< std::pair< Element, size_t > > combi)
    {
      int axis = 0;
      if (_elements.size() > 0) {
        using traits_type = element_traits< Element >;
        using kdop_type = typename traits_type::kdop_type;
        auto kdops = bvh::transform_range( _elements.begin(), _elements.end(), traits_type::get_kdop );
        auto kdop = kdop_type::from_kdops( kdops.begin(), kdops.end() );
        axis = AxisSelector::axis( kdop );
      }

      auto delta = split_permutation_ml< SplittingMethod >( _elements, _perm, axis, combi );

      auto sp_ptr = _perm.begin();
      std::advance(sp_ptr, delta);

      auto ep = _elements.begin();
      std::advance(ep, delta);

      auto cbp = combi.begin();
      std::advance(cbp, delta);

      if ( _depth > 0 ) {
        split_permutations_recursive_impl_ml< SplittingMethod, AxisSelector >(
            span< Element >{_elements.begin(), ep}, _depth - 1,
            _start, span<size_t>{_perm.begin(), sp_ptr}, _splits,
            span< std::pair< Element, size_t > >{combi.begin(), cbp} );
      }

      _splits.emplace_back( std::distance( &(*_start), sp_ptr ) );

      if ( _depth > 0 ) {
        split_permutations_recursive_impl_ml< SplittingMethod, AxisSelector >(
            span< Element >{ep, _elements.end()}, _depth - 1,
            _start, span<size_t>{sp_ptr, _perm.end()}, _splits,
            span< std::pair< Element, size_t > >{cbp, combi.end()} );
      }

    }

    template< typename SplittingMethod, typename AxisSelector, typename Element >
    void
    split_permutations_recursive_impl( span< const Element > _elements, int _depth,
                                       const std::vector< std::size_t >::iterator _start,
                                       permute_range _perm, std::vector< std::size_t > &_splits )
    {
      using traits_type = element_traits< Element >;
      using kdop_type = typename traits_type::kdop_type;

      auto kdops = bvh::transform_range( _elements.begin(), _elements.end(), traits_type::get_kdop );
      auto kdop = kdop_type::from_kdops( kdops.begin(), kdops.end() );

      int axis = AxisSelector::axis( kdop );
      auto sp = split_permutation< SplittingMethod >( _elements, _perm, axis );

      if ( _depth > 0 ) {
        split_permutations_recursive_impl< SplittingMethod, AxisSelector >( _elements, _depth - 1,
            _start, make_range( _perm.begin(), sp ), _splits );
      }

      _splits.emplace_back( std::distance( _start, sp ) );

      // Check if range begin is sp because that means we are only at one element
      if ( _depth > 0 ) {
        split_permutations_recursive_impl< SplittingMethod, AxisSelector >( _elements, _depth - 1,
                                                                        _start, make_range( sp, _perm.end()), _splits );
      }

    }
  }

  /**
   * Split a range in place recursively until _depth. Will make a full tree that can include
   * empty ranges.
   *
   * \tparam SplittingMethod
   * \tparam AxisSelector
   * \tparam Iterator
   * \param _range
   * \param _depth
   * \return
   */
  template< typename SplittingMethod, typename AxisSelector, typename Iterator >
  auto
  split_in_place_recursive( range< Iterator > _range, int _depth )
  {
    std::vector< Iterator > ret;
    ret.reserve( ( 1ULL << static_cast< unsigned >( _depth ) ) + 1 );

    ret.push_back( _range.begin() );

    if ( _depth > 0 )
      detail::split_in_place_recursive_impl< SplittingMethod, AxisSelector >( _range, _depth - 1, ret );

    ret.push_back( _range.end() );

    return ret;
  }

  template< typename SplittingMethod, typename AxisSelector, typename Element = bvh::entity_snapshot >
  void
  split_permutations_ml( span< Element > _elements, int _depth, element_permutations *_permutations )
  {
    _permutations->indices.resize( _elements.size() );
    std::iota( _permutations->indices.begin(), _permutations->indices.end(), 0 );

    const auto splits_len = (1ULL << static_cast< unsigned >( _depth )) + 1;
    _permutations->splits.clear();
    _permutations->splits.reserve( splits_len );

    if ( _depth > 0 ) {
      std::vector< std::pair< Element, size_t > > combi( _elements.size() );
      Kokkos::View< std::pair< Element, size_t >*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > h_combi(combi.data(), _elements.size());
      Kokkos::parallel_for("CopyInit", Kokkos::RangePolicy< Kokkos::DefaultHostExecutionSpace >( 0, static_cast< int >( _elements.size() ) ), KOKKOS_LAMBDA (int i) {
                     h_combi[i] = std::make_pair( _elements[i], i );
                     } );
      detail::split_permutations_recursive_impl_ml< SplittingMethod, AxisSelector >( _elements, _depth - 1,
                        _permutations->indices.begin(),
                        _permutations->indices, _permutations->splits,
                        span< std::pair< Element, size_t > >{combi.data(), _elements.size()} );
    }
  }

  template< typename SplittingMethod, typename AxisSelector, typename Element >
  void
  split_permutations( span< const Element > _elements, int _depth, element_permutations *_permutations )
  {
    _permutations->indices.resize( _elements.size() );
    std::iota( _permutations->indices.begin(), _permutations->indices.end(), 0 );

    _permutations->splits.clear();
    _permutations->splits.reserve( ( 1ULL << static_cast< unsigned >( _depth ) ) + 1 );

    if ( _depth > 0 )
      detail::split_permutations_recursive_impl< SplittingMethod, AxisSelector >( _elements, _depth - 1,
                                                                                  _permutations->indices.begin(),
                                                                               make_range( _permutations->indices.begin(), _permutations->indices.end() ), _permutations->splits );
  }
}

#endif  // INC_BVH_SPLIT_HPP
