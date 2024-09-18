#ifndef INC_BVH_CONTACT_ENTITY_HPP
#define INC_BVH_CONTACT_ENTITY_HPP

#include "util/kokkos.hpp"
#include "types.hpp"
#include "hash.hpp"
#include "traits.hpp"
#include <Kokkos_Core.hpp>
#include <limits>
#include "vt/print.hpp"

namespace bvh
{
  struct min_inv_diag_bounds
  {
    m::vec3d min = m::vec3d{ std::numeric_limits< double >::max(), std::numeric_limits< double >::max(),
                             std::numeric_limits< double >::max() };
    m::vec3d inv_diag = m::vec3d{ std::numeric_limits< double >::lowest(), std::numeric_limits< double >::lowest(),
                              std::numeric_limits< double >::lowest() };
  };

  namespace detail
  {
    template< typename Entity >
    struct bounds_union
    {
      using value_type = bphase_kdop;
      using size_type = typename view< const Entity * >::size_type;

      explicit bounds_union( const view< const Entity * > &_v )
        : entities( _v )
      {}

      KOKKOS_INLINE_FUNCTION
        void
        operator()( size_type _i, bphase_kdop &_update ) const noexcept
      {
        _update.union_with( entities( _i ).kdop() );
      }

      KOKKOS_INLINE_FUNCTION
        void
        join( value_type &_dst, const value_type &_src ) const noexcept
      {
        _dst.union_with( _src );
      }

      KOKKOS_INLINE_FUNCTION
        void
        init( value_type &_dst ) const noexcept
      {
        _dst = value_type{};
      }

      view< const Entity * > entities;
    };

    template< typename Entity >
    struct min_diag_bounds_union
    {
      using value_type = min_inv_diag_bounds;
      using size_type = typename view< const Entity * >::size_type;
      using traits_type = element_traits< Entity >;

      KOKKOS_INLINE_FUNCTION explicit min_diag_bounds_union( const view< const Entity * > &_v )
        : entities( _v )
      {}

      KOKKOS_INLINE_FUNCTION
      void
      operator()( size_type _i, min_inv_diag_bounds &_update ) const noexcept
      {
        const auto &cmin = traits_type::get_kdop( entities( _i ) ).cardinal_min();
        const auto &cmax = traits_type::get_kdop( entities( _i ) ).cardinal_max();
        for ( int i = 0; i < 3; ++i )
        {
          _update.min[i] = Kokkos::min( _update.min[i], cmin[i] );
          // Use diag for max right now
          _update.inv_diag[i] = Kokkos::max( _update.inv_diag[i], cmax[i] );
        }
      }

      KOKKOS_INLINE_FUNCTION
      void
      join( value_type &_dst, const value_type &_src ) const noexcept
      {
        for ( int i = 0; i < 3; ++i )
        {
          _dst.min[i] = Kokkos::min( _dst.min[i], _src.min[i] );
          // Use diag for max right now
          _dst.inv_diag[i] = Kokkos::max( _dst.inv_diag[i], _src.inv_diag[i] );
        }
      }

      KOKKOS_INLINE_FUNCTION
        void
        init( value_type &_dst ) const noexcept
      {
        _dst = {};
      }

      view< const Entity * > entities;
    };

    template< typename Entity, typename T >
    KOKKOS_INLINE_FUNCTION
    void
    morton_impl( view< const Entity * > _elements,
                 single_view< min_inv_diag_bounds > _bounds,
                 view< T * > _out_codes )
    {
      using traits_type = element_traits< Entity >;
      Kokkos::parallel_for( _elements.extent( 0 ), KOKKOS_LAMBDA( int _i ){
          const auto p = traits_type::get_centroid( _elements( _i ) );
          m::vec3< T > discretized = quantize< T >( p, _bounds().min, _bounds().inv_diag );

          _out_codes( _i ) = morton( discretized.x(), discretized.y(), discretized.z() );
        } );
    }
  }

  template< typename Entity >
  void compute_bounds( view< const Entity * > _elements,
                       single_view< bphase_kdop > _bounds )
  {
    Kokkos::parallel_reduce( "compute_bounds", _elements.extent( 0 ),
                             detail::bounds_union< Entity >{ _elements }, _bounds );
  }

  template< typename Entity >
  void compute_bounds( view< Entity * > _elements,
                  single_view< bphase_kdop > _bounds )
  {
    return compute_bounds< Entity >( view< const Entity * >( _elements ), _bounds );
  }

  template< typename Entity >
  void compute_bounds( view< const Entity * > _elements,
                       single_view< min_inv_diag_bounds > _bounds )
  {
    Kokkos::parallel_reduce( "compute_bounds_min_diag", _elements.extent( 0 ),
                             detail::min_diag_bounds_union< Entity >{ _elements },
                             _bounds );
    Kokkos::parallel_for( 1, KOKKOS_LAMBDA( int ) {
      auto width = _bounds().inv_diag - _bounds().min;
      _bounds().inv_diag = 1.0 / width;
    } );
  }

  template< typename Entity >
  void compute_bounds( view< Entity * > _elements,
                       single_view< min_inv_diag_bounds > _bounds )
  {
    compute_bounds< Entity >( view< const Entity * >( _elements ), _bounds );
  }

  template< typename Entity >
  void
  morton( view< const Entity * > _elements,
          single_view< min_inv_diag_bounds > _bounds,
          view< morton32_t * > _out_codes )
  {
    detail::morton_impl( _elements, _bounds, _out_codes );
  }

  template< typename Entity >
  void
  morton( view< Entity * > _elements,
          single_view< min_inv_diag_bounds > _bounds,
          view< morton32_t * > _out_codes )
  {
    return morton< Entity >( view< const Entity * >( _elements ), _bounds, _out_codes );
  }

  template< typename Entity >
  void
  morton( view< const Entity * > _elements,
          single_view< min_inv_diag_bounds > _bounds,
          view< morton64_t * > _out_codes )
  {
    detail::morton_impl( _elements, _bounds, _out_codes );
  }

  template< typename Entity >
  void
  morton( view< Entity * > _elements,
          single_view< min_inv_diag_bounds > _bounds,
          view< morton64_t * > _out_codes )
  {
    return morton< Entity >( view< const Entity * >( _elements ), _bounds, _out_codes );
  }

}

#endif  // INC_BVH_CONTACT_ENTITY_HPP
