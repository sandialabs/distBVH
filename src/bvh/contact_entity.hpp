#ifndef INC_BVH_CONTACT_ENTITY_HPP
#define INC_BVH_CONTACT_ENTITY_HPP

#include "util/kokkos.hpp"
#include "types.hpp"
#include "hash.hpp"

namespace bvh
{
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

    template< typename Entity, typename T >
    KOKKOS_INLINE_FUNCTION
    void
    morton_impl( view< const Entity * > _elements,
                 single_view< const bphase_kdop > _bounds,
                 view< T * > _out_codes )
    {
      Kokkos::parallel_for( _elements.extent( 0 ), KOKKOS_LAMBDA( int _i ){
          const auto p = _elements( _i ).centroid();
          m::vec3< T > discretized = quantize< T >( p, _bounds().cardinal_min(), _bounds().cardinal_max() );

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
  void
  morton( view< const Entity * > _elements,
          single_view< const bphase_kdop > _bounds,
          view< morton32_t * > _out_codes )
  {
    detail::morton_impl( _elements, _bounds, _out_codes );
  }

  template< typename Entity >
  void
  morton( view< Entity * > _elements,
          single_view< const bphase_kdop > _bounds,
          view< morton32_t * > _out_codes )
  {
    return morton< Entity >( view< const Entity * >( _elements ), _bounds, _out_codes );
  }

  template< typename Entity >
  void
  morton( view< const Entity * > _elements,
          single_view< const bphase_kdop > _bounds,
          view< morton64_t * > _out_codes )
  {
    detail::morton_impl( _elements, _bounds, _out_codes );
  }

  template< typename Entity >
  void
  morton( view< Entity * > _elements,
          single_view< const bphase_kdop > _bounds,
          view< morton64_t * > _out_codes )
  {
    return morton< Entity >( view< const Entity * >( _elements ), _bounds, _out_codes );
  }

}

#endif  // INC_BVH_CONTACT_ENTITY_HPP
