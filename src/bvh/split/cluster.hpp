#ifndef INC_BVH_CLUSTER_HPP
#define INC_BVH_CLUSTER_HPP

#include "../util/span.hpp"
#include "../util/kokkos.hpp"
#include "../util/sort.hpp"
#include "../snapshot.hpp"
#include "split.hpp"

namespace bvh
{
  class morton_cluster
  {
  public:

    using index_type = std::size_t;

    morton_cluster() = default;
    explicit morton_cluster( std::size_t _n );

    template< typename Element >
    void operator()( view< const Element * > _elements,
                     view< index_type * > _indices,
                     view< index_type * > _splits,
                     int _d, std::size_t &_out_num_splits );

    std::size_t size() const noexcept { return m_size; }

  private:

    using morton_type = std::uint32_t;

    std::size_t m_size = 0;
    single_view< bvh::bphase_kdop > m_bounds;
    view< morton_type * > m_hashes;
    radix_sorter< morton_type, index_type > m_sorter;
    single_view< std::size_t > m_count;
    single_host_view< std::size_t > m_host_count;
    view< index_type * > m_expanded_indices;
    view< index_type * > m_should_split;
  };

  inline morton_cluster::morton_cluster( std::size_t _n )
    : m_size( _n ),
      m_bounds( "morton_cluster_bounds" ),
      m_hashes( "morton_hashes", _n ),
      m_sorter( _n ),
      m_count( "morton_cluster_split_count" ),
      m_host_count( "morton_cluster_split_count_host" ),
      m_should_split( "morton_cluster_should_split", _n ),
      m_expanded_indices( "morton_cluster_expanded_indices", _n )
  {

  }

  template< typename Element >
  void
  morton_cluster::operator()( view< const Element * > _elements,
                            view< index_type * > _indices,
                            view< index_type * > _splits,
                            int _d, std::size_t &_out_num_splits )
  {
    // Depth of zero is not interesting
    assert( _d > 0 );

          // Well we can't have more than 30 bits of morton hash
    assert( _d <= 30 );

    assert( m_hashes.extent( 0 ) == m_size );
    assert( m_hashes.extent( 0 ) == _elements.extent( 0 ) );
    assert( m_hashes.extent( 0 ) == _indices.extent( 0 ) );
    assert( m_hashes.extent( 0 ) == _splits.extent( 0 ) );

    // Reinitialize count; we don't need to reset our buffers since they get overwritten
    Kokkos::deep_copy( m_count, 0 );

    // 1. Compute morton codes
    // 2. Sort the indices according to spatial hash
    // 3. Compute split points from d-bit change

    compute_bounds( _elements, m_bounds );
    morton( _elements, m_bounds, m_hashes );

    m_sorter( m_hashes, _indices );

    // We may have some imbalance on the resulting "tree"
    // There may be a better approach that could rectify this

    // Subtract 1 here because msb of 0 split still has two choices
    // But then add 2 because of the morton encoding padding
    // If we change this, we could create a subtle bug here since
    // 32 bit morton hashes have two bits of padding, but 64 bit morton
    // hashes only have one bit. This static_assert is here to trigger
    // a compiler error if the assumption changes
    static_assert( sizeof( morton_type ) == 4 );

    const auto n = m_hashes.extent( 0 );
    // Exclude the last index from the range since there is not going
    // to be a split in the tree after it...
    Kokkos::parallel_for( n - 1, [this, _d] KOKKOS_FUNCTION( int _i ) {
      for ( morton_type shift = 0; shift < morton_type( _d ); ++shift )
      {
        // We should only be looking at the lower 30 bits (10 bit morton codes per component), so skip top two
        const auto msb_shift = static_cast< morton_type >( shift + 2 ); // This is always >= 2 and <= 31
        const morton_type mask = 0x1 << ( 31 - msb_shift );

        // If the bit at our depth differs, that means there is a split
        // in the tree
        // This split carries on to the leaves so we can exit early
        if ( ( m_hashes( _i ) & mask ) != ( m_hashes( _i + 1 ) & mask ) )
        {
          m_should_split( _i ) = 1;
        } else {
          m_should_split( _i ) = 0;
        }
      }
    } );

    // Get the compressed indices preserving the order
    Kokkos::deep_copy( m_expanded_indices, m_should_split );
    prefix_sum( m_expanded_indices );

    Kokkos::parallel_for( n - 1, [this, _splits, n] KOKKOS_FUNCTION( int _i ) {
      if ( m_should_split( _i ) != 0 )
      {
        auto new_idx = m_expanded_indices( _i );
        _splits( new_idx ) = _i;
      }

      // Get the count since we are using exlusive prefix sum
      if ( _i == ( n - 2 ) )
        m_count() = m_should_split( _i ) + m_expanded_indices( _i );
    } );

    Kokkos::deep_copy( m_host_count, m_count );
    Kokkos::fence();

    _out_num_splits = m_host_count();
  }
}

#endif  // INC_BVH_CLUSTER_HPP
