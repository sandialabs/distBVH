#ifndef INC_BVH_CLUSTER_HPP
#define INC_BVH_CLUSTER_HPP

#include "../util/span.hpp"
#include "../vt/print.hpp"
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
                     std::size_t _cluster_count );

    std::size_t size() const noexcept { return m_size; }

  private:

    using morton_type = std::uint32_t;

    std::size_t m_size = 0;
    single_view< bvh::bphase_kdop > m_bounds;
    view< morton_type * > m_hashes;
    radix_sorter< morton_type, index_type > m_sorter;
    single_view< std::size_t > m_count;
    single_host_view< std::size_t > m_host_count;
    view< index_type * > m_depths_indices;
    view< unsigned * > m_depths;
    view< index_type * > m_reindex;
    view< index_type * > m_initial_splits;
    host_view< unsigned * > m_host_depths;
    radix_sorter< unsigned, index_type > m_depth_sorter;
  };

  inline morton_cluster::morton_cluster( std::size_t _n )
    : m_size( _n ),
      m_bounds( "morton_cluster_bounds" ),
      m_hashes( "morton_hashes", _n ),
      m_sorter( _n ),
      m_count( "morton_cluster_split_count" ),
      m_host_count( "morton_cluster_split_count_host" ),
      m_depths_indices( "morton_cluster_split_indices", _n - 1 ),
      m_depths( "morton_cluster_depths", _n - 1 ),
      m_reindex( "morton_cluster_reindex", _n - 1 ),
      m_initial_splits( "morton_cluster_reindex", _n - 1 ),
      m_host_depths( Kokkos::create_mirror_view( m_depths ) ),
      m_depth_sorter( _n - 1 )
  {

  }

  template< typename Element >
  void
  morton_cluster::operator()( view< const Element * > _elements,
                            view< index_type * > _indices,
                            view< index_type * > _splits,
                            std::size_t _cluster_count )
  {
    assert( _cluster_count < m_size );

    assert( m_hashes.extent( 0 ) == m_size );
    assert( m_hashes.extent( 0 ) == _elements.extent( 0 ) );
    assert( m_hashes.extent( 0 ) == _indices.extent( 0 ) );
    assert( m_hashes.extent( 0 ) - 1 == _splits.extent( 0 ) );

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
    Kokkos::parallel_for( n - 1, [this, _splits] KOKKOS_FUNCTION( int _i ) {
      auto mask = m_hashes( _i ) ^ m_hashes( _i + 1 );
      m_depths_indices( _i ) = _i;

      // Kokkos doesn't currently offer clz but they do offer int_log2 but in the Impl namespace ;_;
      constexpr int shift = sizeof(unsigned) * CHAR_BIT - 1;
      m_depths( _i ) = ( mask != 0 ) ? shift - Kokkos::Impl::int_log2( mask ) : static_cast< unsigned >( -1 );  // this could break at any version of Kokkos...
    } );

    // Sort in order of increasing depth
    m_depth_sorter( m_depths, m_depths_indices );

    // We want the first cluster_count indices -- but they have to be in sorted index order
    // Mark these with 1, then we can execute an exclusive scan to re-index
    Kokkos::parallel_for( n - 1, [this, _cluster_count] KOKKOS_FUNCTION( int _i ) {
      m_reindex( m_depths_indices( _i ) ) = ( _i < _cluster_count ) ? 1 : 0;
    } );

    prefix_sum( m_reindex );
    Kokkos::parallel_for( n - 1, [this, _splits, _cluster_count] KOKKOS_FUNCTION( int _i ) {
      if ( _i < _cluster_count )
      {
        auto new_idx = m_reindex( m_depths_indices( _i ) );
        _splits( new_idx ) = m_depths_indices( _i );
      }
    } );

#ifdef BVH_ENABLE_CLUSTERING_PERFORMANCE_WARNING
    Kokkos::deep_copy( m_host_depths, m_depths );

    for ( std::size_t i = 0; i < _cluster_count; ++i )
    {
      if ( m_host_depths( i ) > 31 )  // no diff found
        ::bvh::vt::warn( "identical hash encountered at split index {} during clustering, this could lead to performance degradation\n", i );
    }
#endif // BVH_ENABLE_CLUSTERING_PERFORMANCE_WARNING
  }
}

#endif  // INC_BVH_CLUSTER_HPP
