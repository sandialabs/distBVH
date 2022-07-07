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

    explicit morton_cluster( std::size_t _n );

    void operator()( view< const entity_snapshot * > _elements,
                     view< index_type * > _indices,
                     view< index_type * > _splits,
                     int _d, std::size_t &_out_num_splits );

  private:

    using morton_type = std::uint32_t;

    single_view< bvh::bphase_kdop > m_bounds;
    view< morton_type * > m_hashes;
    radix_sorter< morton_type, index_type > m_sorter;
    single_view< std::size_t > m_count;
    single_host_view< std::size_t > m_host_count;
  };
}

#endif  // INC_BVH_CLUSTER_HPP
