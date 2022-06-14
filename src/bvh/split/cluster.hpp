#ifndef INC_BVH_CLUSTER_HPP
#define INC_BVH_CLUSTER_HPP

#include "../util/span.hpp"
#include "../util/kokkos.hpp"
#include "../snapshot.hpp"
#include "split.hpp"

namespace bvh
{
  KOKKOS_FUNCTION
  void
  cluster_permutations( view< const entity_snapshot * > _elements,
                        view< std::size_t * > _indices,
                        view< std::size_t * > _splits,
                        int _d );
}

#endif  // INC_BVH_CLUSTER_HPP
