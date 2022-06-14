#include "cluster.hpp"
#include "../hash.hpp"

namespace bvh
{

  KOKKOS_FUNCTION
  void
  cluster_permutations( view< entity_snapshot > _elements,
                        view< std::size_t > _indices,
                        view< std::size_t > _splits,
                        int _d )
  {
    // 1. Compute morton codes
    // 2. Sort the indices according to spatial hash
    // 3. Compute split points from d-bit change


  }
}
