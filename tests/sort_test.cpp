#include <catch2/catch.hpp>

#include <bvh/util/sort.hpp>
#include "TestCommon.hpp"
#include <array>
#include <random>
#include <chrono>

TEST_CASE("radix sort", "[utility][kokkos]")
{
  static constexpr std::size_t sz = 8;
  bvh::radix_sorter< std::uint32_t > sorter( sz );

  bvh::host_view< std::uint32_t * > nums{ "Numbers", sz };
  bvh::host_view< std::uint32_t * > indices{ "Indices", sz };

  gen_array( nums, 3, 7, 2, 1, 4, 9, 1, 3 );
  gen_array( indices, 0, 1, 2, 3, 4, 5, 6, 7 );

  auto dev_nums = Kokkos::create_mirror_view_and_copy( bvh::default_execution_space{}, nums );
  auto dev_indices = Kokkos::create_mirror_view_and_copy( bvh::default_execution_space{}, indices );

  sorter( dev_nums, dev_indices );

  Kokkos::deep_copy( nums, dev_nums );
  Kokkos::deep_copy( indices, dev_indices );

  test_array( nums, 1, 1, 2, 3, 3, 4, 7, 9 );
  test_array( indices, 3, 6, 2, 0, 7, 4, 1, 5 );
}
