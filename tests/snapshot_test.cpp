#include "TestCommon.hpp"
#include <bvh/contact_entity.hpp>
#include <bvh/types.hpp>
#include <bvh/snapshot.hpp>

TEST_CASE("snapshot", "[snapshot][kokkos]")
{
  std::default_random_engine eng( 0 );
  auto bound_min = bvh::m::vec3d{ -100.0, -100.0, -100.0 };
  auto bound_max = bvh::m::vec3d{ 100.0, 100.0, 100.0 };

  SECTION("bounding box")
  {
    auto kdops = generate_random_kdops( eng, 1000,
                                        bound_min,
                                        bound_max,
                                        5.0 );

    auto snapshots = snapshots_from_kdops( kdops );
    bvh::single_view< bvh::bphase_kdop > bounds( "bounds" );
    bvh::compute_bounds( snapshots, bounds );
    auto host_bounds = Kokkos::create_mirror_view_and_copy( bvh::host_execution_space{}, bounds );
    REQUIRE( host_bounds().cardinal_min().x() >= bound_min.x() );
    REQUIRE( host_bounds().cardinal_min().y() >= bound_min.y() );
    REQUIRE( host_bounds().cardinal_min().z() >= bound_min.z() );
    REQUIRE( host_bounds().cardinal_max().x() <= bound_max.x() );
    REQUIRE( host_bounds().cardinal_max().y() <= bound_max.y() );
    REQUIRE( host_bounds().cardinal_max().z() <= bound_max.z() );
  }

  SECTION("morton hashing")
  {
    const auto half_cell = ( bound_max - bound_min ) / 2048.0; // 10 bit max value divided by 2
    const auto radius = std::min( std::min( half_cell[0], half_cell[1] ), half_cell[2] );
    auto kdops = generate_kdop_grid( 8, bound_min + half_cell, bound_max + half_cell, radius );
    auto snapshots = snapshots_from_kdops( kdops );

    bvh::single_view< bvh::min_inv_diag_bounds > bounds( "min_bound" );
    Kokkos::deep_copy( bounds, bvh::min_inv_diag_bounds{
      bvh::m::vec3d{ -100.0, -100.0, -100.0 },
      bvh::m::vec3d{ 0.005, 0.005, 0.005 } } );

    bvh::view< bvh::morton32_t * > hashes( "hashes", snapshots.extent( 0 ) );
    bvh::morton( snapshots, bounds, hashes );

    auto host_hashes = Kokkos::create_mirror_view_and_copy( bvh::host_execution_space{}, hashes );

    // We are dividing into 8 portions along each axis so each dimension will be
    // quantized to max value (0x400) >> 3 so 0x80
    std::uint32_t incr = 0x80;
    auto q = bvh::m::vec3< std::uint32_t >{};

    for ( std::size_t z = 0; z < 8; ++z )
    {
      for ( std::size_t y = 0; y < 8; ++y )
      {
        for ( std::size_t x = 0; x < 8; ++x )
        {
          const auto idx = z * 64 + y * 8 + x;

          // Regular morton hash already tested...
          auto hash = bvh::morton( q.x(), q.y(), q.z() );
          REQUIRE( hash == host_hashes( idx ) );
          q.x() += incr;
        }
        q.y() += incr;
        q.x() = 0;
      }
      q.z() += incr;
      q.y() = 0;
    }
  }
}
