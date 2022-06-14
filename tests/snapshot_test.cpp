#include "TestCommon.hpp"
#include <bvh/snapshot.hpp>

TEST_CASE("snapshot", "[snapshot][kokkos]")
{
  std::default_random_engine eng( 0 );
  auto bound_min = bvh::m::vec3d{ -100.0, -100.0, -100.0 };
  auto bound_max = bvh::m::vec3d{ 100.0, 100.0, 100.0 };
  auto kdops = generate_random_kdops( eng, 1000,
                                      bound_min,
                                      bound_max,
                                      5.0 );

  auto snapshots = snapshots_from_kdops( kdops );

  SECTION("bounding box")
  {
    bvh::single_view< bvh::bphase_kdop > bounds( "bounds" );
    bvh::compute_bounds( snapshots, bounds );
    auto host_bounds = Kokkos::create_mirror_view_and_copy( bvh::host_execution_space{}, bounds );
    REQUIRE( host_bounds( 0 ).cardinal_min().x() >= bound_min.x() );
    REQUIRE( host_bounds( 0 ).cardinal_min().y() >= bound_min.y() );
    REQUIRE( host_bounds( 0 ).cardinal_min().z() >= bound_min.z() );
    REQUIRE( host_bounds( 0 ).cardinal_max().x() <= bound_max.x() );
    REQUIRE( host_bounds( 0 ).cardinal_max().y() <= bound_max.y() );
    REQUIRE( host_bounds( 0 ).cardinal_max().z() <= bound_max.z() );
    fmt::print( "min: {}", host_bounds( 0 ).cardinal_min() );
  }
}
