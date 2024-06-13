/*
 * distBVH 1.0
 *
 * Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "granular_object.hpp"

#include <Kokkos_Core.hpp>

void
GranularObject::generate_spheres(const int &global_id_offset,
                                 const bvh::m::vec3d &shift) {
  const std::string view_name = object_name_ + "_spheres";
  const int num_sphere = nx_ * ny_ * nz_;

  spheres_ = bvh::view<Sphere *>(view_name, num_sphere);

  const double radius = 0.5;

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<3> >{{0, 0, 0}, {nx_, ny_, nz_}},
      KOKKOS_LAMBDA(int ix, int iy, int iz) {
        const auto &local_id = get_local_id(ix, iy, iz);
        const auto &global_id = global_id_offset + local_id;
        const auto &position =
            bvh::m::vec3d(double(ix), double(iy), double(iz)) + shift;

        auto &sphere = spheres_(local_id);

        sphere.set_global_id(global_id);
        sphere.set_radius(radius);
        sphere.set_position(position);
      });
}

void
GranularObject::compute_collision_sphere_indices() {
  const std::string view_name = object_name_ + "_collision_sphere_indices";
  int num_collision_sphere = 0;
  bool all_spheres_on_boundary = false;

  if (nx_ <= 2 || ny_ <= 2 || nz_ <= 2) {
    num_collision_sphere = nx_ * ny_ * nz_;
    all_spheres_on_boundary = true;
  } else {
    num_collision_sphere =
        2 * (nx_ * ny_ + ny_ * nz_ + nz_ * nx_) - 4 * (nx_ + ny_ + nz_) + 8;
    all_spheres_on_boundary = false;
  }

  collision_sphere_indices_ = bvh::view<int *>(view_name, num_collision_sphere);

  if (all_spheres_on_boundary) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<int>(0, num_collision_sphere),
        KOKKOS_LAMBDA(int i) { collision_sphere_indices_(i) = i; });

    return;
  }

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2> >{{0, 0}, {nx_, ny_}},
      KOKKOS_LAMBDA(int ix, int iy) {
        int local_id = get_local_id(ix, iy, 0);
        int collision_local_id = ix + iy * nx_;  // bottom layer
        collision_sphere_indices_(collision_local_id) = local_id;

        local_id = get_local_id(ix, iy, nz_ - 1);
        collision_local_id =
            nx_ * ny_ + (nz_ - 2) * (2 * nx_ + 2 * (ny_ - 2)) + ix + iy * nx_;
        collision_sphere_indices_(collision_local_id) = local_id;
      });

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2> >{{0, 1}, {nx_, nz_ - 1}},
      KOKKOS_LAMBDA(int ix, int iz) {
        int local_id = get_local_id(ix, 0, iz);
        int collision_local_id =
            nx_ * ny_ + (iz - 1) * (2 * nx_ + 2 * (ny_ - 2)) + ix;
        collision_sphere_indices_(collision_local_id) = local_id;

        local_id = get_local_id(ix, ny_ - 1, iz);
        collision_local_id = nx_ * ny_ + (iz - 1) * (2 * nx_ + 2 * (ny_ - 2)) +
                             nx_ + 2 * (ny_ - 2) + ix;
        collision_sphere_indices_(collision_local_id) = local_id;
      });

  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2> >{{1, 1}, {ny_ - 1, nz_ - 1}},
      KOKKOS_LAMBDA(int iy, int iz) {
        int local_id = get_local_id(0, iy, iz);
        int collision_local_id = nx_ * ny_ +
                                 (iz - 1) * (2 * nx_ + 2 * (ny_ - 2)) + nx_ +
                                 2 * (iy - 1);
        collision_sphere_indices_(collision_local_id) = local_id;

        local_id = get_local_id(nx_ - 1, iy, iz);
        collision_local_id = nx_ * ny_ + (iz - 1) * (2 * nx_ + 2 * (ny_ - 2)) +
                             nx_ + 2 * (iy - 1) + 1;
        collision_sphere_indices_(collision_local_id) = local_id;
      });
}

void
GranularObject::allocate_collision_spheres() {
  const std::string view_name = object_name_ + "_collision_spheres";
  collision_spheres_ =
      bvh::view<CollisionSphere *>(view_name, collision_sphere_indices_.size());
}

void
GranularObject::copy_collision_spheres() {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, collision_sphere_indices_.size()),
      KOKKOS_LAMBDA(int i) {
        collision_spheres_(i) = spheres_(collision_sphere_indices_(i));
      });
}

void
GranularObject::step(const double &dt) {
  const bvh::m::vec3d gravity(0.0, 0.0, -10.0);

  velocity_ += dt * gravity;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, spheres_.size()),
      KOKKOS_LAMBDA(int i) { spheres_(i).update_position(dt, velocity_); });

  copy_collision_spheres();
}
