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

#ifndef GRANULAR_OBJECT_HPP
#define GRANULAR_OBJECT_HPP

#include <bvh/math/vec.hpp>
#include <cstddef>
#include <string>

#include "collision_sphere.hpp"
#include "sphere.hpp"

class GranularObject {
public:
  GranularObject(const std::string &object_name,
                 const int &nx,
                 const int &ny,
                 const int &nz,
                 const int &global_id_offset = 0,
                 const bvh::m::vec3d &shift = bvh::m::vec3d::zeros())
      : object_name_(object_name),
        nx_(nx),
        ny_(ny),
        nz_(nz),
        spheres_(),
        collision_sphere_indices_(),
        collision_spheres_(),
        velocity_(bvh::m::vec3d::zeros()) {
    generate_spheres(global_id_offset, shift);
    compute_collision_sphere_indices();
    allocate_collision_spheres();
    copy_collision_spheres();
  }

  const bvh::m::vec3d &
  position() const {
    return spheres_(0).position();
  }

  const bvh::m::vec3d &
  velocity() const {
    return velocity_;
  }

  void
  set_velocity(const bvh::m::vec3d &velocity) {
    velocity_ = velocity;
  }

  void
  step(const double &dt);

  bvh::view<CollisionSphere *>
  collision_spheres() const {
    return collision_spheres_;
  }

private:
  int
  get_local_id(const int &ix, const int &iy, const int &iz) {
    return ix + iy * nx_ + iz * nx_ * ny_;
  }

  void
  generate_spheres(const int &global_id_offset, const bvh::m::vec3d &offset);

  void
  compute_collision_sphere_indices();

  void
  allocate_collision_spheres();

  void
  copy_collision_spheres();

private:
  const std::string object_name_;
  const int nx_;
  const int ny_;
  const int nz_;
  bvh::view<Sphere *> spheres_;
  Kokkos::View<int *> collision_sphere_indices_;
  bvh::view<CollisionSphere *> collision_spheres_;
  bvh::m::vec3d velocity_;
};

#endif
