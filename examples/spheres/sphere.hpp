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

#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <bvh/math/vec.hpp>

class Sphere {
public:
  KOKKOS_INLINE_FUNCTION Sphere() = default;

  KOKKOS_INLINE_FUNCTION ~Sphere() = default;

  Sphere(const Sphere &) = default;

  Sphere(Sphere &&) noexcept = default;

  KOKKOS_INLINE_FUNCTION Sphere &
  operator=(const Sphere &) = default;

  KOKKOS_INLINE_FUNCTION Sphere &
  operator=(Sphere &&) noexcept = default;

  void
  set_global_id(const int &global_id) {
    global_id_ = global_id;
  }

  void
  set_radius(const double &radius) {
    radius_ = radius;
  }

  void
  set_position(const bvh::m::vec3d &position) {
    position_ = position;
  }

  void
  update_position(const double &dt, const bvh::m::vec3d &velocity) {
    position_ += dt * velocity;
  }

  const int &
  global_id() const {
    return global_id_;
  }

  const double &
  radius() const {
    return radius_;
  }

  const bvh::m::vec3d &
  position() const {
    return position_;
  }

private:
  int global_id_;
  double radius_;
  bvh::m::vec3d position_;
};

#endif
