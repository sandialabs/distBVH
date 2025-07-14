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

#ifndef COLLISION_SPHERE_HPP
#define COLLISION_SPHERE_HPP

#include <bvh/math/vec.hpp>
#include <bvh/types.hpp>
#include <cstddef>
#include <type_traits>

#include "sphere.hpp"

class CollisionSphere {
public:
  KOKKOS_INLINE_FUNCTION CollisionSphere() = default;

  KOKKOS_INLINE_FUNCTION ~CollisionSphere() = default;

  CollisionSphere(const CollisionSphere &) = default;

  CollisionSphere(CollisionSphere &&) noexcept = default;

  CollisionSphere(const Sphere &s)
      : global_id_(s.global_id()),
        radius_(s.radius()),
        centroid_(s.position()),
        bounds_(bvh::bphase_kdop::from_sphere(s.position(),
                                              (1.0 + buffer_) * s.radius())) {}

  KOKKOS_INLINE_FUNCTION CollisionSphere &
  operator=(const CollisionSphere &) = default;

  KOKKOS_INLINE_FUNCTION CollisionSphere &
  operator=(CollisionSphere &&) noexcept = default;

  KOKKOS_INLINE_FUNCTION CollisionSphere &
  operator=(const Sphere &s) {
    global_id_ = s.global_id();
    radius_ = s.radius();
    centroid_ = s.position();
    bounds_ = bvh::bphase_kdop::from_sphere(s.position(),
                                            (1.0 + buffer_) * s.radius());
    return *this;
  }

  std::size_t
  global_id() const {
    return std::size_t(global_id_);
  }

  bvh::m::vec3d
  centroid() const {
    return centroid_;
  }

  const bvh::bphase_kdop &
  kdop() const {
    return bounds_;
  }

  bool
  is_colliding_with(const CollisionSphere &other) const {
    return bvh::m::length(centroid_ - other.centroid_) <=
           (1 + buffer_) * (radius_ + other.radius_);
  }

private:
  int global_id_;
  double radius_;
  bvh::m::vec3d centroid_;
  bvh::bphase_kdop bounds_;

private:
  static constexpr double buffer_ = 0.01;
};

#endif
