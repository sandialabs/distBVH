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

#include <vt/vt.h>

#include <Kokkos_Core.hpp>
#include <bvh/collision_object.hpp>
#include <bvh/collision_world.hpp>
#include <bvh/math/vec.hpp>
#include <iostream>
#include <utility>
#include <vector>

#include "collision_sphere.hpp"
#include "granular_object.hpp"

using narrowphase_result = std::pair<int, int>;

struct Config {
  Config(int argc, char **argv);

  std::string
  help_message() {
    // clang-format off
    return std::string(
        "collision between two block objects consisting of spheres\n"
        "\n"
        "setup:\n"
        "  - both objects consist of [np * nx, ny, nz] number of spheres placed in a\n"
        "    primitive cubic packing arrangement\n"
        "  - the first object is fixed; the plane through the centers of its top-level\n"
        "    spheres coincides with the x-y coordinate plane\n"
        "  - the second object is free; its initial position coincides with the position\n"
        "    of the second object shifted upwards\n"
        "  - the separation of the plane through the centers of the top-level spheres of\n"
        "    the first object, and the plane through the coenters of the bottom level\n"
        "    spheres of the second is user specified\n"
        "  - both objects are split among MPI ranks along the x direction; splitting is\n"
        "    even for the second object"
        "  - the x-directional splitting of the first object among MPI ranks could be\n"
        "    uneven based on a parameter dnx; essentially, if np >= 2,\n"
        "    - rank 0 has (nx - dnx) along x direction\n"
        "    - rank k has (nx - (-1)^k * 2 * dnx) along x direction for 1 <= k <= np - 2\n"
        "    - rank (np - 1) has (nx - (-1)^(np - 1) dnx) spheres along x direction\n"
        "  - the second object falls under gravity; after collision with the first\n"
        "    object, its velocity is reversed in the z direction\n"
        "  - time integration is performed using forward Euler scheme with fixed step\n"
        "    size h for a specified number k of step\n"
        "\n"
        "usage:"
        "  ") + std::string(program_name_) + std::string(" \\\n"
        "      [vt-options] \\\n"
        "      [kokkos-options] \\\n"
        "      -- \\\n"
        "      [--num_sphere_x nx  (default: 5)   ] \\\n"
        "      [--num_sphere_y ny  (default: 5)   ] \\\n"
        "      [--num_sphere_z nz  (default: 5)   ] \\\n"
        "      [--imbalance_x  dnx (default: 2)   ] \\\n"
        "      [--step_size    h   (default: 0.01)] \\\n"
        "      [--num_step     k   (default: 100) ] \\\n"
        "\n");
    // clang-format on
  }

  const std::string program_name_;

  bool emit_help_message = false;

  int num_sphere_x = 5;
  int num_sphere_y = 5;
  int num_sphere_z = 5;
  int imbalance_x = 2;
  double object_2_height = 2.0;
  double step_size = 1.0e-02;
  int num_step = 100;
};

Config::Config(int argc, char **argv) : program_name_(argv[0]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      emit_help_message = true;
    } else if (arg == "--num_sphere_x") {
      num_sphere_x = std::stoi(argv[++i]);

      if (num_sphere_x < 1) {
        throw std::invalid_argument(
            "number of spheres along x direction must be positive");
      }
    } else if (arg == "--num_sphere_y") {
      num_sphere_y = std::stoi(argv[++i]);

      if (num_sphere_y < 1) {
        throw std::invalid_argument(
            "number of spheres along y direction must be positive");
      }
    } else if (arg == "--num_sphere_z") {
      num_sphere_z = std::stoi(argv[++i]);

      if (num_sphere_z < 1) {
        throw std::invalid_argument(
            "number of spheres along z direction must be positive");
      }
    } else if (arg == "--imbalance_x") {
      imbalance_x = std::stoi(argv[++i]);

      if (imbalance_x < 0) {
        throw std::invalid_argument(
            "imbalance along x direction must be non-negative");
      } else if (2 * imbalance_x > num_sphere_x) {
        throw std::invalid_argument(
            "imbalance must be no larger than half the number of spheres along "
            "x direction");
      }
    } else if (arg == "--object_2_height") {
      object_2_height = std::stof(argv[++i]);

      if (object_2_height < 1.0) {
        throw std::invalid_argument("height of object 2 must be at least 1.0");
      }
    } else if (arg == "--step_size") {
      step_size = std::stof(argv[++i]);

      if (step_size <= 0.0) {
        throw std::invalid_argument(" step size must be positive");
      }
    } else if (arg == "--num_step") {
      num_step = std::stoi(argv[++i]);

      if (num_step < 1) {
        throw std::invalid_argument("number of steps must be positive");
      }
    } else {
      std::cerr << "ignoring unknown option \"" << arg << "\"\n";
    }
  }

  std::cerr << std::flush;
}

void
Main(const Config &config) {
  const int num_sphere_x = config.num_sphere_x;
  const int num_sphere_y = config.num_sphere_y;
  const int num_sphere_z = config.num_sphere_z;
  const double object_2_height = config.object_2_height;
  const double step_size = config.step_size;
  const int num_step = config.num_step;

  const int imbalance_x = config.imbalance_x;

  const auto n_nodes = vt::theContext()->getNumNodes();
  const auto node_id = vt::theContext()->getNode();

  int num_sphere_x_object_1 = num_sphere_x;
  int gid_offset_x_object_1 = node_id * num_sphere_x;

  if (n_nodes > 1) {
    if (node_id % 2 == 0) {
      num_sphere_x_object_1 -= (node_id == 0 || node_id == n_nodes - 1)
                                   ? imbalance_x
                                   : 2 * imbalance_x;
      gid_offset_x_object_1 += node_id > 0 ? imbalance_x : 0;
    } else {
      num_sphere_x_object_1 += (node_id == 0 || node_id == n_nodes - 1)
                                   ? imbalance_x
                                   : 2 * imbalance_x;
      gid_offset_x_object_1 -= node_id > 0 ? imbalance_x : 0;
    }
  }

  const int num_sphere_x_object_2 = num_sphere_x;
  const int gid_offset_x_object_2 = node_id * num_sphere_x;

  GranularObject object_1(
      "sheet_1",
      num_sphere_x_object_1,
      num_sphere_y,
      num_sphere_z,
      gid_offset_x_object_1 * num_sphere_y * num_sphere_z,
      bvh::m::vec3d(
          double(gid_offset_x_object_1), 0.0, 1.0 - double(num_sphere_z)));
  GranularObject object_2(
      "sheet_2",
      num_sphere_x_object_2,
      num_sphere_y,
      num_sphere_z,
      (gid_offset_x_object_2 + n_nodes * num_sphere_x) * num_sphere_y *
          num_sphere_z,
      bvh::m::vec3d(double(gid_offset_x_object_2), 0.0, object_2_height));

  bvh::collision_world world(2);
  bvh::collision_object &collision_object_1 = world.create_collision_object();
  bvh::collision_object &collision_object_2 = world.create_collision_object();

  std::ofstream os;
  if (node_id == 0) {
    os.open("collide_two_blocks.csv");
    os << "time,obj_2_pos_z,obj_2_vel_z,num_collision\n";
  }

  vt::runInEpochCollective("time_iterations", [&]() {
    for (int step_index = 0; step_index < num_step; ++step_index) {
      world.start_iteration();

      collision_object_1.set_entity_data(object_1.collision_spheres(),
                                         bvh::split_algorithm::geom_axis);
      collision_object_1.init_broadphase();

      collision_object_2.set_entity_data(object_2.collision_spheres(),
                                         bvh::split_algorithm::geom_axis);
      collision_object_2.init_broadphase();

      world.set_narrowphase_functor<CollisionSphere>(
          [node_id](const bvh::broadphase_collision<CollisionSphere> &set_1,
                    const bvh::broadphase_collision<CollisionSphere> &set_2) {
            auto result = bvh::narrowphase_result_pair();

            result.a = bvh::narrowphase_result(sizeof(narrowphase_result));
            result.b = bvh::narrowphase_result(sizeof(narrowphase_result));

            auto &result_a = static_cast<
                bvh::typed_narrowphase_result<narrowphase_result> &>(result.a);
            auto &result_b = static_cast<
                bvh::typed_narrowphase_result<narrowphase_result> &>(result.b);

            for (auto &&sphere_1 : set_1.elements) {
              for (auto &&sphere_2 : set_2.elements) {
                if (sphere_1.is_colliding_with(sphere_2)) {
                  result_a.emplace_back(std::make_pair(sphere_1.global_id(),
                                                       sphere_2.global_id()));
                }
              }
            }

            return result;
          });

      collision_object_1.broadphase(collision_object_2);

      std::vector<narrowphase_result> results;
      collision_object_1.for_each_result<narrowphase_result>(
          [node_id, &results](const narrowphase_result &res) {
            results.emplace_back(res);
          });

      world.finish_iteration();

      if (results.size() != 0) {
        const auto velocity = object_2.velocity();
        object_2.set_velocity(
            bvh::m::vec3d(velocity.x(), velocity.y(), -velocity.z()));
      }

      object_2.step(step_size);

      if (node_id == 0) {
        os << fmt::format("{:.3e},{:.3e},{:.3e},{}\n",
                          (step_index + 1) * step_size,
                          object_2.position().z(),
                          object_2.velocity().z(),
                          results.size());
      }
    }
  });
}

int
main(int argc, char **argv) {
  int return_code = 0;
  std::string error_message = "";

  vt::initialize(argc, argv);
  Kokkos::initialize(argc, argv);

  try {
    Config config(argc, argv);
    if (config.emit_help_message) {
      std::cout << config.help_message() << std::endl;
    } else {
      Main(config);
    }
    return_code = 0;
  } catch (std::exception &e) {
    error_message = e.what();
    return_code = 1;
  } catch (...) {
    error_message = "encountered unknown exception";
    return_code = 1;
  }

  Kokkos::finalize();
  vt::finalize();

  if (return_code) {
    std::cerr << error_message << std::endl;
  }

  return return_code;
}
