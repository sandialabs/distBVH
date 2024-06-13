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

#include <vt/vt.h>

#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>

struct Config {
  Config(int argc, char **argv);

  int num_sphere_x = 5;
  int num_sphere_y = 5;
  int num_sphere_z = 5;
};

Config::Config(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--num_sphere_x") {
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
    } else {
      std::cerr << "ignoring unknown option \"" << arg << "\"\n";
    }
  }
}

void
Main(const Config &config) {
  const int &num_sphere_x = config.num_sphere_x;
  const int &num_sphere_y = config.num_sphere_y;
  const int &num_sphere_z = config.num_sphere_z;

  GranularObject object("object", num_sphere_x, num_sphere_y, num_sphere_z);
}

int
main(int argc, char **argv) {
  int return_code = 0;
  std::string error_message = "";

  vt::initialize(argc, argv);
  Kokkos::initialize(argc, argv);

  try {
    Config config(argc, argv);
    Main(config);
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
