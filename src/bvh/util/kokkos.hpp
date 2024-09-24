/*
 * distBVH 1.0
 *
 * Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef INC_BVH_UTIL_KOKKOS_HPP
#define INC_BVH_UTIL_KOKKOS_HPP

#include <ostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <limits>
#include <algorithm>
#include <type_traits>

namespace bvh
{
  using default_execution_space = Kokkos::DefaultExecutionSpace;
#ifdef BVH_ENABLE_CUDA
  using primary_execution_space = Kokkos::Cuda;
#elif defined(BVH_ENABLE_OPENMP)
  using primary_execution_space = Kokkos::OpenMP;
#else
  using primary_execution_space = default_execution_space;
#endif

  using host_execution_space = Kokkos::DefaultHostExecutionSpace;

  using host_memory = Kokkos::HostSpace;

  template< typename T >
  using view = Kokkos::View< T, primary_execution_space::array_layout, primary_execution_space >;

  template< typename T >
  using single_view = view< T >;

  template< typename T >
  using host_view = Kokkos::View< T, primary_execution_space::array_layout, host_execution_space >;

  template< typename T >
  using single_host_view = host_view< T >;

  template< typename T >
  using unmanaged_view = Kokkos::View< T, primary_execution_space::array_layout, primary_execution_space,
                                       Kokkos::MemoryTraits< Kokkos::Unmanaged > >;
}

#endif  // INC_BVH_UTIL_KOKKOS_HPP
