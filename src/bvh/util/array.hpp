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
#ifndef INC_BVH_UTIL_ARRAY_HPP
#define INC_BVH_UTIL_ARRAY_HPP

#include "attributes.hpp"

#ifndef BVH_ENABLE_CUDA
#include <array>
#endif

namespace bvh
{
#ifndef BVH_ENABLE_CUDA
  template< typename T, std::size_t N >
  using array = std::array< T, N >;
#else
  template< typename T, std::size_t N >
  struct array
  {
    using pointer = T *;
    using iterator = pointer;
    using const_iterator = const T *;
    
    constexpr BVH_INLINE T operator[]( std::size_t _n ) const noexcept
    {
      return m_data[_n];
    }
    
    BVH_INLINE T &operator[]( std::size_t _n )
    {
      return m_data[_n];
    }
    
    iterator begin() { return &m_data[0]; }
    const_iterator begin() const { return &m_data[0]; }
    const_iterator cbegin() const { return &m_data[0]; }
  
    iterator end() { return &m_data[0] + N; }
    const_iterator end() const { return &m_data[0] + N; }
    const_iterator cend() const { return &m_data[0] + N; }
    
    T m_data[N];
  };
#endif
}

#endif  // INC_BVH_UTIL_ARRAY_HPP
