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
#ifndef INC_BVH_REGISTRY_HPP
#define INC_BVH_REGISTRY_HPP

#include <vector>

namespace bvh
{
  namespace detail
  {
    template< typename T >
    struct registry
    {
      static std::vector< T > &reg()
      {
        static std::vector< T > sreg;
        return sreg;
      }

      template< T Val >
      static std::size_t register_value()
      {
        std::size_t ret = reg().size();
        reg().push_back( Val );

        return ret;
      }

      template< T Val >
      struct element
      {
        static const std::size_t id;
      };
    };

    template< typename T >
    template< T Val >
    const std::size_t
    registry< T >::element< Val >::id = register_value< Val >();
  }

  template< typename T, T Val >
  std::size_t register_value()
  {
    return detail::registry< T >::template element< Val >::id;
  }

  template< typename T >
  T retrieve_value( std::size_t _idx )
  {
    auto ret = detail::registry< T >::reg().at( _idx );
    return ret;
  }
}

#endif  // INC_BVH_REGISTRY_HPP
