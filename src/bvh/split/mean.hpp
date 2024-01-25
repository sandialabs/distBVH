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
#ifndef INC_BVH_SPLIT_MEAN_HPP
#define INC_BVH_SPLIT_MEAN_HPP

#include <numeric>
#include "../range.hpp"
#include "../traits.hpp"
#include <iterator>

#include <Kokkos_Core.hpp>

namespace bvh
{
  namespace split
  {
    struct mean
    {
      template< typename InputIterator >
      static auto split_point( range <InputIterator> _elements, int _axis )
      {
        using traits_type = element_traits< typename std::iterator_traits< InputIterator >::value_type >;
        using kdop_type = typename traits_type::kdop_type;
        using arithmetic_type = typename kdop_type::arithmetic_type;

        auto centroids_range = transform_range( _elements, traits_type::get_centroid );

        auto projected_range = transform_range( centroids_range, [_axis]( auto _c ) {
          return kdop_type::project( _c, _axis );
        } );

        auto sum = std::accumulate( projected_range.begin(), projected_range.end(), arithmetic_type{ 0 } );

        return sum / static_cast< arithmetic_type >( projected_range.size());
      }

      template< typename Input >
      static auto split_point( span < Input > _elements, int _axis )
      {
        using traits_type = element_traits< Input >;
        using kdop_type = typename traits_type::kdop_type;
        using arithmetic_type = typename kdop_type::arithmetic_type;
        arithmetic_type sum = 0;
        Kokkos::parallel_reduce("LoopMean", _elements.size(), KOKKOS_LAMBDA (const int& i, arithmetic_type& lsum ) {
             const auto &c = _elements[i].centroid();
             lsum += kdop_type::project( c, _axis );
            }, sum);
        return sum / static_cast< arithmetic_type >( _elements.size() );
      }

    };
  }
}

#endif  // INC_BVH_SPLIT_MEAN_HPP
