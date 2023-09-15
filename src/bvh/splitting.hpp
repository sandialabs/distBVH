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
#ifndef INC_BVH_SPLITTING_HPP
#define INC_BVH_SPLITTING_HPP

#include <numeric>
#include "range.hpp"

namespace bvh
{
  /**
   *  Policy for bvh_tree construction where elements are split by a pivot point that is the mean
   *  of all the elements' centroids.
   *
   *  \tparam KDop  The type of k-DOP the tree is using.
   */
  template< typename KDop >
  struct mean_splitting
  {
    using kdop_type = KDop;
    using arithmetic_type = typename kdop_type::arithmetic_type;
    
    /**
     *  Compute the split point of a range of pointers-to-elements.
     *
     *  \tparam InputIterator   The iterator type of the range.
     *  \param _begin           The beginning of the pointer-to-element range.
     *  \param _end             The end of the pointer-to-element range.
     *  \param axis             The axis to split on.
     *  \return                 The split point as a signed distance along the axis.
     */
    template< typename InputIterator >
    static arithmetic_type split_point( InputIterator _begin, InputIterator _end, int axis )
    {
      auto trange = transform_range( _begin, _end,
        [axis]( auto c ) -> arithmetic_type { return kdop_type::project( c, axis ); } );
      
      auto sum = std::accumulate( trange.begin(), trange.end(), arithmetic_type( 0 ) );
      
      return sum / static_cast< arithmetic_type >( trange.size() );
    }
  };
}

#endif  // INC_BVH_SPLITTING_HPP
