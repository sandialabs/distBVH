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
#ifndef BVH_PERF_INSTRUMENT_HPP
#define BVH_PERF_INSTRUMENT_HPP

#include <ostream>

namespace bvh
{
  namespace perf
  {
    template< typename T >
    class instrument
    {
    public:
      
      static int num_allocations;
      static int num_copies;
      static int num_moves;
      
      static std::ostream &summarize_perf( std::ostream &_os )
      {
        _os << "Num allocations: " << num_allocations << ", Num copies: " << num_copies
            << ", Num moves: " << num_moves << '\n';
        return _os;
      }
      
      static void reset_copy_counter()
      {
        num_copies = 0;
      }
      
      static void reset_move_counter()
      {
        num_moves = 0;
      }
      
      instrument()
      {
        ++num_allocations;
      }
      
      instrument( const instrument & )
      {
        ++num_allocations;
        ++num_copies;
      }
      
      instrument &operator=( const instrument & )
      {
        ++num_copies;
        return *this;
      }
      
      instrument( instrument && )
      {
        ++num_allocations;
        ++num_moves;
      }
      
      instrument &operator=( instrument && )
      {
        ++num_moves;
        return *this;
      }
      
    protected:
      
      ~instrument()
      {
        --num_allocations;
      }
    };
    
    template< typename T >
    int instrument< T >::num_allocations = 0;
  
    template< typename T >
    int instrument< T >::num_copies = 0;
  
    template< typename T >
    int instrument< T >::num_moves = 0;
  }
}

#endif  // BVH_PERF_INSTRUMENT_HPP
