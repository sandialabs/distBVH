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
#ifndef INC_BVH_VT_HELPERS_HPP
#define INC_BVH_VT_HELPERS_HPP

#include <initializer_list>
#include <vt/transport.h>

namespace bvh
{
  namespace vt
  {
    template< typename T >
    struct reducable_vector
    {
      using opt = ::vt::collective::PlusOp< reducable_vector >;

      reducable_vector() = default;
      
      reducable_vector( std::initializer_list< T > _init )
        : vec( _init )
      {
        
      }
      
      reducable_vector &operator+=( const reducable_vector &_other )
      {
        vec.insert( vec.end(), _other.vec.begin(), _other.vec.end() );
      
        return *this;
      }
    
      friend reducable_vector operator+( reducable_vector _lhs, const reducable_vector &_rhs )
      {
        return _lhs += _rhs;
      }
    
      template< typename Serializer >
      void serialize( Serializer &_s )
      {
        _s | vec;
      }
    
      dynarray< T > vec;
    };
  }
}

#endif  // INC_BVH_VT_HELPERS_HPP
