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
#ifndef INC_BVH_RANGE_HPP
#define INC_BVH_RANGE_HPP

#include <iterator>
#include "iterators/transform_iterator.hpp"

namespace bvh
{
  /**
   *  Range of elements between two iterators.
   *
   *  Ranges themselves act like containers and can be iterated on using foreach loops.
   *
   *  \tparam Iterator  The type of iterator to be used for the range. This parameter is deduced from
   *                    the constructor arguments.
   */
  template< typename Iterator >
  struct range
  {
  public:
    
    using iterator = Iterator;
    
    using value_type = typename std::iterator_traits< Iterator >::value_type;
    
    /**
     *  Construct a range via a pair of iterators.
     *
     *  \param _begin   the beginning of a range
     *  \param _end     the end of a range
     */
    range( Iterator _begin, Iterator _end )
      : first( _begin ), second( _end )
    {}
    
    iterator begin() const noexcept { return first; }
    iterator end() const noexcept { return second; }
    
    /**
     *  The signed distance between the beginning and end.
     *
     *  \return the value of `std::distance` between beginning and end
     */
    std::ptrdiff_t distance() const { return std::distance( first, second ); }
    
    /**
     *  The unsigned distance between the beginning and end.
     *
     *  \return the unsigned value of `std::distance` between beginning and end. Invalid if
     *          end is before beginning in the iterator sequence.
     */
    std::size_t size() const { return static_cast< std::size_t >( distance() ); }
    
    Iterator first;
    Iterator second;
  };
  
  template< typename Iterator >
  range< Iterator > make_range( Iterator _begin, Iterator _end )
  {
    return range< Iterator >( _begin, _end );
  }
  
  template< typename Iterator, typename UnaryFunction >
  range< transform_iterator< UnaryFunction, Iterator > > transform_range( Iterator _begin, Iterator _end, UnaryFunction _f )
  {
    return make_range( make_transform_iterator( _begin, _f ), make_transform_iterator( _end, _f ) );
  }
  
  template< typename Iterator, typename UnaryFunction >
  auto transform_range( range< Iterator > _range, UnaryFunction &&_fun )
  {
    return transform_range( _range.begin(), _range.end(), std::forward< UnaryFunction >( _fun ) );
  }
}

#endif  // INC_BVH_RANGE_HPP
