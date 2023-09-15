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
#ifndef INC_BVH_VT_PRINT_HPP
#define INC_BVH_VT_PRINT_HPP

#include <vt/transport.h>

namespace bvh
{
  namespace vt
  {
#ifdef BVH_DEBUG_LEVEL
    constexpr int print_debug_level = BVH_DEBUG_LEVEL;
#else
    constexpr int print_debug_level = 0;
#endif
    
    // Use VT's format interop for preventing interleave of MPI prints
    namespace detail
    {
      template< bool Enable >
      struct print_impl
      {
        template< typename... Args >
        static void print( Args &&... )
        {
          // Do nothing
        }
      };
      
      template<>
      struct print_impl< true >
      {
        template< typename... Args >
        static void print( Args &&... _args )
        {
          ::fmt::print( std::forward< Args >( _args )... );
        }
      };
    }
    
    template< int DebugLevel, typename... Args >
    void print_level( Args &&... _args )
    {
      static constexpr bool enable_print = print_debug_level >= DebugLevel;
      detail::print_impl< enable_print >::print( std::forward< Args >( _args )... );
    }

    template< typename... Args >
    void error( Args &&... _args )
    {
      print_level< 0 >( stderr, std::forward< Args >( _args )... );
    }

    template< typename... Args >
    void print( Args &&... _args )
    {
      print_level< 0 >( std::forward< Args >( _args )... );
    }

    template< typename... Args >
    void warn( Args &&... _args )
    {
      print_level< 1 >( stderr, std::forward< Args >( _args )... );
    }
    
    template< typename... Args >
    void trace( Args &&... _args )
    {
      print_level< 2 >( std::forward< Args >( _args )... );
    }
    
    template< typename... Args >
    void debug( Args &&... _args )
    {
      print_level< 3 >( std::forward< Args >( _args )... );
    }

    template< typename... Args >
    void note( Args &&... _args )
    {
      print_level< 4 >( std::forward< Args >( _args )... );
    }
  }
}

#endif // INC_BVH_VT_PRINT_HPP
