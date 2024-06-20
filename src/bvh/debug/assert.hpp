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
#ifndef INC_BVH_DEBUG_ASSERT_HPP
#define INC_BVH_DEBUG_ASSERT_HPP

#include <cstdlib>
#include <string>
#include <iostream>
#include "../vt/print.hpp"
#include <spdlog/spdlog.h>

namespace bvh
{
#ifdef BVH_DEBUG_LEVEL
  constexpr int assert_debug_level = BVH_DEBUG_LEVEL;
#else
  constexpr int assert_debug_level = 0;
#endif

  namespace detail
  {
    template< bool Enable >
    struct debug_assert_impl
    {
      template< typename... Args >
      static void debug_assert( bool, const std::string &, Args &&... _args )
      {
        // Do nothing
      }
    };

    template<>
    struct debug_assert_impl< true >
    {
      template< typename... Args >
      static void debug_assert( bool _val, const std::string &_assert_str, Args &&..._args )
      {
        if ( !_val )
        {
          vt::error( _assert_str, std::forward< Args >( _args )... );
          std::terminate();
        }
      }
    };
  }

  template< int DebugLevel, typename... Args >
  void debug_assert_level( bool _val, const std::string &_msg, Args &&... _args )
  {
    static constexpr bool enable_assert = assert_debug_level >= DebugLevel;
    detail::debug_assert_impl< enable_assert >::debug_assert( _val, _msg, std::forward< Args >( _args )... );
  }

  template< typename... Args >
  inline void debug_assert( bool _val, const std::string &_msg, Args &&... _args )
  {
    debug_assert_level< 3 >( _val, _msg, std::forward< Args >( _args )... );
  }

  template< typename... Args >
  inline void always_assert( bool _val, const std::string &_msg, Args &&... _args )
  {
    debug_assert_level< 0 >( _val, _msg, std::forward< Args >( _args )... );
  }

  template< typename... Args >
  inline void abort( const std::string &_msg, Args &&... _args )
  {
    always_assert( false, _msg, std::forward< Args >( _args )... );
  }

  namespace detail
  {
    inline void assert_die( spdlog::logger &_logger, const char *_file, unsigned int _line, const char *_assertion,
                            const std::string &_msg )
    {
      _logger.critical( "assertion failed at {}:{}: {} ({})", _file, _line, _assertion, _msg );
      std::terminate();
    }

    inline void assert_die( std::shared_ptr< spdlog::logger > _logger, const char *_file, unsigned int _line,
                            const char *_assertion, const std::string &_msg )
    {
      assert_die( *_logger, _file, _line, _assertion, _msg );
    }
  }  // namespace detail
}

#define BVH_ASSERT_ALWAYS( expr, logger, ... )                                                                         \
  ( static_cast< bool >( expr ) ? void( 0 ) : ::bvh::detail::assert_die( logger, __FILE__, __LINE__, #expr, fmt::format( __VA_ARGS__ ) ) )

#endif  // INC_BVH_DEBUG_ASSERT_HPP
