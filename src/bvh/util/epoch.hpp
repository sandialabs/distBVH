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
#ifndef INC_BVH_UTIL_EPOCH_HPP
#define INC_BVH_UTIL_EPOCH_HPP

#include <vt/transport.h>
#include <functional>

namespace bvh
{
  class epoch
  {
  public:

    enum class type
    {
      rooted,
      collective
    };

    epoch()
      : m_epoch( ::vt::no_epoch )
    {}

    explicit epoch( type _init_type )
        : m_epoch( ::vt::no_epoch )
    {
      switch ( _init_type )
      {
        case type::rooted: m_epoch = ::vt::theTerm()->makeEpochRooted(); break;
        case type::collective: m_epoch = ::vt::theTerm()->makeEpochCollective(); break;
      }
    }

    explicit epoch( ::vt::EpochType _ep )
      : m_epoch( _ep )
    {}

    epoch( const epoch & ) = delete;
    epoch( epoch &&_other ) noexcept
      : epoch()
    {
      std::swap( m_epoch, _other.m_epoch );
    }

    epoch &operator=( const epoch & ) = delete;
    epoch &operator=( epoch &&_other ) noexcept
    {
      std::swap( m_epoch, _other.m_epoch );
      return *this;
    }

    static epoch make_collective()
    {
      return epoch( ::vt::theTerm()->makeEpochCollective() );
    }

    static epoch make_rooted()
    {
      return epoch{ ::vt::theTerm()->makeEpochRooted() };
    }

    void push()
    {
      ::vt::theMsg()->pushEpoch( m_epoch );
    }

    void pop()
    {
      ::vt::theMsg()->popEpoch( m_epoch );
    }

    template< typename F >
    void add_action( F &&_fun )
    {
      ::vt::theTerm()->addAction( m_epoch, std::forward< F >( _fun ) );
    }

    ~epoch()
    {
      if ( m_epoch != ::vt::no_epoch )
        ::vt::theTerm()->finishedEpoch( m_epoch );
    }

    void reset()
    {
      if ( m_epoch != ::vt::no_epoch )
      {
        ::vt::theTerm()->finishedEpoch( m_epoch );
        m_epoch = ::vt::no_epoch;
      }
    }

    operator bool() const noexcept
    {
      return m_epoch != ::vt::no_epoch;
    }

    operator ::vt::EpochType() const noexcept
    {
      return m_epoch;
    }

  private:

    ::vt::EpochType m_epoch;
  };

  class epoch_guard
  {
  public:

    explicit epoch_guard( epoch &_ep )
      : m_epoch( &_ep )
    {
      m_epoch->push();
    }

    ~epoch_guard()
    {
      finish();
    }

    void finish() noexcept
    {
      if ( *m_epoch )
        m_epoch->pop();
      m_epoch->reset();
    }

    template< typename F >
    void add_action( F &&_fun )
    {
      m_epoch->add_action( std::forward< F >( _fun ) );
    }

    void run_scheduler_through()
    {
      ::vt::runSchedulerThrough( *m_epoch );
    }

  protected:

    epoch *m_epoch;
  };

  class sync_epoch_guard : public epoch_guard
  {
  public:

    explicit sync_epoch_guard( epoch &_ep )
      : epoch_guard( _ep )
    {

    }

    ~sync_epoch_guard()
    {
      if ( *m_epoch )
        m_epoch->pop();
      run_scheduler_through();
      m_epoch->reset();
    }
  };
}

#endif  // INC_BVH_UTIL_EPOCH_HPP
