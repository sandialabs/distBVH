#ifndef INC_BVH_LOGGING_HPP
#define INC_BVH_LOGGING_HPP

#include <string>
#include <string_view>
#include <vt/transport.h>
#include <spdlog/spdlog.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/common.h>

namespace bvh::logging
{
  class rank_formatter_flag : public spdlog::custom_flag_formatter
  {
  public:

    void
    format(const spdlog::details::log_msg &, const std::tm &, spdlog::memory_buf_t &_dest ) override
    {
      std::string rank_str = std::to_string( ::vt::theContext()->getNode() );
      _dest.append( rank_str.data(), rank_str.data() + rank_str.size() );
    }

    std::unique_ptr< spdlog::custom_flag_formatter >
    clone() const override
    {
      return std::make_unique< rank_formatter_flag >();
    }
  };

  inline std::unique_ptr< spdlog::pattern_formatter >
  make_formatter()
  {
    auto ret = std::make_unique< spdlog::pattern_formatter >();
    ret->add_flag< rank_formatter_flag >( 'N' ).set_pattern( "[%N] %+" );
    return ret;
  }

  inline std::shared_ptr< spdlog::logger >
  make_logger( std::string _name, spdlog::sink_ptr _sink )
  {
    auto ret = std::make_shared< spdlog::logger >( std::move( _name ), std::move( _sink ) );
    ret->set_formatter( make_formatter() );
    ret->set_level( spdlog::level::trace );
    ret->flush_on( spdlog::level::trace );

    return ret;
  }
}

#endif  // INC_BVH_LOGGING_HPP
