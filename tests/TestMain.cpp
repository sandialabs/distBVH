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
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <Kokkos_Core.hpp>

#include <vt/transport.h>

int
main( int _argc, char **_argv )
{

  const std::string catch_init_flag("--list-test-names-only");
  for (int ii = 0; ii < _argc; ++ii) {
    std::string arg_string(_argv[ii]);
    if (arg_string.find(catch_init_flag) != std::string::npos) {
      int result = Catch::Session().run( _argc, _argv );
      return result;
    }
  }

  Kokkos::initialize( _argc, _argv );
  int ret = 0;

  {
    ::vt::initialize(_argc, _argv);

    Catch::Session sesh;
    int argret = sesh.applyCommandLine( _argc, _argv );
    if ( argret != 0 )
      return argret;

    vt::runInEpochCollective( "main test", [_argc, _argv, &ret, &sesh](){
      ret = sesh.run();
    } );

    ::vt::finalize();
  }

  Kokkos::finalize();
  return ret;
}
