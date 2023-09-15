#[[
distBVH 1.0

Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
]]

find_package(Doxygen)

find_program(SPHINX_EXECUTABLE NAMES sphinx-build DOC "Sphinx document generator")

mark_as_advanced(SPHINX_EXECUTABLE)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx REQUIRED_VARS SPHINX_EXECUTABLE DOXYGEN_FOUND)

set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
set(DOXYGEN_EXCLUDE_SYMBOLS detail)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_HIDE_UNDOC_MEMBERS YES)
set(DOXYGEN_HIDE_UNDOC_CLASSES YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_HIDE_FRIEND_COMPOUNDS YES)

if (SPHINX_EXECUTABLE)
  if (NOT TARGET Sphinx::build)
    add_executable(Sphinx::build IMPORTED GLOBAL)
    set_target_properties(Sphinx::build PROPERTIES IMPORTED_LOCATION "${SPHINX_EXECUTABLE}")
  endif()
endif()

set(_SPHINX_METHODS_DIR ${CMAKE_CURRENT_LIST_DIR})

function(sphinx_add_docs targetName _src)
  cmake_parse_arguments("ARG"
      ""
      "CONFIG;DOCDIR"
      ""
      ${ARGN})

  if (NOT DEFINED ARG_CONFIG)
    set(ARG_CONFIG "conf.py.in")
  endif()

  if (NOT DEFINED ARG_DOCDIR)
    set(ARG_DOCDIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()

  doxygen_add_docs(${targetName}_doxy ${_src})
  configure_file(${ARG_CONFIG} conf.py @ONLY)
  add_custom_target(${targetName} VERBATIM
      COMMAND ${CMAKE_COMMAND} --debug-output -DSPHINX_EXECUTABLE=${SPHINX_EXECUTABLE} -DSRC_DIR=${ARG_DOCDIR} -DDOC_DIR=${CMAKE_CURRENT_BINARY_DIR} -P ${_SPHINX_METHODS_DIR}/run_sphinx.cmake
    )

  add_dependencies(${targetName} ${targetName}_doxy)
endfunction()
