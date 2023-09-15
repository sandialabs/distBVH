find_package(Sphinx)

if (Sphinx_FOUND)
  sphinx_add_docs(doc ${PROJECT_SOURCE_DIR}/src/)
endif()

