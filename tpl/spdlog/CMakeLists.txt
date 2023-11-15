
#find_package(fmt 10.0.0 REQUIRED)

#target_link_libraries(bvh PUBLIC fmt::fmt)

target_compile_definitions(bvh PUBLIC SPDLOG_FMT_EXTERNAL SPDLOG_ACTIVE_LEVEL=0)

target_include_directories(bvh PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        )
