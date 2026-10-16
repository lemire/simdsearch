cmake_minimum_required(VERSION 3.14)

# The VERSION is the last release; the Release workflow bumps it, do not edit.
project(simdsearch VERSION 0.4.0 LANGUAGES CXX
        DESCRIPTION "AVX-512 substring search: Guardzilla and its benchmarks")

if(CMAKE_VERSION VERSION_LESS 3.21)
    string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" PROJECT_IS_TOP_LEVEL)
endif()

# ---------------------------------------------------------------------------
# The library: header-only. Link simdsearch::simdsearch and include
# <simdsearch/simdsearch.h>.
# ---------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(simdsearch INTERFACE)
add_library(simdsearch::simdsearch ALIAS simdsearch)
target_include_directories(simdsearch INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(simdsearch INTERFACE cxx_std_17)

# The library needs AVX-512 (F + BW), chosen at compile time. By default the
# target carries the flags, so that linking it is enough.
# Set SIMDSEARCH_ARCH_FLAGS to override (e.g. "-march=native"), or to "none"
# to add nothing and manage the flags yourself.
set(SIMDSEARCH_ARCH_FLAGS "" CACHE STRING
    "Compiler flags the simdsearch target adds (default: AVX-512 on x86-64; \"none\" for nothing)")
if(SIMDSEARCH_ARCH_FLAGS STREQUAL "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(SIMDSEARCH_ARCH_FLAGS "-mavx512f;-mavx512bw")
    endif()
endif()
if(NOT SIMDSEARCH_ARCH_FLAGS STREQUAL "" AND NOT SIMDSEARCH_ARCH_FLAGS STREQUAL "none")
    target_compile_options(simdsearch INTERFACE ${SIMDSEARCH_ARCH_FLAGS})
endif()


# ---------------------------------------------------------------------------
# Install: the headers and a config so find_package(simdsearch) works.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(DIRECTORY include/simdsearch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS simdsearch EXPORT simdsearchTargets)
install(EXPORT simdsearchTargets
        NAMESPACE simdsearch::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdsearch)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/simdsearchConfigVersion.cmake
    COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/simdsearchConfig.cmake
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/simdsearchTargets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/simdsearchConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/simdsearchConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdsearch)

# ---------------------------------------------------------------------------
# Tests and benchmarks: on when this is the top-level project, off when
# simdsearch is pulled in as a dependency.
# ---------------------------------------------------------------------------
option(SIMDSEARCH_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(SIMDSEARCH_BENCHMARKS "Build the benchmark (needs network access to fetch StringZilla)" ${PROJECT_IS_TOP_LEVEL})
if(SIMDSEARCH_TESTS OR SIMDSEARCH_BENCHMARKS)
    enable_testing()
    add_subdirectory(benchmark)
endif()
