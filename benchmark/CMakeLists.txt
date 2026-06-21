cmake_minimum_required(VERSION 3.10)

project(BenchmarkProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include(cmake/CPM.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Per-architecture SIMD flags. On x86-64 enable the AVX-512 subsets the
# avx512search.h backend uses (F + BW for masked byte compares, plus VL/DQ).
# On ARM, NEON is part of the AArch64 baseline so no extra flags are needed.
# Override with -DSIMDSEARCH_ARCH_FLAGS=... (e.g. "-march=native") if desired.
set(SIMDSEARCH_ARCH_FLAGS "" CACHE STRING "Architecture SIMD flags")
if(SIMDSEARCH_ARCH_FLAGS STREQUAL "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        if(MSVC)
            set(SIMDSEARCH_ARCH_FLAGS "/arch:AVX512")
        else()
            set(SIMDSEARCH_ARCH_FLAGS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq")
        endif()
    endif()
endif()

CPMAddPackage("gh:lemire/counters#v3.1.0")

add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE benchmarks include)
target_compile_options(benchmark PRIVATE ${SIMDSEARCH_ARCH_FLAGS})
target_link_libraries(benchmark PRIVATE counters)

enable_testing()
add_executable(test_search tests/test_search.cpp)
target_include_directories(test_search PRIVATE include)
target_compile_options(test_search PRIVATE ${SIMDSEARCH_ARCH_FLAGS})
add_test(NAME test_search COMMAND test_search)
