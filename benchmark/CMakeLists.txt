cmake_minimum_required(VERSION 3.10)

project(BenchmarkProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include(cmake/CPM.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Needle-Hammer's two constants can be overridden at configure time, which is
# how the tau/mu sweep harness re-fits them per machine without editing the
# header. Empty means "use the header's shipped value".
set(NEON_NH_TAU "" CACHE STRING "NEON Needle-Hammer needle threshold (tau)")
set(NEON_NH_MU "" CACHE STRING "NEON Needle-Hammer haystack guard (mu)")
foreach(knob NEON_NH_TAU NEON_NH_MU)
    if(NOT ${knob} STREQUAL "")
        add_compile_definitions(${knob}=${${knob}})
    endif()
endforeach()

# SIMD flags: on x86-64, the AVX-512 subsets avx512search.h uses (F + BW for
# masked byte compares, plus VL/DQ) and the AVX2/SSE2 the narrower kernels need.
# On AArch64 NEON is architectural, so no flag is required.
# Override with -DSIMDSEARCH_ARCH_FLAGS=... (e.g. "-march=native") if desired.
set(SIMDSEARCH_ARCH_FLAGS "" CACHE STRING "Architecture SIMD flags")
if(SIMDSEARCH_ARCH_FLAGS STREQUAL "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        if(MSVC)
            set(SIMDSEARCH_ARCH_FLAGS "/arch:AVX512")
        else()
            # -mavx2/-msse2 are implied by the AVX-512 subsets, but the 256-bit
            # and 128-bit needle-hammer kernels depend on them directly, so name
            # them rather than rely on the implication.
            set(SIMDSEARCH_ARCH_FLAGS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx2;-msse2")
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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # Needle-Hammer's own battery: block boundaries at every alignment,
    # small haystacks, periodic needles, escalation and guard give-up, fuzz.
    add_executable(test_needle_hammer tests/test_needle_hammer.cpp)
    target_include_directories(test_needle_hammer PRIVATE include)
    target_compile_options(test_needle_hammer PRIVATE ${SIMDSEARCH_ARCH_FLAGS})
    add_test(NAME test_needle_hammer COMMAND test_needle_hammer)
endif()

