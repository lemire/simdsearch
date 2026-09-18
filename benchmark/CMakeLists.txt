cmake_minimum_required(VERSION 3.10)

project(BenchmarkProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include(cmake/CPM.cmake)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Needle-Hammer's anchor-count knobs can be overridden at configure time;
# empty means the header's shipped value.
set(NH2_START_ANCHORS "" CACHE STRING "Needle-Hammer: anchors a wide-alphabet needle starts with (2, 3 or 4)")
if(NOT NH2_START_ANCHORS STREQUAL "")
    add_compile_definitions(NH2_START_ANCHORS=${NH2_START_ANCHORS})
endif()
set(NH2_THREE_ANCHOR_MAX "" CACHE STRING "Needle-Hammer: longest needle that starts with fewer than four anchors")
if(NOT NH2_THREE_ANCHOR_MAX STREQUAL "")
    add_compile_definitions(NH2_THREE_ANCHOR_MAX=${NH2_THREE_ANCHOR_MAX})
endif()

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

# The Rust searchers (memchr crate, std::str::find) as extra rows, built from
# rust/ with cargo and linked statically. Off by default so the benchmark
# builds without a Rust toolchain; the rows appear in --list only when on.
option(SIMDSEARCH_RUST "Link the Rust searchers (memchr crate, std) via cargo" OFF)
if(SIMDSEARCH_RUST)
    find_program(CARGO_EXECUTABLE cargo REQUIRED)
    set(RUST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust)
    set(RUST_LIB ${RUST_DIR}/target/release/libsimdsearch_rust.a)
    add_custom_command(
        OUTPUT ${RUST_LIB}
        COMMAND ${CARGO_EXECUTABLE} build --release --quiet
        WORKING_DIRECTORY ${RUST_DIR}
        DEPENDS ${RUST_DIR}/Cargo.toml ${RUST_DIR}/src/lib.rs ${RUST_DIR}/build.rs
        COMMENT "cargo build --release (rust/)"
        VERBATIM)
    add_custom_target(simdsearch_rust DEPENDS ${RUST_LIB})
    add_dependencies(benchmark simdsearch_rust)
    target_compile_definitions(benchmark PRIVATE SIMDSEARCH_RUST=1)
    target_link_libraries(benchmark PRIVATE ${RUST_LIB})
    if(NOT APPLE)
        target_link_libraries(benchmark PRIVATE pthread dl m)
    endif()
endif()

enable_testing()
add_executable(test_search tests/test_search.cpp)
target_include_directories(test_search PRIVATE include)
target_compile_options(test_search PRIVATE ${SIMDSEARCH_ARCH_FLAGS})
add_test(NAME test_search COMMAND test_search)

# Needle-Hammer's own battery: block boundaries at every alignment, small
# haystacks, periodic needles, escalation and guard give-up, fuzz.
add_executable(test_needle_hammer tests/test_needle_hammer.cpp)
target_include_directories(test_needle_hammer PRIVATE include)
target_compile_options(test_needle_hammer PRIVATE ${SIMDSEARCH_ARCH_FLAGS})
add_test(NAME test_needle_hammer COMMAND test_needle_hammer)

