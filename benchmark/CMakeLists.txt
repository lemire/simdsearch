# Tests and the benchmark driver, added by the top-level CMakeLists.txt, which
# defines the simdsearch::simdsearch target, the architecture flags and
# Needle-Hammer's knobs. Configure from the repository root: cmake -B build -S .

# ---------------------------------------------------------------------------
# Tests: C++17 is enough for the library.
# ---------------------------------------------------------------------------
if(SIMDSEARCH_TESTS)
    add_executable(test_search tests/test_search.cpp)
    target_link_libraries(test_search PRIVATE simdsearch::simdsearch)
    add_test(NAME test_search COMMAND test_search)

    # Needle-Hammer tests: block boundaries at every alignment, small
    # haystacks, periodic needles, escalation and guard give-up, fuzz.
    add_executable(test_needle_hammer tests/test_needle_hammer.cpp)
    target_link_libraries(test_needle_hammer PRIVATE simdsearch::simdsearch)
    add_test(NAME test_needle_hammer COMMAND test_needle_hammer)

    # Compiles the public header in a translation unit of its own, at the
    # library's C++ standard, and checks that including it from two
    # translation units links.
    add_executable(test_api tests/test_api.cpp tests/test_api_second_tu.cpp)
    target_link_libraries(test_api PRIVATE simdsearch::simdsearch)
    set_target_properties(test_api PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    add_test(NAME test_api COMMAND test_api)

    # The same test against the single-header build in singleheader/, with
    # nothing but that directory on the include path, and a check that the
    # generated file is in sync with include/simdsearch/.
    add_executable(test_api_singleheader tests/test_api.cpp tests/test_api_second_tu.cpp)
    target_include_directories(test_api_singleheader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../singleheader)
    target_compile_definitions(test_api_singleheader PRIVATE SIMDSEARCH_SINGLE_HEADER=1)
    target_compile_options(test_api_singleheader PRIVATE $<TARGET_PROPERTY:simdsearch,INTERFACE_COMPILE_OPTIONS>)
    set_target_properties(test_api_singleheader PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    add_test(NAME test_api_singleheader COMMAND test_api_singleheader)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME singleheader_up_to_date
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/amalgamate.py --check)
    endif()
endif()

# ---------------------------------------------------------------------------
# Benchmark driver: C++23 (std::print), the counters library and StringZilla
# fetched with CPM, optionally the Rust searchers.
# ---------------------------------------------------------------------------
if(SIMDSEARCH_BENCHMARKS)
    include(cmake/CPM.cmake)
    CPMAddPackage("gh:lemire/counters#v3.1.0")
    # StringZilla, header-only, for the find_stringzilla row (the library as
    # shipped, beside our ports of its anchored kernel).
    CPMAddPackage(NAME stringzilla GITHUB_REPOSITORY ashvardanian/StringZilla GIT_TAG v5.1.2 DOWNLOAD_ONLY YES)

    add_executable(benchmark benchmarks/benchmark.cpp)
    target_include_directories(benchmark PRIVATE benchmarks)
    # SYSTEM: third-party headers are not held to this project's warning flags.
    target_include_directories(benchmark SYSTEM PRIVATE ${stringzilla_SOURCE_DIR}/include)
    target_link_libraries(benchmark PRIVATE simdsearch::simdsearch counters)
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)

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
endif()
