# Tests and the benchmark, added by the top-level CMakeLists.txt, which defines
# the simdsearch::simdsearch target and the architecture flags. Configure from
# the repository root: cmake -B build -S .

# ---------------------------------------------------------------------------
# Tests: C++17 is enough for the library.
# ---------------------------------------------------------------------------
if(SIMDSEARCH_TESTS)
    # Guardzilla against std::string_view::find: fuzz, every alignment, the
    # adversarial families and the guard's hand-over to two-way. Built once per
    # setting the benchmark measures, so that every measured variant is tested.
    set(GUARDZILLA_TEST_SETTINGS
        "default\;"
        "anchors1\;GUARDZILLA_ANCHORS=1" "anchors2\;GUARDZILLA_ANCHORS=2" "anchors4\;GUARDZILLA_ANCHORS=4"
        "anchors8\;GUARDZILLA_ANCHORS=8" "noalign\;GUARDZILLA_ALIGN=0" "budget1\;GUARDZILLA_BUDGET=1")
    foreach(setting ${GUARDZILLA_TEST_SETTINGS})
        list(GET setting 0 tag)
        list(GET setting 1 def)
        add_executable(test_guardzilla_${tag} tests/test_guardzilla.cpp)
        target_link_libraries(test_guardzilla_${tag} PRIVATE simdsearch::simdsearch)
        if(def)
            target_compile_definitions(test_guardzilla_${tag} PRIVATE ${def})
        endif()
        add_test(NAME test_guardzilla_${tag} COMMAND test_guardzilla_${tag})
    endforeach()

    # NeedleHammer (needs AVX-512 VBMI2): as published, and switching to its
    # second scan almost at once, in windows of 128 bytes, and staying there.
    function(add_needlehammer_test tag)
        add_executable(test_needlehammer_${tag} tests/test_guardzilla.cpp)
        target_link_libraries(test_needlehammer_${tag} PRIVATE simdsearch::simdsearch)
        target_compile_definitions(test_needlehammer_${tag} PRIVATE TEST_NEEDLEHAMMER ${ARGN})
        target_compile_options(test_needlehammer_${tag} PRIVATE -mavx512vbmi2)
        add_test(NAME test_needlehammer_${tag} COMMAND test_needlehammer_${tag})
    endfunction()
    add_needlehammer_test(default)
    add_needlehammer_test(eager NEEDLEHAMMER_WINDOW=2 NEEDLEHAMMER_LONG_WINDOW=3 NEEDLEHAMMER_BAND=1 NEEDLEHAMMER_TIMED=0)
    add_needlehammer_test(eager_anchors2 NEEDLEHAMMER_WINDOW=2 NEEDLEHAMMER_LONG_WINDOW=3 NEEDLEHAMMER_BAND=1 NEEDLEHAMMER_TIMED=0 NEEDLEHAMMER_ANCHORS=2)

    # The public header in a translation unit of its own, at the library's C++
    # standard, included from two translation units of one program.
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
# The benchmark: one executable per searcher, search_bench_<id>. The searcher
# is fixed at compile time; with several in one program, the measured speed
# of one moved by up to 1.45x depending on what else was compiled in.
# ---------------------------------------------------------------------------
if(SIMDSEARCH_BENCHMARKS)
    include(cmake/CPM.cmake)
    # The counters library: cycles, instructions and branch misses per search.
    CPMAddPackage("gh:lemire/counters#v3.1.0")
    # StringZilla, header-only, as shipped.
    CPMAddPackage(NAME stringzilla GITHUB_REPOSITORY ashvardanian/StringZilla GIT_TAG v5.1.2 DOWNLOAD_ONLY YES)

    # The Rust searchers (memchr crate, std's str::find), built from rust/ with
    # cargo and linked statically. Off by default so that the benchmark builds
    # without a Rust toolchain.
    option(SIMDSEARCH_RUST "Build the Rust searchers (memchr crate, str::find) with cargo" OFF)
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
    endif()

    # search_bench_<name> for SEARCHER=<id> plus extra definitions.
    function(add_search_bench name id)
        add_executable(search_bench_${name} benchmarks/search_bench.cpp)
        target_link_libraries(search_bench_${name} PRIVATE simdsearch::simdsearch counters)
        target_include_directories(search_bench_${name} SYSTEM PRIVATE ${stringzilla_SOURCE_DIR}/include)
        target_compile_definitions(search_bench_${name} PRIVATE SEARCHER=SEARCHER_${id} ${ARGN})
        set_target_properties(search_bench_${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
        if(id MATCHES "^rust_")
            add_dependencies(search_bench_${name} simdsearch_rust)
            target_link_libraries(search_bench_${name} PRIVATE ${RUST_LIB})
            if(NOT APPLE)
                target_link_libraries(search_bench_${name} PRIVATE pthread dl m)
            endif()
        endif()
    endfunction()

    # The searchers compared in the paper.
    add_search_bench(guardzilla guardzilla)
    add_search_bench(needlehammer needlehammer)
    target_compile_options(search_bench_needlehammer PRIVATE -mavx512vbmi2)
    add_search_bench(stringzilla stringzilla)
    add_search_bench(memmem memmem)
    add_search_bench(string_view string_view)
    add_search_bench(boyer_moore_horspool boyer_moore_horspool)
    add_search_bench(twoway twoway)
    if(SIMDSEARCH_RUST)
        add_search_bench(rust_memmem rust_memmem)
        add_search_bench(rust_finder rust_finder)
        add_search_bench(rust_str rust_str)
    endif()
    # Sensitivity: the number of anchors, alignment, the budget.
    foreach(k 1 2 4 5 6 7 8)
        add_search_bench(guardzilla_anchors${k} guardzilla GUARDZILLA_ANCHORS=${k})
    endforeach()
    add_search_bench(guardzilla_noalign guardzilla GUARDZILLA_ALIGN=0)
    add_search_bench(guardzilla_budget2 guardzilla GUARDZILLA_BUDGET=2)
    add_search_bench(guardzilla_budget8 guardzilla GUARDZILLA_BUDGET=8)
    # The constants of NeedleHammer's switch threshold on this processor.
    add_executable(branch_penalty tools/branch_penalty.cpp)
    target_link_libraries(branch_penalty PRIVATE simdsearch::simdsearch counters)
    target_compile_options(branch_penalty PRIVATE -mavx512vbmi2)
    set_target_properties(branch_penalty PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    # NeedleHammer's switch threshold, window and second-scan anchors.
    foreach(b 10 25)
        add_search_bench(needlehammer_band${b} needlehammer NEEDLEHAMMER_BAND=${b})
        target_compile_options(search_bench_needlehammer_band${b} PRIVATE -mavx512vbmi2)
    endforeach()
    add_search_bench(needlehammer_window16 needlehammer NEEDLEHAMMER_WINDOW=16)
    target_compile_options(search_bench_needlehammer_window16 PRIVATE -mavx512vbmi2)
    foreach(a 1 2 4 5 6)
        add_search_bench(needlehammer_anchors${a} needlehammer NEEDLEHAMMER_ANCHORS=${a})
        target_compile_options(search_bench_needlehammer_anchors${a} PRIVATE -mavx512vbmi2)
    endforeach()
endif()
