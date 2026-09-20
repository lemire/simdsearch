# simdsearch

A SIMD substring search in a header, and the benchmarks it was built with.

The searcher is **Needle-Hammer**: it filters every candidate position on two
to four needle bytes chosen for selectivity, 256 positions per iteration on
AVX-512 (64 on NEON), narrows the survivors byte by byte, and counts the work
it does. When the filter stops paying -- on a haystack built to defeat it --
the search hands over to a linear-time two-way, so **every input is searched
in O(n + m)** and there is no needle-length threshold to fit per machine. On
ordinary text it runs at memory speed (about 45 GB/s on an Apple M-series
core, 30x the platform `memmem`); on adversarial input its cost per byte is
flat in the needle length where an unguarded filter grows linearly with it.

Two things live here:

- **The library**: `include/simdsearch/`, header-only, C++17 -- or the same
  thing as one file, `singleheader/simdsearch.h`. Section
  [Using the library](#using-the-library).
- **The benchmarks**: `benchmark/`, a driver that measures Needle-Hammer
  against its own component kernels, the classical linear-time searchers,
  StringZilla, SSEF, the C library and (optionally) Rust's `memchr` crate.
  Section [Benchmarks and tests](#benchmarks-and-tests).

## Hardware and compiler requirements

The kernels are written for one of two instruction sets, **chosen when you
compile, not when you run**. There is no runtime dispatch: a binary built for
AVX-512 executes AVX-512 instructions unconditionally and dies with an illegal
instruction on a CPU that lacks them. If your program must run on machines
you do not control, dispatch to `simdsearch` yourself (with `cpuid` or your
compiler's `__builtin_cpu_supports`) and keep a fallback.

| Architecture | Requirement | Compiler flags | Examples |
|---|---|---|---|
| x86-64 | **AVX-512F and AVX-512BW** | `-mavx512f -mavx512bw` (the CMake target adds them) | Intel Skylake-SP and later server parts, Ice Lake, Tiger Lake, Sapphire/Emerald Rapids; AMD Zen 4 and Zen 5 |
| AArch64 | **NEON** (architectural on every 64-bit ARM) | none | Apple M1-M4, AWS Graviton, Ampere Altra, Raspberry Pi 4/5, NVIDIA Grace |

Not supported: x86-64 without AVX-512 (Intel client parts from Alder Lake on
have it fused off; AMD Zen 3 and earlier lack it), 32-bit anything, RISC-V,
POWER. On those the headers stop with a `#error`. The AVX2 and SSE2 kernels in
`avx512search.h` exist to *measure* the design at narrower widths; they are
not a fallback path.

Compilers: GCC 14 or later, or a recent Clang (tested: GCC 14 and 15, Apple
clang 17). The library needs C++17. The benchmark driver needs C++23
(`std::print`). MSVC is not supported (the headers use GCC/Clang builtins and
attributes).

## Using the library

### Quick start

```cpp
#include <simdsearch/simdsearch.h>
#include <string_view>

int main() {
    std::string_view hay = "The quick brown fox jumps over the lazy dog";

    // string_view in, index out (std::string_view::npos when absent)
    size_t pos = simdsearch::find(hay, "lazy");            // 35

    // pointer + length in, {found, index} out
    auto [found, index] = simdsearch::find(hay.data(), hay.size(), "fox", 3);   // {true, 16}
}
```

Build it on x86-64 with `g++ -std=c++17 -O2 -mavx512f -mavx512bw -Ipath/to/simdsearch/include app.cpp`,
on AArch64 with `g++ -std=c++17 -O2 -Ipath/to/simdsearch/include app.cpp`.
With CMake the flags come with the target (next section).

### Getting the headers into your project

The library is header-only; there is nothing to link. Pick one:

**CPM** ([cpm-cmake/CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)):

```cmake
CPMAddPackage("gh:lemire/simdsearch#main")     # or a tag
target_link_libraries(app PRIVATE simdsearch::simdsearch)
```

**FetchContent** (plain CMake, 3.14+):

```cmake
include(FetchContent)
FetchContent_Declare(simdsearch
    GIT_REPOSITORY https://github.com/lemire/simdsearch.git
    GIT_TAG        main)
FetchContent_MakeAvailable(simdsearch)
target_link_libraries(app PRIVATE simdsearch::simdsearch)
```

**Git submodule / vendored copy**:

```cmake
add_subdirectory(third_party/simdsearch)
target_link_libraries(app PRIVATE simdsearch::simdsearch)
```

**Installed** (`cmake --install build` from this repository, then):

```cmake
find_package(simdsearch REQUIRED)
target_link_libraries(app PRIVATE simdsearch::simdsearch)
```

**Single header from a release** (no git, no CMake): every release on the
[releases page](https://github.com/lemire/simdsearch/releases) attaches the
library amalgamated into one file, `simdsearch.h`, and its checksum. The
latest release is **v0.2.0**.

```sh
curl -LO https://github.com/lemire/simdsearch/releases/download/v0.2.0/simdsearch.h
curl -LO https://github.com/lemire/simdsearch/releases/download/v0.2.0/SHA256SUMS
sha256sum -c SHA256SUMS            # shasum -a 256 -c SHA256SUMS on macOS
```

Put `simdsearch.h` next to your sources, `#include "simdsearch.h"`, and
build with the flags from the requirements table -- nothing else is needed:

```sh
g++ -std=c++17 -O2 -mavx512f -mavx512bw app.cpp     # x86-64
g++ -std=c++17 -O2 app.cpp                          # AArch64
```

`https://github.com/lemire/simdsearch/releases/latest/download/simdsearch.h`
always points at the newest release, for scripts that want to track it.

**Single header from the repository**: the same file is checked in as
`singleheader/simdsearch.h`, generated by `tools/amalgamate.py` from
`include/simdsearch/` (both backends included, selected by the same
preprocessor tests) and checked in CI against the sources.

**Several headers, no CMake**: copy `include/simdsearch/` into your include
path and `#include <simdsearch/simdsearch.h>`. The headers only include each
other by bare name, so the directory can sit anywhere. Only five of them are
part of the library's closure (`simdsearch.h`, `needle_hammer.h`,
`twoway_simd.h`, `twoway_prep.h`, `avx512_naive.h`); the rest belong to the
benchmark.

Linking `simdsearch::simdsearch` gives you the include directory, `cxx_std_17`
and, on x86-64, the AVX-512 flags. When simdsearch is a dependency its tests
and benchmarks are not built, and their dependencies are not fetched.

To take control of the architecture flags yourself:

```cmake
set(SIMDSEARCH_ARCH_FLAGS "-march=native" CACHE STRING "")   # something else
set(SIMDSEARCH_ARCH_FLAGS "none" CACHE STRING "")            # nothing at all
```

### The API

Everything is in `<simdsearch/simdsearch.h>`, namespace `simdsearch`.

```cpp
// First occurrence of pattern[0..m) in text[0..n).
std::pair<bool, size_t> find(const char* text, size_t n, const char* pattern, size_t m);

// The same on string_views: the byte index, or std::string_view::npos.
size_t find(std::string_view haystack, std::string_view needle);

// The same kernel with its work counter compiled out (see below).
std::pair<bool, size_t> find_unguarded(const char* text, size_t n, const char* pattern, size_t m);
```

What you can rely on:

- **Result.** The byte offset of the *first* occurrence, or `{false, 0}` /
  `npos`. The same answer as `std::string_view::find`, on every input; the
  tests check exactly that.
- **Empty needle** matches at 0, including in an empty haystack.
- **Any bytes.** Neither buffer needs a NUL terminator; NUL and bytes above
  0x7F are ordinary bytes. This is a byte search, not a character search: on
  UTF-8 text a match can start inside a multi-byte character if the needle
  does.
- **No over-read.** No byte outside `[text, text + n)` or
  `[pattern, pattern + m)` is ever loaded, so buffers need no padding and the
  tests run under AddressSanitizer with exact-size allocations.
- **Linear time.** O(n + m) on every input. On ordinary text the filter runs
  the whole way; on input built to defeat it, the work counter hands the
  remaining haystack to a Crochemore-Perrin two-way that never re-reads a
  byte. `find_unguarded` is the same search without the counter: a few
  percent faster on text, O(n·m) in the worst case. It exists to measure the
  guard; use it only on input you control.
- **No allocation, no state.** Nothing is malloc'ed, nothing is cached, there
  are no globals; calls are reentrant and thread-safe. Needle preprocessing
  (a few vector compares over the needle) happens inside every call, so
  there is no `Finder` object to keep around and nothing to amortize.
- **Sizes.** `n` and `m` are `size_t`; needles longer than the haystack
  return "not found" at once.

### Tuning knobs

Three constants in `needle_hammer.h` can be overridden at compile time, with
`-D` or the CMake cache variables of the same names. The shipped values were
fitted on Emerald Rapids, Zen 5 and Apple M4 Max; you should not need to
touch them.

| Macro | Default | Meaning |
|---|---|---|
| `NH2_START_ANCHORS` | 2 | Filter bytes a wide-alphabet needle starts with (2, 3 or 4); more are added as the haystack demands |
| `NH2_THREE_ANCHOR_MAX` | unlimited | Needles longer than this start with all four anchors |
| `NH2_BUDGET_DEN` | 64 (AVX-512), 16 (NEON) | Work budget before handing over to two-way, as `n / NH2_BUDGET_DEN` narrowing rounds |

### What is in `include/simdsearch/`

| Header | Contents |
|---|---|
| `simdsearch.h` | The public API above |
| `needle_hammer.h` | Needle-Hammer: anchor selection, the wide kernel, the masked/overlap windows, the short-needle loop, the escalation driver; AVX-512 and NEON backends |
| `twoway_simd.h` | The linear-time fallback: Crochemore-Perrin two-way with 64-byte (AVX-512) or 16-byte (NEON) comparison loops |
| `twoway_prep.h` | The two-way critical factorization both two-ways build on |
| `avx512_naive.h` | The AVX-512 alignment head and four-byte-prefix kernels Needle-Hammer dispatches to |
| `kmp_twoway.h` | Scalar KMP and two-way (benchmark baselines) |
| `avx512search.h`, `neonsearch.h` | The component kernels Needle-Hammer is measured against (naive single-window and wide-stride filters, the StringZilla-style three-anchor filter), plus AVX2/SSE2 builds of the x86 ones |
| `common_search.h` | Shared anchor selector, and the scalar and C-library baselines |
| `ssef.h` | SSEF (Külekci 2009), a sublinear block-skipping filter for long needles |

Only `simdsearch.h` is API. The rest is included by the benchmark and the
tests and is documented in the headers themselves, which explain every
constant with the measurement behind it. After editing any of the first five,
run `python3 tools/amalgamate.py` to refresh `singleheader/simdsearch.h`
(`ctest` fails the `singleheader_up_to_date` test otherwise).

## Benchmarks and tests

```sh
cmake -B build -S .                     # Release by default
cmake --build build -j
ctest --test-dir build --output-on-failure

./build/benchmark/benchmark horspool benchmark/data/43-0.txt
./build/benchmark/benchmark horspool corpora/english.dat --lengths 4,8,16,64 --patterns 500
./build/benchmark/benchmark horspool --list   # every searcher the driver knows
```

Modes: `synthetic` (64 KiB random text, 100k short needles), `horspool`
(random substrings of a file), `ashvardanian` (StringWars-style find-all),
`worstcase` (adversarial haystack/needle shapes), `bigscan` (the file tiled
up to 1 GiB, GB/s once the haystack no longer fits in cache), `findall`.
`benchmark/README.md` has the full algorithm list, the options and what each
row measures. `benchmark/tools/corpora.py` builds the nine ~1 MB datasets
(English, DNA, protein, JSON, base64, logs, C source, Cyrillic and CJK UTF-8)
into `corpora/` with a `MANIFEST.txt` of SHA-256 digests.

Tests (`ctest`, or the binaries in `build/benchmark/`):

- `test_search` -- every searcher against `std::string::find` across the
  needle- and haystack-length boundaries of every kernel, misaligned heads,
  the find-all enumerator, and two deterministic guard tests that make
  Needle-Hammer exhaust its budget and check that the two-way resume still
  finds a match placed past the give-up point.
- `test_needle_hammer` -- Needle-Hammer tests: block boundaries at every
  alignment, small haystacks, periodic needles, escalation, 320k fuzz cases.
- `test_api` -- the public header at C++17 from two translation units,
  exact-size heap buffers (so an over-read is a sanitizer error, not a read
  into `std::string`'s slack), NUL bytes, the full byte range.
- `test_api_singleheader` -- the same against `singleheader/simdsearch.h`
  with nothing else on the include path; `singleheader_up_to_date` checks
  the generated file matches the sources.

To run the tests under AddressSanitizer and UBSan:

```sh
cmake -B build-san -S . -DSIMDSEARCH_BENCHMARKS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-san -j && ctest --test-dir build-san --output-on-failure
```

### Continuous integration

`.github/workflows/ci.yml` runs the tests, with sanitizers and `-Werror`, on
three runners:

- **NEON**, natively: GitHub's Linux ARM64 runner (GCC 14) and its Apple
  Silicon macOS runner (Apple clang).
- **AVX-512**, under [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
  on an x86-64 runner. GitHub's x86-64 runners are not guaranteed to have
  AVX-512, and QEMU cannot help: its TCG stops at AVX2 (there is an open
  request, [qemu#2878](https://gitlab.com/qemu-project/qemu/-/issues/2878)).
  SDE translates AVX-512 instructions on any x86-64 host, so the job builds
  with `-mavx512f -mavx512bw` and runs the test binaries under
  `sde64 -spr` (Sapphire Rapids). It is slower than native, and sanitizer
  builds are run on that job only when the runner turns out to have AVX-512.

Benchmark numbers are never taken from CI; run the driver on the machine you
care about.

### Releases

A release is cut by the **Release** workflow, from the Actions tab (Release,
"Run workflow") or from a terminal:

```sh
gh workflow run release.yml -f bump=patch      # or minor, major
```

The workflow runs the CI, takes the last `v*` tag and bumps the chosen
component (the first release counts from v0.0.0), writes the new version into
`CMakeLists.txt` and into the release section of this README, commits that
to `main`, tags the commit, and publishes a GitHub release with
`simdsearch.h` and `SHA256SUMS` attached and notes generated from the commits
since the previous tag. Nothing is edited by hand for a release; the version
in `CMakeLists.txt` is always the last one published.

## Layout

```
include/simdsearch/     the library (header-only)
singleheader/           the library as one generated file
tools/amalgamate.py     generates it
benchmark/
  benchmarks/           the driver
  tests/                test_search, test_needle_hammer, test_api
  tools/corpora.py      builds the datasets in corpora/
  rust/                 optional Rust searchers (-DSIMDSEARCH_RUST=ON)
  data/43-0.txt         a Project Gutenberg text the driver defaults to
corpora/                generated datasets (git-ignored; MANIFEST.txt lists digests)
```
