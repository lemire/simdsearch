# simdsearch

Fast substring search with AVX-512, in one header.

The searcher is **Guardzilla**: the substring search of
[StringZilla](https://github.com/ashvardanian/StringZilla) with two changes.

1. **The scan is aligned.** StringZilla tests three bytes of the needle at 64
   haystack positions per step; Guardzilla takes those steps on 64-byte
   boundaries of the haystack.
2. **The work is guarded.** StringZilla compares every position that passes
   the test against the whole needle, which costs O(nm) on repetitive input.
   Guardzilla compares a candidate against the needle's first 64 bytes, counts
   the comparisons beyond those, and hands the rest of the search to the
   two-way algorithm when they exceed a budget proportional to the bytes
   scanned. **Every search takes O(n + m) time.**

On ordinary text it is as fast as StringZilla, about twice as fast as Rust's
`memchr` crate, and several times faster than the C library's `memmem`.

A second searcher, **NeedleHammer** (`simdsearch/needlehammer.h`, namespace
`needlehammer`), is Guardzilla with a second scan for inputs that defeat the
branch predictor: when a fair share of the 64-byte blocks have a candidate, at
an irregular rate (DNA, some logs), it compresses the candidates' positions
(VPCOMPRESSB) and checks them without a data-dependent branch, an idea of
Geoff Langdale's. It is 2 to 4 times faster than Guardzilla on DNA and at most
about 5% slower elsewhere, also in O(n + m) time. It also needs AVX-512 VBMI2 and VL
(Ice Lake, Sapphire Rapids and later; Zen 4 and later): compile with
`-mavx512vbmi2 -mavx512vl`.

```cpp
#include <simdsearch/needlehammer.h>
auto [found, index] = needlehammer::find(hay.data(), hay.size(), "fox", 3);
```

## Requirements

An x86-64 processor with **AVX-512F and AVX-512BW** (Intel Skylake-SP and
later server parts, Ice Lake, Sapphire/Emerald/Granite Rapids; AMD Zen 4 and
Zen 5), chosen at compile time: there is no runtime dispatch. GCC 14 or a
recent Clang, C++17.

## Use

```cpp
#include <simdsearch/simdsearch.h>   // or singleheader/simdsearch.h

std::string_view hay = "The quick brown fox jumps over the lazy dog";
size_t pos = simdsearch::find(hay, "fox");            // 16, or npos if absent
auto [found, index] = simdsearch::find(hay.data(), hay.size(), "fox", 3);
```

The empty needle matches at 0. Neither buffer needs a terminator, both may
hold any byte, and no byte outside them is read.

With CMake: `add_subdirectory(simdsearch)` or `find_package(simdsearch)`, then
link `simdsearch::simdsearch`; the target adds `-mavx512f -mavx512bw`
(override with `-DSIMDSEARCH_ARCH_FLAGS=-march=native`). Otherwise copy
`singleheader/simdsearch.h` and compile with those flags.

**Single header from a release** (no git, no CMake): every release on the
[releases page](https://github.com/lemire/simdsearch/releases) attaches the
library amalgamated into one file, `simdsearch.h` (Guardzilla), and its
checksum. The latest release is **v0.3.1**.

```sh
curl -LO https://github.com/lemire/simdsearch/releases/download/v0.3.1/simdsearch.h
curl -LO https://github.com/lemire/simdsearch/releases/download/v0.3.1/SHA256SUMS
sha256sum -c SHA256SUMS            # shasum -a 256 -c SHA256SUMS on macOS
g++ -std=c++17 -O2 -mavx512f -mavx512bw app.cpp
```

## Tests and benchmarks

```sh
cmake -B build -DSIMDSEARCH_RUST=ON -DSIMDSEARCH_ARCH_FLAGS=-march=native
cmake --build build -j
ctest --test-dir build
python3 benchmark/tools/corpora.py     # the corpora; compare with benchmark/data/corpora.sha256
benchmark/run.sh results/$(hostname)   # every measurement in the paper
```

`-DSIMDSEARCH_RUST=ON` needs a Rust toolchain; it adds Rust's `memchr` crate
and `str::find` to the comparison. Each searcher is its own executable,
`build/benchmark/search_bench_<name>`: compiled into one program, the
searchers' speeds moved by up to 1.45x depending on what else was linked in.

The compared searchers are Guardzilla, NeedleHammer, StringZilla 5.1.2 (`sz_find`), the C
library's `memmem`, `std::string_view::find`,
`std::boyer_moore_horspool_searcher`, and Rust's `memchr::memmem::find`,
`memchr::memmem::Finder` and `str::find`. The settings `GUARDZILLA_ANCHORS`,
`GUARDZILLA_ALIGN` and `GUARDZILLA_BUDGET` (see `guardzilla.h`), and the
`NEEDLEHAMMER_*` settings (see `needlehammer.h`), exist for the sensitivity
measurements. When the processor allows it, each search also records cycles,
instructions, branches and mispredicted branches (the
[counters](https://github.com/lemire/counters) library); on Linux this needs
`kernel.perf_event_paranoid` at 1 or less. `build/benchmark/branch_penalty`
measures the two constants of NeedleHammer's switch on the current processor.
