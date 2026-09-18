# simdsearch

Benchmarks and reference kernels for SIMD substring search.

## Requirements

A C++23 compiler (GCC 14+ or a recent Clang: the driver uses `std::print`) and
one of two architectures:

- **x86-64 with AVX-512F and AVX-512BW.** This is what Needle-Hammer is
  written for. The AVX2 and SSE2 builds of the component kernels in
  `avx512search.h` exist to read the AVX-512 kernels against the same designs
  at narrower widths, not as deployment targets.
- **AArch64 with NEON.** Currently the previous, length-dispatched scheme at
  128-bit width, with its constants re-fitted for ARM; see the header.

A build on anything else stops at a `#error`.

## What is here

- `benchmark/include/needle_hammer.h` — **Needle-Hammer**: one wide anchored
  kernel with chosen filter bytes, three or four of them as the haystack
  demands, a dedicated loop for needles of one to three bytes, and a work
  counter that resumes with the vectorized two-way in
  `benchmark/include/twoway_simd.h`, so every input is searched in linear time.
- `benchmark/include/avx512search.h` — the AVX-512 component kernels it is
  read against (single-window and 256-byte-stride naive filters, the
  three-anchor StringZilla-style filter), plus 256-bit (AVX2) and 128-bit
  (SSE2) builds of the same kernels over a traits struct, so one x86 binary
  measures all three register widths.
- `benchmark/include/neonsearch.h` — the previous scheme at 128-bit AArch64
  NEON width, with both of its constants exposed as sweep instances.
- `benchmark/include/common_search.h` — what both backends share: the anchor
  selector and the scalar and library baselines.
- `benchmark/benchmarks/benchmark.cpp` — the driver. Modes: `synthetic` (64 KiB
  random text, 100k short needles), `horspool`, `ashvardanian` (find-all),
  `worstcase`, `findall`.
- `benchmark/tests/` — validation against `std::string::find` across the
  needle- and haystack-length boundaries of every kernel, plus the find-all
  enumerator and misaligned haystack heads; `test_needle_hammer.cpp` adds the
  featured kernel's own battery (block boundaries at every alignment, small
  haystacks, periodic needles, escalation and guard give-up, fuzz).
- `benchmark/tools/corpora.py` — builds the eight benchmark datasets (English
  prose, DNA, protein, minified JSON, base64, log lines, C/C++ source, and UTF-8
  Cyrillic and CJK), about 1 MB each, and writes a `MANIFEST.txt` of their
  SHA-256 digests. Two of them are fetched from Project Gutenberg and fall back
  to a seeded synthetic generator offline, so check the manifest before comparing
  numbers across machines.
- `benchmark/README.md` — the algorithm list, build options and test notes.

## Build and run

```sh
cmake -B benchmark/build -S benchmark -DCMAKE_BUILD_TYPE=Release
cmake --build benchmark/build -j
benchmark/build/benchmark horspool benchmark/data/43-0.txt
benchmark/build/test_search        # validation
benchmark/build/test_needle_hammer # the featured kernel's battery (x86-64)
```
