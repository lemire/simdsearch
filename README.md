# simdsearch

Benchmarks and reference kernels for SIMD substring search.

## Requirements

A C++23 compiler (GCC 14+ or a recent Clang: the driver uses `std::print`) and
one of two architectures:

- **x86-64 with AVX-512F and AVX-512BW.** This is what Needle-Hammer was
  designed for and what the shipped switch points are fitted to. The AVX2 and
  SSE2 kernels in the same header exist to show that the switch point belongs to
  the register width rather than to the scheme, not as deployment targets.
- **AArch64 with NEON.** The same scheme at 128-bit width. Its constants are
  *not* the AVX-512 ones: the haystack guard rescales with the block size, the
  guard budget is re-fitted to preserve its free range, and the needle threshold
  turns out to be set by the adversarial bound rather than by a benign crossover.
  See the header for the argument.

A build on anything else stops at a `#error`.

## What is here

- `benchmark/include/avx512search.h` — the AVX-512 kernels and the
  length-dispatched **Needle-Hammer** scheme, plus 256-bit (AVX2) and 128-bit
  (SSE2) builds of the same kernels over a traits struct, so one x86 binary
  measures all three register widths.
- `benchmark/include/neonsearch.h` — the same three kernels, the same dispatch
  and the same work-counting guard at 128-bit AArch64 NEON width, with both
  constants exposed as sweep instances.
- `benchmark/include/common_search.h` — what both backends share: the anchor
  selector and the scalar and library baselines.
- `benchmark/benchmarks/benchmark.cpp` — the driver. Modes: `synthetic` (64 KiB
  random text, 100k short needles), `horspool`, `ashvardanian` (find-all),
  `worstcase`, `findall`.
- `benchmark/tests/` — validation against `std::string::find` across the
  needle- and haystack-length boundaries of every kernel, plus the find-all
  enumerator and misaligned haystack heads.
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
```
