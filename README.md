# simdsearch

Benchmarks and reference kernels for SIMD substring search.

## Requirements

**An x86-64 processor with AVX-512, specifically AVX-512F and AVX-512BW, running
Linux with GCC 14 or newer.** That is the only supported configuration: it is
what Needle-Hammer is designed for and what the switch points are fitted to.

Nothing else is supported. The AVX2 and SSE2 builds exist to demonstrate that the
threshold belongs to the register width rather than to the scheme, not as
deployment targets. `neonsearch.h` is a set of ARM NEON reference kernels used
for cross-architecture comparison; there is no NEON Needle-Hammer, no NEON
guarded searcher, and the NEON kernels are not tuned, not fitted, and not
recommended for use. Builds on non-AVX-512 hardware will compile and run, but the
scheme they exercise is not the one these kernels are built around.

## What is here

- `benchmark/include/avx512search.h` — the AVX-512 kernels and the
  length-dispatched **Needle-Hammer** scheme, plus 256-bit (AVX2) and 128-bit
  (SSE2) builds of the same kernels over a traits struct, so one x86 binary
  measures all three register widths.
- `benchmark/include/neonsearch.h` — ARM NEON kernels, unsupported (see below).
- `benchmark/benchmarks/benchmark.cpp` — the driver. Modes: `synthetic`,
  `horspool`, `ashvardanian` (find-all), `worstcase`, `findall`.
- `benchmark/tests/` — validation against `std::string::find` across the
  needle- and haystack-length boundaries of every kernel.
- `benchmark/README.md` — the detailed measurements and design notes.

## Build and run

```sh
cmake -B benchmark/build -S benchmark -DCMAKE_BUILD_TYPE=Release
cmake --build benchmark/build -j
benchmark/build/benchmark horspool benchmark/data/43-0.txt
benchmark/build/test_search        # validation
```

## The scheme in one paragraph

`find_avx512_needle_hammer` dispatches on the needle length: a wide-stride
kernel that filters on the needle's first four bytes and narrows a mask over a
256-byte block for `m <= 512`, and a port of StringZilla's three-anchor
`sz_find` above that, with a guard routing short haystacks to a 64-byte kernel.
`find_avx512_needle_hammer_guarded` adds a work counter: it measures the work a
*failing* filter causes and hands the remainder to Crochemore-Perrin two-way once
that exceeds a budget proportional to the haystack, which bounds the worst case
without a needle-length argument. `benchmark/README.md` explains how the switch
point is chosen: the two criteria that bear on it, worst-case robustness and
benign throughput, disagree between Intel and AMD, so it minimises regret across
a range of parts rather than being optimal on any one machine.
