# simdsearch

Benchmarks and reference kernels for SIMD substring search.

## Requirements

**An x86-64 processor with AVX-512, specifically AVX-512F and AVX-512BW, running
Linux with GCC 14 or newer.** That is the only supported configuration: it is
what Needle-Hammer is designed for and what the switch points are fitted to.

There is no other backend: a build without AVX-512F and AVX-512BW stops at a
`#error`. The AVX2 and SSE2 kernels in the same header exist to show that the
switch point belongs to the register width rather than to the scheme, not as
deployment targets.

## What is here

- `benchmark/include/avx512search.h` — the AVX-512 kernels and the
  length-dispatched **Needle-Hammer** scheme, plus 256-bit (AVX2) and 128-bit
  (SSE2) builds of the same kernels over a traits struct, so one x86 binary
  measures all three register widths.
- `benchmark/benchmarks/benchmark.cpp` — the driver. Modes: `synthetic`,
  `horspool`, `ashvardanian` (find-all), `worstcase`, `findall`.
- `benchmark/tests/` — validation against `std::string::find` across the
  needle- and haystack-length boundaries of every kernel.
- `benchmark/README.md` — the algorithm list, build options and test notes.

## Build and run

```sh
cmake -B benchmark/build -S benchmark -DCMAKE_BUILD_TYPE=Release
cmake --build benchmark/build -j
benchmark/build/benchmark horspool benchmark/data/43-0.txt
benchmark/build/test_search        # validation
```
