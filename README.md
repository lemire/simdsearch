# simdsearch

Benchmarks and reference kernels for SIMD substring search.

## What is here

- `benchmark/include/avx512search.h` — the AVX-512 kernels and the
  length-dispatched **Needle-Hammer** scheme, plus 256-bit (AVX2) and 128-bit
  (SSE2) builds of the same kernels over a traits struct, so one x86 binary
  measures all three register widths.
- `benchmark/include/neonsearch.h` — the ARM NEON kernels.
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
256-byte block for `m <= 256`, and a port of StringZilla's three-anchor
`sz_find` above that, with a guard routing short haystacks to a 64-byte kernel.
`find_avx512_needle_hammer_guarded` adds a work counter: it measures the work a
*failing* filter causes and hands the remainder to Crochemore-Perrin two-way once
that exceeds a budget proportional to the haystack, which bounds the worst case
without a needle-length argument. `benchmark/README.md` explains why the switch
point is 256 rather than the benign optimum.
