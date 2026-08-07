# simdsearch benchmark

SIMD substring search benchmarks for **AVX-512 capable processors**. The build
requires AVX-512F and AVX-512BW; there is no other backend.

`include/avx512search.h` carries the kernels -- `find_avx512`,
`find_avx512_256`, `find_avx512_stringzilla`, `find_avx512_needle_hammer`,
`find_avx512_stringzilla_256` -- along with 256-bit AVX2 (`find_avx256*`) and
128-bit SSE2 (`find_avx128*`) builds of the same kernels over a traits struct, so
one binary measures all three register widths. It also provides the portable
scalar and library searchers (`strstr`, `memmem`, `std::search` variants,
Boyer-Moore-Horspool) alongside the linear-time searchers in
`include/kmp_twoway.h`.

`find_memmem` is the C library's length-delimited `memmem`. It is the closest
library counterpart to the kernels here (no NUL terminator needed, so it is also
the fair baseline on binary data), but its speed is entirely a property of the
platform's libc: glibc runs a two-way variant with a bad-character table, while
Apple's libc uses a much simpler scan and lands well behind `strstr`.

## Linear-time searchers (`include/kmp_twoway.h`)

- `find_kmp` — Knuth–Morris–Pratt, with the "strong" failure links from the
  original paper (a text byte is never compared against a pattern byte already
  known to differ). O(m) preprocessing, at most 2n comparisons, never skips
  ahead — so it is slow on ordinary text and only pays off against adversarial
  input.
- `find_twoway` — the Crochemore–Perrin two-way algorithm: an O(m) critical
  factorization of the needle, then O(1) extra space and at most 2n comparisons.
- `find_twoway_bc` — two-way plus a 256-entry bad-character skip table, i.e. the
  variant glibc's `memmem`/`strstr` runs for needles longer than 32 bytes.
  Sublinear in practice, still linear in the worst case.

Each also has an `_amortized` row, where the preprocessing is built once per
needle and reused (the same split the `std::search` searchers already get).

The point of these three is the `worstcase` mode, where the filter-based and
naive searchers degrade to O(n·m) but the linear-time ones do not. KMP and
two-way are flat in the needle length -- that is the guarantee they buy -- and
several times slower than the SIMD kernels on ordinary text, which is the trade.

## Build and run

```
cmake -B build
cmake --build build
./build/benchmark <mode>      # synthetic | horspool | ashvardanian | worstcase | findall
```

On x86-64 the build enables `-mavx512f -mavx512bw -mavx512vl -mavx512dq`
automatically. Override the SIMD flags if needed, e.g. `-march=native`:

```
cmake -B build -DSIMDSEARCH_ARCH_FLAGS="-march=native"
```

Modes:

- `synthetic` — random 64 KiB haystack, 100k short needles (first-occurrence)
- `horspool` — random substrings of a source text (optional datafile)
- `ashvardanian` — StringWars-style find-all over a datafile (default
  `./data/43-0.txt` when cwd is `benchmark/`; pass an explicit path from the
  repo root)
- `worstcase` — adversarial haystack/needle shapes
- `findall` — overlapping find-all: first-match loop vs block enumerator

Synthetic and horspool draws use a fixed RNG seed (override with `--seed`).

## Tests

Correctness tests cross-check every searcher against `std::string::find` over
deterministic edge cases (alignment boundaries, all-equal runs, found/missing,
needle == haystack), the find-all enumerator, plus a seeded randomized fuzz
sweep:

```
ctest --test-dir build --output-on-failure
# or directly:
./build/test_search
```
