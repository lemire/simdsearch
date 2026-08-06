# simdsearch benchmark

SIMD substring search benchmarks. The SIMD backend is selected at compile time
from the host architecture:

- **x86-64 with AVX-512** (F + BW): `include/avx512search.h`
  (`find_avx512`, `find_avx512_256`, `find_avx512_stringzilla`,
  `find_avx512_needle_hammer`, `find_avx512_stringzilla_256`). The same header
  also carries the 256-bit AVX2 (`find_avx256*`) and 128-bit SSE2
  (`find_avx128*`) kernels, so one x86 build measures all three widths.
- **ARM / AArch64 NEON**: `include/neonsearch.h`
  (`find_neon`, `find_neon64`, `find_neon_stringzilla`)

Both headers also provide the portable scalar/library searchers (`strstr`,
`memmem`, `std::search` variants, Boyer–Moore–Horspool, and the linear-time
searchers in `include/kmp_twoway.h`), so the table is identical across
architectures apart from the SIMD rows.

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
./build/benchmark <mode>      # synthetic | horspool | ashvardanian
```

On x86-64 the build enables `-mavx512f -mavx512bw -mavx512vl -mavx512dq`
automatically. Override the SIMD flags if needed, e.g. `-march=native`:

```
cmake -B build -DSIMDSEARCH_ARCH_FLAGS="-march=native"
```

## Tests

Correctness tests cross-check every searcher against `std::string::find` over
deterministic edge cases (alignment boundaries, all-equal runs, found/missing,
needle == haystack) plus a seeded randomized fuzz sweep:

```
ctest --test-dir build --output-on-failure
# or directly:
./build/test_search
```
