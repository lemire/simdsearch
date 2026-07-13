
# simdsearch benchmark

SIMD substring search benchmarks. The SIMD backend is selected at compile time
from the host architecture:

- **x86-64 with AVX-512** (F + BW): `include/avx512search.h`
  (`find_avx512`, `find_avx512_256`, `find_avx512_stringzilla`)
- **ARM / AArch64 NEON**: `include/neonsearch.h`
  (`find_neon`, `find_neon64`, `find_neon_stringzilla`)

Both headers also provide the portable scalar/library searchers (`strstr`,
`std::search` variants, Boyer–Moore–Horspool, and the linear-time searchers in
`include/kmp_twoway.h`), so the table is identical across architectures apart
from the SIMD rows.

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
naive searchers degrade to O(n·m) but the linear-time ones do not. Needle `'a'*L`
against a haystack of `('a'*(L-1) + 'b')` repeated, 64 KB, ns per full-haystack
search (Apple M-series, clang):

```
algo                        L=8      L=32     L=128     L=512
find_classic             335291    364834    399792    510500
find_neon                  6779     27500    139875    515250
find_neon_stringzilla     58125     76250    141666    371500
find_strstr              114333    222417    320167    511500
find_kmp                  30833     29916     32166     30000
find_twoway               16358     20662     18570     18416
find_twoway_bc             2412       703       482      1236
```

KMP and two-way are flat in `L` — that is the guarantee they buy. On ordinary
text (`horspool`, `ashvardanian` modes) they are several times slower than the
SIMD kernels, which is the trade.

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

## AVX-512 results (Intel Xeon Gold 6548N, GCC 14.3)

`ashvardanian` mode (StringWars-style forward find-all over a 141 KB text,
GB/s of haystack scanned, higher is better):

```
find_classic                  2.534 GB/s
find_avx512                   8.936 GB/s
find_avx512_stringzilla       9.392 GB/s
find_strstr                   7.632 GB/s
find_bmh                      0.833 GB/s
```

## Glob pattern-matching benchmark

`glob_benchmark` compares glob ("stringmatchlen") matchers: the recursive Redis
baseline against the `memchr`/`memmem`-accelerated fast path from Chris
Pretorius's [barch](https://github.com/tjizep/barch) project (`src/glob.cpp`).
The recursive baseline retries the rest of the pattern at every text position
for a pattern like `*needle*` (O(n·m)); barch's fast path advances a leading
`*literal` with `memmem`/`memchr`, skipping across the haystack instead.

```
cmake -B build
cmake --build build
./build/glob_benchmark            # defaults to ./data/43-0.txt
./build/glob_benchmark --list     # list matcher names
```

Matchers (`include/globmatch.h`):

- `glob_redis` — the upstream Redis `stringmatchlen` (recursive baseline).
- `glob_barch_general` — barch's faithful copy of that general matcher.
- `glob_barch_asterisk` — barch's `memchr`/`memmem` fast path.
- `glob_barch` — barch's dispatcher (fast path for star-only patterns).

Options: `--algos x,y`, `--count N`, `--minlen N`, `--maxlen N`,
`--absent-ratio f`. The benchmark generates `*literal*` patterns (half cut from
the text, half random/absent) and matches each against the whole file. Matching
is case-sensitive: barch's fast path compares raw bytes.

The fast path is correct only on its target domain — case-sensitive, no `[...]`
character classes, and no `?` immediately after a `*` (an upstream bug). The
dispatcher and this benchmark stay within that domain; `tests/test_glob.cpp`
cross-checks every matcher against an independent oracle and the Redis baseline.

```
ctest --test-dir build --output-on-failure   # or: ./build/test_glob
```
The AVX-512 routines lead the field; `find_avx512_stringzilla` (first+last byte
anchoring) is fastest. `find_avx512_256` uses a 256-byte stride and only helps
when matches are rare and the scan streams through long non-matching runs (e.g.
`horspool` mode at pattern length ≥ 4); for short or frequently-found needles
prefer `find_avx512` or `find_avx512_stringzilla`.
