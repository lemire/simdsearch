
# simdsearch benchmark

SIMD substring search benchmarks. The SIMD backend is selected at compile time
from the host architecture:

- **x86-64 with AVX-512** (F + BW): `include/avx512search.h`
  (`find_avx512`, `find_avx512_256`, `find_avx512_stringzilla`)
- **ARM / AArch64 NEON**: `include/neonsearch.h`
  (`find_neon`, `find_neon64`, `find_neon_stringzilla`)

Both headers also provide the portable scalar/library searchers (`strstr`,
`std::search` variants, Boyer–Moore–Horspool), so the table is identical across
architectures apart from the SIMD rows.

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
