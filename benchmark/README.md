
# simdsearch benchmark

SIMD substring search benchmarks. The SIMD backend is selected at compile time
from the host architecture:

- **x86-64 with AVX-512** (F + BW): `include/avx512search.h`
  (`find_avx512`, `find_avx512_256`, `find_avx512_stringzilla`,
  `find_avx512_hybrid`, `find_avx512_stringzilla_256`)
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

## Hybrid: 256-byte stride below a needle-length threshold, anchored above

`find_avx512_hybrid` runs `avx512_naive_search256` for needles up to 512 bytes
and `avx512_stringzilla_find` above that. The two kernels fail in opposite
directions, and the threshold picks whichever failure is cheaper:

- the naive 256-byte kernel filters on the needle's **first four bytes** and
  narrows a match mask byte by byte, so its per-block cost grows with the needle
  length whenever those four bytes fail to reject — but when they do reject
  (ordinary text) it is the fastest kernel here, because each pattern-byte
  broadcast is amortized over four 64-byte chunks;
- the StringZilla kernel filters on **three anomaly-chosen anchors**, a fixed
  three compares per window regardless of needle length, then verifies survivors
  with `memcmp`.

`find_avx512_hybrid16/32/64/128/256` are the same scheme at other switch points,
so the crossover can be measured instead of assumed.

On ordinary text (`horspool` mode over the 141 KB `data/43-0.txt`, ns per
first-occurrence search, lower is better) the hybrid tracks the lower envelope of
its two parents at every length, and 512 is close to the true crossover — the
parents cross between 512 and 640:

```
algo                          8     32    128    256    512    640   1024   2048   4096
find_avx512                2560   3491   3571   3578   3823   4089   4370   5190   7007
find_avx512_256            1933   2670   2816   2970   3339   3679   4238   5814   9026
find_avx512_stringzilla    3163   3991   3758   3600   3520   3645   3643   3581   3700
find_avx512_hybrid         1935   2674   2823   2970   3336   3646   3649   3587   3701
find_avx512_hybrid64       1939   2673   3762   3601   3518   3643   3641   3585   3698
find_avx512_stringzilla_256 2231  2756   2741   2721   2668   2752   2773   2789   2826
find_strstr                4554   6012   6073   6007   5953   6363   6390   6636   7338
```

Switching earlier is worse: `find_avx512_hybrid64` gives up 33% at length 128
(3762 vs 2820) for no gain. In `ashvardanian` mode every needle is a short word,
so the length switch never fires; the hybrid leads anyway (11.20 vs 11.11 GB/s
for `find_avx512_256`) on the strength of the short-haystack guard below.

The hybrid carries a **short-haystack guard** (`n < 2048` → the 64-byte
`avx512_naive_search`). Without it the scheme lost 3× on small inputs: the
256-byte kernel builds four chunk masks and only then scans them, so it cannot
report a match until it has finished the whole block the match sits in, while
the 64-byte kernel returns as soon as its single chunk has a hit. On a small
haystack the match arrives early relative to 256 bytes, so that granularity is
paid for nothing. In `synthetic` mode (1 KB text) the guard takes the hybrid from
133–159 ns/search to 43–54 ns, level with `find_avx512`. It also helps
`ashvardanian`, whose find-all loop searches a shrinking suffix and so keeps
entering the small-haystack regime: 9.91 → 11.11 GB/s in the same run. It is
free everywhere else — `horspool` (141 KB) and `worstcase` (64 KB) never trigger
it. A 8192-byte cutoff measures the same as 2048.

Note that the narrower structural condition (`n < m + 255`, i.e. too short for
even one wide block) is *not* what mattered here: in `synthetic`, n = 1024 and
m ≤ 20, so `n >= m + 255` always holds and that guard alone never fires.

The threshold does **not** rescue the adversarial `worstcase` inputs, because
needle length is only a proxy for the thing that matters — whether the filter's
chosen bytes are selective in this haystack — and the proxy breaks both ways.
Below the threshold the hybrid still inherits the naive kernel's O(n·m) collapse
(`tail`, L=512: 205817 ns vs StringZilla's 3920, 52× worse), and in the `block`
shape StringZilla is the *slower* parent, so switching to it above 512 costs
~1.9× (L=1024: 532420 vs 256614 ns). Fixing the adversarial case needs a better
filter, not a better threshold.

`find_avx512_stringzilla_256` is that filter: StringZilla's three anchors run at
the naive kernel's 256-byte stride, so each anchor broadcast is reused across
four 64-byte chunks and the mask bookkeeping is paid once per 256 bytes instead
of once per 64. It needs no threshold — the anchor cost is already independent of
m — and it is the fastest kernel in `horspool` from length 32 up and in every
`worstcase` shape except `block`, where no anchor is selective by construction
and only the linear-time `find_twoway_bc` does well.

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
The AVX-512 routines lead the field. Which one is fastest depends on the needle
length and on whether the filter's chosen bytes are selective in the haystack:
`find_avx512_256` wins on short needles, `find_avx512_hybrid` extends that win to
long needles on ordinary text, and `find_avx512_stringzilla_256` is fastest from
length ~32 up and is the only kernel that stays flat under the adversarial
`worstcase` shapes (except `block`). See the hybrid section above for the
measurements.
