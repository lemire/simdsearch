
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

## Needle-hammer: wide-stride below a needle-length threshold, anchored above

`find_avx512_needle_hammer` runs `avx512_naive_search256` for needles up to 256
bytes and `avx512_stringzilla_find` above that. The two kernels fail in opposite
directions, and the threshold picks whichever failure is cheaper:

- the naive 256-byte kernel filters on the needle's **first four bytes** and
  narrows a match mask byte by byte, so its per-block cost grows with the needle
  length whenever those four bytes fail to reject — but when they do reject
  (ordinary text) it is the fastest kernel here, because each pattern-byte
  broadcast is amortized over four 64-byte chunks;
- the StringZilla kernel filters on **three anomaly-chosen anchors**, a fixed
  three compares per window regardless of needle length, then verifies each
  survivor. Verification is specialised, never a library call: no check at all
  for `m <= 3` (the anchors already covered every byte), one masked register
  compare against the preloaded needle for `m < 64`, and `sz_equal_avx512` above
  that. An earlier version of this port used a single `std::memcmp` instead,
  which cost up to 2.4x on adversarial input; see the paper's Appendix B.

`find_avx512_needle_hammer16/32/64/128/256/512` are the same scheme at other
switch points, so the crossover can be measured instead of assumed.

On ordinary text (`horspool` mode over the 141 KB `data/43-0.txt`, ns per
first-occurrence search, lower is better) needle-hammer tracks the lower envelope
of its two parents at every length:

```
algo                                      8     32    128    256    512   1024   2048   4096
find_avx512_256                        1958   2725   2798   3012   3425   4204   5796   8999
find_avx512_stringzilla                2034   2496   2428   2461   2464   2426   2350   2385
find_avx512_needle_hammer              1960   2725   2794   3014   2468   2429   2352   2392
find_avx512_needle_hammer_guarded      1959   2725   2804   3022   2470   2437   2355   2392
find_strstr                            4600   6084   6028   6124   6092   6192   6720   7200
```

### Why 256 and not the benign optimum

On this machine the two parents cross at about 9 bytes, so a benign-only choice
would switch almost immediately. 256 is deliberately higher, for two reasons that
only appear once you measure more than one machine and more than benign data:

- **Robustness.** Below the switch point an adversary faces the wide kernel;
  above it, the anchored one. The scheme's adversarial crossover with two-way is
  therefore `min(threshold, m*_wide)`, where `m*_wide` is the wide kernel's own
  crossover: 128/133/137 bytes on Emerald/Granite/Sapphire Rapids and 87/91 on
  Zen 5/Zen 4. Any threshold at or above 137 extracts all the robustness
  available; a lower one gives some away.
- **Portability.** Above that floor the choice is purely benign, and the vendors
  disagree by two orders of magnitude: the parents cross at 9-13 bytes on Intel
  but at 830-2300 on AMD. Scoring each candidate by its worst-case regret against
  the best choice per machine and length gives 1.82x at 16, 1.24x at 128, 1.10x
  at 256 and 1.11x at 512. 256 is the smallest round value that clears the
  robustness floor and ties for the best regret.

The 128-bit and 256-bit hammers keep their own switch points (8192 and 3072),
because the anchored kernel's fixed per-window cost is amortized over fewer
candidate positions as the register narrows.

Needle-hammer carries a **short-haystack guard** (`n < 2048` → the 64-byte
`avx512_naive_search`). Without it the scheme lost 3× on small inputs: the
256-byte kernel builds four chunk masks and only then scans them, so it cannot
report a match until it has finished the whole block the match sits in, while
the 64-byte kernel returns as soon as its single chunk has a hit. On a small
haystack the match arrives early relative to 256 bytes, so that granularity is
paid for nothing. In `synthetic` mode (1 KB text) the guard takes needle-hammer from
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
Below the threshold needle-hammer still inherits the naive kernel's O(n·m) collapse
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

### The same scheme at 256-bit and 128-bit

`find_avx256_needle_hammer` (AVX2) and `find_avx128_needle_hammer` (SSE2) rebuild
the whole scheme on narrower vectors: each width gets its own single-window naive
kernel, its own four-chunk wide kernel (128-byte and 64-byte blocks) and its own
three-anchor kernel, and the same length dispatch selects between them. The one
structural difference is where the match state lives — AVX-512 keeps it in an
`__mmask64` and folds "compare only where a candidate is still alive" into a
single instruction, while AVX2 and SSE2 have no mask registers and must build the
bitmap with `cmpeq` + `movemask` and narrow it with `&=`.

**The threshold belongs to the register width, not to the scheme.** Reusing
AVX-512's 512-byte switch point at the narrower widths is not merely suboptimal,
it is worse than not switching at all: the anchored kernel is much weaker down
here, so the crossover moves out sharply. `horspool` ns per search:

```
width  kernel              512   1024   2048   4096   8192   crossover
512b   naive wide         3450   4167   -      9011   -
512b   anchored           3674   3542   -      3594   -      ~512
256b   naive wide         4657   5507   7201  10404  16902
256b   anchored           9406   9137   9526   9375   9091    ~2500
128b   naive wide         6468   7149   8525  10957  15975
128b   anchored          17185  16885  17635  17144  16841    ~8000
```

The anchored kernel's per-window cost is three loads, three `movemask`
round-trips through a GPR and two early-out branches, all amortized over only W
candidate positions — so it grows as the width shrinks, while the naive wide
kernel pays no such fixed cost. The defaults are therefore the measured
crossovers: 512 at 512-bit, 2048 at 256-bit, 8192 at 128-bit.
`find_avx{256,128}_needle_hammer{64,512,8192}` expose the other switch points.

With those thresholds each hammer tracks the lower envelope of its own two
parents at its own width:

```
algo                          8    128   1024   4096   8192
find_avx512_needle_hammer  2021   2793   3602   3645   3717
find_avx256_needle_hammer  3055   4038   5523   8042   7848
find_avx128_needle_hammer  4680   5947   7154  10778  15847
```

Two caveats worth stating. First, a later switch means the adversarial
`worstcase` shapes stay on the naive kernel longer, so the O(n·m) collapse is not
cut off until the (now much higher) threshold — the same trade documented above,
and another reason the threshold is not an adversarial defence. Second, the
**short-haystack guard is an absolute 2048 bytes at every width, not a multiple
of the block size.** Scaling it per width (8 blocks → 2048/1024/512) was the
obvious guess and it measured wrong: on the 1 KB haystack of `synthetic` the wide
kernel loses at *every* width (AVX2 74.6 ns vs 50.3 for the single-window kernel;
SSE2 78.7 vs 65.3), so the cutoff has to sit above 1024 everywhere. The scaled
rule put it at 1024 and 512 — just below the haystack — and both narrow hammers
took the wide path and lost ~40%. With the absolute 2048 they come back to
52.5 ns and 66.3 ns respectively.

Width buys close to what you would expect on ordinary text — roughly 1.5× from
128-bit to 256-bit and 1.5× again to 512-bit at short needles — but the gap
widens with needle length, because the anchored kernel that carries the long-needle
end is precisely the one that degrades most as the register narrows.

## Bounding the worst case: `find_avx512_needle_hammer_guarded`

Needle-hammer is fast on real data and degrades to O(n*m) on inputs chosen to
defeat its filter. The usual remedy is a needle-length rule, but needle length
is only a proxy -- what decides is whether the filter's chosen bytes are
selective *in this haystack*, and that is observable at run time.

So each kernel counts the work a *failing* filter causes and hands the rest of
the haystack to Crochemore-Perrin once that count exceeds a budget proportional
to `n`. The wide kernel counts narrowing rounds past the four unrolled ones; the
anchored kernel counts bytes examined by its verifying compare. Both are ~0 on
ordinary text, so the budget is a bound on pathology rather than a throttle.
Two-way resumes at the give-up point, not at zero -- the kernel has already
proved there is no match below it.

`find_avx512_needle_hammer_guarded` ships the same 256-byte switch point as the
unguarded scheme, with a budget of `n/8` narrowing rounds and `16n` verification
bytes; `_tight` (`n/32`) and `_loose` (`n/2`) exist so the constant can be
measured rather than asserted.

Needles of at most 36 bytes skip the guard entirely: the narrowing loop runs at
most `m-4` times per 256-byte block, so `(n/256)(m-4)` rounds already fits the
budget whenever `m - 4 <= 256/8`. That bound is a compile-time constant, which is
why short needles -- the ones real workloads use -- pay nothing at all.

Benign cost, Xeon Gold 6548N / GCC 14.3 over `data/43-0.txt`:

```
ns per search                8       16       32       64      128      512     2048
find_avx512_needle_hammer      1887     2653     2617     2605     2480     2251     2279
..._guarded                    1892     2647     2620     2607     2481     2249     2277
find-all throughput: 11.07 GB/s unguarded, 10.94 GB/s guarded (-1.2%)
```

Worst case at the shipped 256-byte threshold (16 KB haystack, ns per search):

```
needle length                      32       64      128      512     2048
block  unguarded                 3743     6988    13140    55421   172653
block  guarded                   3741     6225     7665     8470    13368
tail   unguarded                 3743     6952    13177     1031     2618
tail   guarded                   3740    10330    14363     1032     2619
two-way (amortized, worst
  of the three shapes)          13437    13284    13211    12940    11690
```

The guard is free on `tail`, where the anchored kernel already copes, and buys a
15x bound on `block`, where it does not. Taking each searcher's worst shape at
each length, the guarded scheme stays within about 1.25x of two-way everywhere,
which the unguarded scheme does not do at any threshold. Residual growth in `m`
is two-way's own O(m) preprocessing under a stateless API, not the guard:
non-amortized `find_twoway` grows the same way on its own.

### One body, one instantiation

`avx512_stringzilla_body` is shared by `avx512_stringzilla_find` (which passes an
unreachable budget) and `avx512_stringzilla_guarded`. That is not tidiness. An
earlier version kept two copies, and a later one templated the body on whether
the counter was compiled in; both produced two machine-code bodies, and GCC
generated markedly better code for the guarded one -- 2251 ns against 3621 ns at
`m = 2048` for identical work, with budgets set so that neither could bail.

Collapsing to a single instantiation makes guarded and unguarded comparable by
construction, and incidentally hands the anchored kernel the faster code path:
`find_avx512_stringzilla` went from 3624 ns to 2223 ns at `m = 2048`.

**That fix moved the switch point, and the default is now 128 rather than 512.**
With the anchored kernel no longer paying the penalty it becomes the cheaper
parent from about `m = 24`:

```
ns per search              16       24       32       40       64       96      256      512
find_avx512_256          2615     2592     2620     2657     2697     2739     3015     3445
find_avx512_stringzilla  2611     2552     2606     2611     2577     2307     2367     2241
```

The two parents are within 1% of each other over a wide band, so the exact value
matters little on benign data -- the benign optimum is around 8.

**But 8 is not what ships, and the reason is the next section.** At 512-bit the
switch point does more than pick the faster kernel: it decides which of the two
an adversary is allowed to attack. 128 is the largest needle length at which the
wide kernel is still ahead of two-way under its own worst case, so putting the
threshold there gives away nothing exploitable, at a cost of up to ~25% on benign
text at intermediate lengths. Measured, a 16-byte threshold would drop the
adversarial crossover from ~126 bytes to ~28.

### The switch point trades benign speed against robustness

Lowering it hands more needle lengths to the anchored kernel, and the
all-common-byte `block` adversary defeats exactly that kernel. The two criteria
therefore disagree, and neither threshold is right for both. Worst case over the
three adversaries, 16 KB haystack, ns per search:

```
m                            32      64     128     512    2048
threshold 128 (shipped)    3707    6989   13177   55444  172580
threshold 16               3707   46917   54292   87714  205853
guarded, threshold 128     3707   10327   14370    8486   13386
two-way (amortized)       13318   13296   13432   13204   11622
```

On benign data a low threshold is right; on adversarial data a high one is; and
no threshold is right at large `m`, where both parents collapse. The guarded
searcher is what actually resolves this -- it bounds both kernels instead of
hoping the threshold avoids them, and it stays within about 1.25x of two-way's
own worst case at every length while keeping the benign optimum. **If the
worst-case behaviour matters at all, prefer
`find_avx512_needle_hammer_guarded` over tuning the threshold.**

## Capping the needle count: `--needles`

`ashvardanian` mode scans the whole haystack once per needle and repeats each
timed pass at least ten times, so a cell costs `needles * n * repeats`. At 256 MB
that is ~2.5 TB of scanning per cell -- minutes for the vector kernels, hours for
the classical ones -- which is what kept the haystack sweep inside the last-level
cache.

```
./build/benchmark ashvardanian data/43-0.txt --needles 50
```

Lower caps make large haystacks tractable, but they change the *workload* as well
as its cost: the cap keeps the first N word tokens, which are shorter and commoner
than the tail of the list. Compare across algorithms at one setting, never across
settings.

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
The AVX-512 routines lead the field (the 256-bit and 128-bit needle-hammers are
built in the same binary for comparison, not because they win). Which one is fastest depends on the needle
length and on whether the filter's chosen bytes are selective in the haystack:
`find_avx512_256` wins on short needles, `find_avx512_needle_hammer` extends that win to
long needles on ordinary text, and `find_avx512_stringzilla_256` is fastest from
length ~32 up and is the only kernel that stays flat under the adversarial
`worstcase` shapes (except `block`). See the needle-hammer section above for the
measurements.
