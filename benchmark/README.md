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

## Needle-hammer: wide-stride below a needle-length threshold, anchored above

`find_avx512_needle_hammer` runs `avx512_naive_search256` for needles up to 512
bytes and `avx512_stringzilla_find` above that. The two kernels fail in opposite
directions, and the threshold picks whichever failure is cheaper:

- the naive 256-byte kernel filters on the needle's **first four bytes** and
  narrows a match mask byte by byte, so its per-block cost grows with the needle
  length whenever those four bytes fail to reject -- but when they do reject
  (ordinary text) it is the fastest kernel here, because each pattern-byte
  broadcast is amortized over four 64-byte chunks;
- the StringZilla kernel filters on **three anomaly-chosen anchors**, a fixed
  three compares per window regardless of needle length, then verifies each
  survivor. Verification is specialised, never a library call: no check at all
  for `m <= 3` (the anchors already covered every byte), one masked register
  compare against the preloaded needle for `m < 64`, and `sz_equal_avx512` above
  that. A single `std::memcmp` instead is much worse on adversarial input, where
  a filter-defeating haystack makes every position a candidate and multiplies the
  per-candidate constant by `n`.

`find_avx512_needle_hammer16/32/64/128/256/512` are the same scheme at other
switch points, so the crossover can be measured rather than assumed.

### The switch point is a fleet minimax, not a per-machine optimum

Two criteria bear on the threshold, and they disagree.

Robustness bounds it from below. A switch point protects the adversarial
crossover only while it sits below the wide kernel's own crossover with two-way,
so there is a floor below which the scheme gives away robustness it could have
had for free.

Above that floor the choice is purely benign, and there the vendors pull apart:
the anchored kernel is far weaker relative to the wide one on AMD than on Intel,
so Intel wants a small threshold and AMD a large one, and no single constant is
optimal for both. The shipped 512 is the value that clears the robustness floor
and minimises worst-case regret across the machines we measured -- deliberately
not the optimum for any one of them. Re-derive it on new hardware with the
`thresholds` mode rather than trusting it.

Needle-hammer also carries a **short-haystack guard** (`n < 1024` uses the
64-byte kernel). The wide kernel builds four chunk masks and only then scans
them, so it cannot report a match until it has finished the whole 256-byte block
the match sits in, while the 64-byte kernel returns as soon as its single chunk
has a hit; on a small haystack the match arrives early relative to 256 bytes and
that granularity is paid for nothing. Note that the narrower structural condition
(`n < m + 255`, too short for even one wide block) is *not* what matters here --
it does not fire in the regime the guard is for.

The threshold does **not** rescue the adversarial `worstcase` inputs. Needle
length is only a proxy for the thing that decides cost -- whether the filter's
chosen bytes are selective in this haystack -- and the proxy breaks both ways:
below the threshold the scheme inherits the naive kernel's O(n·m) collapse on
`tail`, and in the `block` shape the anchored kernel is the *slower* parent, so
switching to it above the threshold costs. Fixing the adversarial case needs a
better filter or a run-time bound, not a better threshold.

### The same scheme at 256-bit and 128-bit

`find_avx256_needle_hammer` (AVX2) and `find_avx128_needle_hammer` (SSE2) rebuild
the whole scheme on narrower vectors: each width gets its own single-window naive
kernel, its own four-chunk wide kernel (128-byte and 64-byte blocks) and its own
three-anchor kernel, and the same length dispatch selects between them. The one
structural difference is where the match state lives -- AVX-512 keeps it in an
`__mmask64` and folds "compare only where a candidate is still alive" into a
single instruction, while AVX2 and SSE2 have no mask registers and must build the
bitmap with `cmpeq` + `movemask` and narrow it with `&=`.

**The threshold belongs to the register width, not to the scheme.** The anchored
kernel's per-window cost -- three loads, three `movemask` round-trips through a
GPR, two early-out branches -- is amortized over only W candidate positions, so
it grows as the width shrinks, while the naive wide kernel pays no such fixed
cost. Reusing AVX-512's 512-byte switch point at the narrower widths is therefore
worse than not switching at all, and the defaults move out sharply: 512 at
512-bit, 2048 at 256-bit, 8192 at 128-bit.
`find_avx{256,128}_needle_hammer{64,512,8192}` expose the other switch points.

Two caveats. A later switch means the adversarial shapes stay on the naive kernel
longer, so the O(n·m) collapse is not cut off until the much higher threshold.
And the **short-haystack guard is absolute at every width, not a multiple of the
block size**: scaling it per width is the obvious guess and it is wrong, because
the wide kernel loses on a small haystack at *every* width, so the cutoff has to
sit above that haystack size everywhere rather than shrinking with the block.

## Bounding the worst case: `find_avx512_needle_hammer_guarded`

Needle-hammer is fast on real data and degrades to O(n*m) on inputs chosen to
defeat its filter. The usual remedy is a needle-length rule, but needle length is
only a proxy -- what decides is whether the filter's chosen bytes are selective
*in this haystack*, and that is observable at run time.

So each kernel counts the work a *failing* filter causes and hands the rest of
the haystack to Crochemore-Perrin once that count exceeds a budget proportional
to `n`. The wide kernel counts narrowing rounds past the four unrolled ones; the
anchored kernel counts bytes examined by its verifying compare. Both are ~0 on
ordinary text, so the budget bounds pathology rather than throttling normal work.
Two-way resumes at the give-up point, not at zero -- the kernel has already
proved there is no match below it.

`find_avx512_needle_hammer_guarded` ships the same 512-byte switch point as the
unguarded scheme, with a budget of `n/8` narrowing rounds and `16n` verification
bytes; `_tight` (`n/32`) and `_loose` (`n/2`) exist so the constant can be
measured rather than asserted.

Needles of at most 36 bytes skip the guard entirely: the narrowing loop runs at
most `m-4` times per 256-byte block, so `(n/256)(m-4)` rounds already fits the
budget whenever `m - 4 <= 256/8`. That bound is a compile-time constant, which is
why short needles -- the ones real workloads use -- pay nothing at all.

The guard is not free above that: it can give up on inputs where the unguarded
kernel would still have beaten two-way, and pay for the restart. What it buys is
that the worst case stops growing. Unguarded, no threshold bounds it at any
length; guarded, it stays near two-way's own cost. **If worst-case behaviour
matters at all, prefer `find_avx512_needle_hammer_guarded` over tuning the
threshold.**

### One body, one instantiation

`avx512_stringzilla_body` is shared by `avx512_stringzilla_find` (which passes an
unreachable budget) and `avx512_stringzilla_guarded`. That is not tidiness.
Either keeping two copies or templating the body on whether the counter is
compiled in produces two machine-code bodies, and GCC generates markedly better
code for the guarded one -- enough to swamp the guard's actual cost. A single
instantiation makes guarded and unguarded comparable by construction rather than
by coincidence.

## Where the measurements are

This repository is the benchmark harness, not the results. Numbers depend on the
processor, the libc, the compiler and the corpus, so quoting any here would date
the file and invite comparison across machines that were never comparable. Run
the modes above on the hardware you care about. The published measurements and
the analysis behind the constants chosen here are in the accompanying paper.

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
