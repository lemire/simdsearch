# simdsearch benchmark

SIMD substring search benchmarks. Two backends, selected by the host
architecture: **AVX-512** (requires AVX-512F and AVX-512BW) and **AArch64
NEON**. A build on anything else stops at a `#error`.

`include/needle_hammer.h` is Needle-Hammer, the featured searcher
(`find_avx512_needle_hammer`, and `find_avx512_needle_hammer_unguarded` for
measuring the guard): a single wide kernel that tests every candidate position
on three or four needle bytes chosen for selectivity -- first, middle and last
byte, plus a quarter point or, when the needle has one, a byte that is rare in
it -- 256 positions per iteration, then narrows the survivors byte by byte.
Three anchors are used while the haystack shows they suffice and a fourth is
added once survivors become frequent; needles of one to three bytes take a
dedicated loop with no verification at all; and a work counter bounds the
narrowing, resuming with the vectorized two-way in `include/twoway_simd.h`
(`find_twoway_simd`) so every input is searched in linear time. There is no
needle-length threshold to fit per machine. The constants are documented in
the header; `NH2_THREE_ANCHOR_MAX` can be set at compile time. AVX-512 only
for now: the NEON backend still carries the previous scheme.

`include/avx512search.h` carries the x86 component kernels the featured one is
read against -- `find_avx512` (single-window naive), `find_avx512_256`
(256-byte-stride naive, the loop Needle-Hammer's is built on),
`find_avx512_stringzilla` (three-anchor filter, StringZilla's design) and
`find_avx512_stringzilla_256` -- along with 256-bit AVX2 (`find_avx256*`) and
128-bit SSE2 (`find_avx128*`) builds of the same kernels over a traits struct,
so one binary measures all three register widths.

`include/neonsearch.h` is the previous scheme at 128-bit NEON width: `find_neon`
(single-window), `find_neon_64` (wide), `find_neon_stringzilla` (anchored),
`find_neon_needle_hammer` (the length-dispatched pair) and its work-counting
`_guarded` variants. Both of its constants are template parameters, with
`find_neon_nh_t*` and `find_neon_nh_m*` sweep instances, because they have to be
fitted across ARM parts rather than read off one machine -- see
`NEON_NH_TAU`/`NEON_NH_MU` in the header.

`include/common_search.h` holds what the two backends share: the three-anchor
selector and the portable scalar and library searchers (`strstr`, `memmem`,
`std::search` variants, Boyer-Moore-Horspool), alongside the linear-time
searchers in `include/kmp_twoway.h`.

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
automatically. On AArch64 NEON is architectural and no flag is needed. Override
the SIMD flags if desired, e.g. `-march=native`:

```
cmake -B build -DSIMDSEARCH_ARCH_FLAGS="-march=native"
```

The NEON backend's two constants can be overridden at configure time, which is
how the sweep harness re-fits them per machine without editing the header:

```
cmake -B build -DNEON_NH_TAU=64 -DNEON_NH_MU=1024
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
