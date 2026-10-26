# Design of the Rust port

The port follows `include/simdsearch/{guardzilla,needlehammer}.h` (upstream
commit `6a1c3fd`) step for step. This file records where and why the Rust
code differs in form.

## Modules

| Module | Role | C++ counterpart |
|---|---|---|
| `simd` | The only module with `unsafe`: safe wrappers over the AVX-512 intrinsics | `<immintrin.h>` calls |
| `anchors` | StringZilla's three anchors, evenly spaced anchors, per-anchor block lanes | `sz_anchors`, `anchors` |
| `verify` | `same_prefix`, `same_suffix`, `rest_matches`, the guarded candidate check | the `verify` lambdas |
| `byte` | Single-byte search for `m == 1` | `std::memchr` |
| `twoway` | Crochemore-Perrin two-way | `detail::twoway` |
| `guardzilla` | Guardzilla | `guardzilla::find` |
| `needlehammer` | NeedleHammer | `needlehammer::find` |

## Unsafe boundary

- The workspace denies `unsafe_code`; `simd.rs` alone allows it.
- Intrinsics that touch no memory are wrapped as plain safe functions. Rust
  marks them `#[target_feature]`, so a call from a function without that
  attribute needs `unsafe`, even when the feature is enabled crate-wide. Their
  safety rests on `lib.rs`: the crate does not compile unless the target
  enables AVX-512F and BW, and `needlehammer` (with the wrappers it needs) is
  compiled only when VBMI2 and VL are enabled too. This is the C++ model
  (`#error` without the flags, no runtime dispatch).
- Wrappers that touch memory take their bounds from slices:
  - `load(&[u8; 64])`: the type guarantees 64 readable bytes.
  - `load_masked(&[u8], mask)`: the mask is cut to `min(64, len)`.
  - `store`, `store_u32`, `load_masked_16`: index a sub-slice of the exact size first.
  - `gather_u32`: lanes whose offset would read past the slice are masked off
    before the gather. Valid positions are never masked, so results are the
    same as in C++; one extra compare per 16 positions.

## Mapping C++ idioms

- **Return codes to an enum.** C++ encodes results as `n`, `q + 1` or
  `n + 1 + q`. Rust returns `Option<Outcome>` with `Outcome::Match(q)` and
  `Outcome::Exhausted(q)`; `Exhausted(q)` restarts two-way at `q`, as C++ does.
- **Settings macros to traits.** `GUARDZILLA_*` and `NEEDLEHAMMER_*` become
  associated constants of `guardzilla::Settings` and `needlehammer::Settings`;
  `Defaults` is the published searcher. `find_with::<S>` monomorphizes per
  setting, as the C++ does per `-D` flag, and `const` asserts replace
  `static_assert`. The tests run every setting that upstream's ctest runs.
- **Byte indices to block lanes.** C++ loads `t + i + o[j]`. Rust slices, once
  per scan, one lane per anchor, `hay[start + o[j] .. start + o[j] + 64 *
  blocks]` viewed as `&[[u8; 64]]` (`as_chunks`), and indexes lanes by block.
  Every lane has the same length, so LLVM removes the bounds checks from the
  hot loops (checked in the disassembly). Windows and chunks of NeedleHammer are
  counted in blocks; `full`, `end` and the timed block count reproduce the C++
  byte arithmetic, including `last - 62` for the timed comparison.
- **NeedleHammer's tail.** When the second scan reaches the end, C++ resumes
  at the byte `last - 62` and Rust at the first unscanned block,
  `start + 64 * blocks`. Every candidate between the two has already been
  checked by the second scan, so the answer is the same.
- **Second-scan buffers.** C++ keeps `pos8` and `pos32` (21 KB) uninitialized on
  the stack of every chunk. Rust allocates them zeroed on the heap once per
  entry into the second scan: zeroing per chunk would cost more than the chunk,
  and `MaybeUninit` would add unsafe outside `simd`.
- **`m == 1`.** C++ calls `std::memchr`. Rust uses `byte::find`, an aligned
  AVX-512 scan, because `core` exposes no `memchr` and the library has no
  dependencies.
- **Out-of-line functions.** `rest_matches`, `second_scan` and
  `compress_chunk` are `#[inline(never)]`, as in C++.

## Performance

- `next_candidates` scans blocks (`find_map`) and returns the next block with
  candidates; verification runs in the caller. The scan loop holds only the
  lanes, the anchor vectors and the block counter.
- `rest_matches` returns its count of 64-byte compares instead of taking
  `&mut work`, so no pointer into the `Verifier` escapes and its fields stay in
  registers.
- The candidate head is sliced to exactly `min(m, 64)` bytes, so the clamp in
  `simd::load_masked` is loop-invariant.
- `anchors::lanes` slices only the lanes in use.
- Timings vary by up to 30% between processes, in both languages.
  `rust/scripts/run_benchmarks.sh` runs each binary several times on one
  pinned core; `rust/scripts/compare.py` takes the median per cell.

## Validation

1. `rust/tests`: upstream's test families against `memchr::memmem::find`, for
   every setting upstream tests, plus mutation-checked coverage of the second
   scan (eager settings switch to it at once).
2. The benchmark: `rust/compare` reproduces upstream's `search_bench.cpp`,
   including its needles (`mt19937_64` and libstdc++'s
   `uniform_int_distribution`). Each binary checks every answer before timing,
   and `rust/scripts/compare.py` checks that both languages measured the same
   needles in every cell.
