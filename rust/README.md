# simdsearch-rs

A Rust port of [lemire/simdsearch](https://github.com/lemire/simdsearch)
(commit `6a1c3fd`): AVX-512 substring search in O(n + m) time.

- **Guardzilla**: StringZilla's three-anchor scan, aligned to 64-byte blocks
  and guarded by a work budget that hands over to two-way.
- **NeedleHammer**: Guardzilla plus a branch-free second scan (VPCOMPRESSB)
  for inputs that defeat the branch predictor, such as DNA.

| Path | Content |
|---|---|
| `/` (repository root) | The C++ original, unchanged |
| `rust/` | The port (`simdsearch` crate) and its tests; design in [`DESIGN.md`](DESIGN.md) |
| `rust/compare/` | The Rust benchmark, the counterpart of upstream's `search_bench.cpp` |
| `rust/scripts/` | `run_benchmarks.sh` (both benchmarks), `compare.py` (tables) |
| `rust/results/` | The CSVs and tables below |

## Use

```rust
let hay = b"The quick brown fox jumps over the lazy dog";
assert_eq!(simdsearch::find(hay, b"fox"), Some(16));               // Guardzilla
assert_eq!(simdsearch::needlehammer::find(hay, b"fox"), Some(16));
```

The empty needle matches at 0. No byte outside either slice is read.
`guardzilla::find_with::<S>` and `needlehammer::find_with::<S>` take upstream's
compile-time settings (`GUARDZILLA_*`, `NEEDLEHAMMER_*`) as a `Settings` impl.

Requirements: x86-64 with AVX-512F and BW (NeedleHammer also VBMI2 and VL),
chosen at compile time as in upstream. `rust/.cargo/config.toml` sets
`-C target-cpu=native`. `unsafe` is confined to `rust/src/simd.rs`, the
AVX-512 wrappers; the rest of the crate denies it.

Commands below run from the repository root.

## Test

```sh
cd rust && cargo test --release
```

The tests are upstream's test families (random, every alignment, adversarial,
exact-size buffers) against `memchr::memmem::find`, for every setting that
upstream's ctest builds.

## Benchmark

```sh
python3 benchmark/tools/corpora.py --out corpora         # upstream's corpora
cmake -S . -B build/cpp -DCMAKE_CXX_COMPILER=clang++ -DSIMDSEARCH_ARCH_FLAGS=-march=native
cmake --build build/cpp --target search_bench_guardzilla search_bench_needlehammer
(cd rust && cargo build --release)
rust/scripts/run_benchmarks.sh rust/results 2 5          # core 2, 5 runs per binary
python3 rust/scripts/compare.py rust/results
```

The C++ side is upstream's `search_bench` as shipped, one executable per
searcher. The Rust side (`rust/compare`) reproduces it: same modes, options,
adversarial families and CSV columns. It also draws the same needles:
`mt19937_64` and libstdc++'s `uniform_int_distribution` are ported, and
`compare.py` checks that every cell has the same `bytes_per_search` in both
languages. Every answer is checked before it is timed. The Rust CSV leaves the
four hardware-counter columns `nan`.

Within a process each cell is the best of 10 timed passes, and that is
stable. Between processes the same cell varies by up to 30%, in both
languages. So every binary runs 5 times and `compare.py` reports the median
per cell.

Setup: Intel Xeon Platinum 8488C (Sapphire Rapids), pinned to core 2. C++:
clang 23.1 with libstdc++ 11, `-march=native`. Rust: rustc 1.99,
`-C target-cpu=native`. The run takes 5.6 minutes. 280 cells per language,
same needles in all. Every cell is in
[`results/comparison_full.md`](results/comparison_full.md).

### Rust / C++ time ratio, median of 5 runs (below 1: Rust faster)

| Algorithm | Workload | Cells | Rust / C++ time (geomean) | Best | Worst |
|---|---|---:|---:|---:|---:|
| Guardzilla | corpora | 95 | 0.564 | 0.46 | 0.95 |
| Guardzilla | adversarial | 45 | 0.680 | 0.46 | 1.24 |
| NeedleHammer | corpora | 95 | 0.892 | 0.78 | 1.09 |
| NeedleHammer | adversarial | 45 | 0.948 | 0.80 | 1.53 |

### Throughput, absent needles (GB/s, geomean over m = 8, 32, 256, 2048)

| Corpus | C++ Guardzilla | Rust Guardzilla | C++ NeedleHammer | Rust NeedleHammer |
|---|---:|---:|---:|---:|
| english | 26.6 | 53.5 | 25.8 | 30.2 |
| log | 26.5 | 54.4 | 26.4 | 29.6 |
| json | 26.9 | 55.0 | 26.3 | 30.8 |
| base64 | 27.8 | 59.4 | 26.9 | 32.3 |
| protein | 26.3 | 52.9 | 26.0 | 30.8 |
| dna | 5.5 | 6.3 | 21.8 | 21.6 |
| russian | 25.4 | 46.7 | 24.6 | 27.7 |
| chinese | 26.1 | 51.5 | 24.9 | 28.1 |
| source | 27.3 | 57.1 | 26.5 | 30.3 |
| replog | 27.3 | 58.1 | 26.3 | 31.0 |
| influenza | 6.4 | 7.4 | 22.3 | 22.3 |
| leaders | 22.8 | 40.2 | 23.7 | 26.5 |

### Throughput, present needles (GB/s of haystack up to the match)

| Corpus | C++ Guardzilla | Rust Guardzilla | C++ NeedleHammer | Rust NeedleHammer |
|---|---:|---:|---:|---:|
| english | 26.3 | 51.5 | 25.9 | 31.0 |
| log | 21.8 | 37.8 | 23.1 | 26.0 |
| json | 26.0 | 52.0 | 25.9 | 29.5 |
| base64 | 27.5 | 58.2 | 26.5 | 32.2 |
| protein | 26.2 | 52.7 | 25.5 | 30.3 |
| dna | 5.7 | 6.4 | 19.7 | 19.0 |
| russian | 25.7 | 48.0 | 25.4 | 29.9 |
| chinese | 26.4 | 49.9 | 24.8 | 28.7 |
| source | 26.7 | 53.7 | 26.4 | 30.3 |
| replog | 24.2 | 30.5 | 23.2 | 24.3 |
| influenza | 6.5 | 7.4 | 20.0 | 18.7 |
| leaders | 23.6 | 42.4 | 23.6 | 26.1 |

### Observations

- Guardzilla: Rust takes 0.56 of the C++ time on the corpora (geomean) and
  0.68 on the adversarial families. On English, absent needles, 53.5 vs
  26.6 GB/s. On `dna` and `influenza`, 6.3 vs 5.5 and 7.4 vs 6.4 GB/s.
- NeedleHammer: Rust takes 0.89 of the C++ time on the corpora and 0.95 on
  the adversarial families. On `dna` and `influenza` the two are within 1%
  (absent needles).
- Rust is slower in these cells: `block` at m = 16 to 64 (Guardzilla 1.22 to
  1.24 times the C++ time, NeedleHammer 1.40 to 1.53), `near` at m = 512 to
  4096 (1.08 to 1.25), and NeedleHammer with present needles on `influenza`
  (1.06 to 1.09) and on `dna` at m = 8 (1.07).
