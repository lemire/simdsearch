#!/bin/bash
# Runs upstream's unmodified C++ benchmark (the repository root, built out of
# tree in build/cpp) and the Rust benchmark (rust/compare) with the same arguments,
# pinned to one core, as upstream's benchmark/run.sh runs them. Each binary
# runs REPEAT times: timings differ by up to 30% between processes, so
# rust/scripts/compare.py takes the median per cell.
#
#   rust/scripts/run_benchmarks.sh [OUTDIR] [CPU] [REPEAT]
#
# Writes OUTDIR/cpp.csv and OUTDIR/rust.csv. Expects corpora/ (built by
# benchmark/tools/corpora.py) and both builds. Run from anywhere; paths are
# relative to the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-rust/results}
CPU=${2:-2}
REPEAT=${3:-5}
CPP=build/cpp/benchmark
RUST=rust/target/release
CORPORA=()
for c in english log json base64 protein dna russian chinese source replog influenza leaders; do
    CORPORA+=("corpora/$c.dat")
done
mkdir -p "$OUT"
: > "$OUT/cpp.csv"
: > "$OUT/rust.csv"
run() { taskset -c "$CPU" "$@"; }

for _ in $(seq "$REPEAT"); do
    for s in guardzilla needlehammer; do
        run "$CPP/search_bench_$s" corpora "${CORPORA[@]}" >> "$OUT/cpp.csv"
        run "$RUST/bench_$s" corpora "${CORPORA[@]}" >> "$OUT/rust.csv"
        run "$CPP/search_bench_$s" adversarial >> "$OUT/cpp.csv"
        run "$RUST/bench_$s" adversarial >> "$OUT/rust.csv"
    done
done
