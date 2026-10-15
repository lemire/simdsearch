#!/bin/bash
# Every measurement in the paper, on this machine. Usage (from the repository
# root, after building with cmake -B build -DSIMDSEARCH_RUST=ON && cmake --build build):
#
#   benchmark/run.sh OUTDIR [ROUNDS] [CPU]
#
# Writes OUTDIR/corpora.csv, adversarial.csv, sensitivity.csv and alignment.csv,
# each the concatenation of ROUNDS rounds (default 3; analyses keep the best
# time per cell). Every searcher is pinned to core CPU (default 2).
set -eu
OUT=${1:?usage: benchmark/run.sh OUTDIR [ROUNDS] [CPU]}
ROUNDS=${2:-3}
CPU=${3:-2}
B=build/benchmark
C=$(for c in english log json base64 protein dna russian chinese source replog; do printf 'corpora/%s.dat ' $c; done)
SEARCHERS="guardzilla needlehammer stringzilla memmem string_view boyer_moore_horspool rust_memmem rust_finder rust_str"
mkdir -p "$OUT"
run() { taskset -c "$CPU" "$B/search_bench_$1" "${@:2}"; }

for r in $(seq "$ROUNDS"); do
    for s in $SEARCHERS; do run "$s" corpora $C >> "$OUT/corpora.csv"; done
    for s in guardzilla needlehammer stringzilla twoway memmem rust_finder; do run "$s" adversarial >> "$OUT/adversarial.csv"; done
    for k in 1 2 4 5 6 7 8; do run "guardzilla_anchors$k" corpora $C >> "$OUT/sensitivity.csv"; done
    for s in guardzilla_budget2 guardzilla_budget8; do run "$s" corpora $C >> "$OUT/sensitivity.csv"; done
    for s in guardzilla_budget2 guardzilla_budget8; do run "$s" adversarial >> "$OUT/adversarial.csv"; done
    for s in needlehammer_band10 needlehammer_band25 needlehammer_window16; do run "$s" corpora $C >> "$OUT/sensitivity.csv"; done
    # The haystack 0, 16, 32 and 48 bytes past a 64-byte boundary.
    for off in 0 16 32 48; do
        for s in guardzilla guardzilla_noalign needlehammer stringzilla; do
            run "$s" --offset "$off" corpora $C | sed "s/^corpora,/offset$off,/" >> "$OUT/alignment.csv"
        done
    done
    echo "round $r done"
done
