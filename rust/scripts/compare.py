"""Compares the results of run_benchmarks.sh: upstream's C++ search_bench
and the Rust bench binaries, which print the same CSV columns.

Usage:
    python3 rust/scripts/compare.py [RESULTS_DIR] [--full]

Takes the median time of every cell (mode, input, case, m, searcher) over
the repeated runs, checks that both languages measured the same needles
(equal bytes_per_search), and prints markdown tables: the Rust/C++ time ratio
per algorithm, and the throughput per corpus. --full adds every cell.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PAIRS = (("Guardzilla", "Guardzilla (Rust)"), ("NeedleHammer", "NeedleHammer (Rust)"))
THROUGHPUT_COLUMNS = (
    ("C++", "Guardzilla", "C++ Guardzilla"),
    ("Rust", "Guardzilla (Rust)", "Rust Guardzilla"),
    ("C++", "NeedleHammer", "C++ NeedleHammer"),
    ("Rust", "NeedleHammer (Rust)", "Rust NeedleHammer"),
)
CORPORA = (
    "english", "log", "json", "base64", "protein", "dna",
    "russian", "chinese", "source", "replog", "influenza", "leaders",
)


@dataclass(frozen=True)
class Cell:
    mode: str
    input: str
    case: str
    m: int


@dataclass(frozen=True)
class Best:
    ns: float
    bytes: float

    @property
    def gbps(self) -> float:
        return self.bytes / self.ns


Table = dict[tuple[Cell, str], Best]


def load(path: Path) -> Table:
    runs: dict[tuple[Cell, str], list[Best]] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "mode":
                continue
            cell = Cell(row[0], row[1], row[2], int(row[3]))
            runs[(cell, row[4])].append(Best(float(row[5]), float(row[6])))
    return {
        key: Best(statistics.median(x.ns for x in xs), xs[0].bytes)
        for key, xs in runs.items()
    }


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def check_needles(cpp: Table, rust: Table) -> int:
    mismatches = 0
    for cpp_name, rust_name in PAIRS:
        for (cell, name), x in cpp.items():
            y = rust.get((cell, rust_name))
            if name == cpp_name and y is not None and abs(x.bytes - y.bytes) > 0.5:
                logger.error("different needles: %s %s (%s vs %s bytes)", cell, name, x.bytes, y.bytes)
                mismatches += 1
    return mismatches


def ratio_table(cpp: Table, rust: Table) -> list[str]:
    lines = [
        "| Algorithm | Workload | Cells | Rust / C++ time (geomean) | Best | Worst |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cpp_name, rust_name in PAIRS:
        for mode in ("corpora", "adversarial"):
            ratios = [
                rust[(cell, rust_name)].ns / x.ns
                for (cell, name), x in cpp.items()
                if name == cpp_name and cell.mode == mode and (cell, rust_name) in rust
            ]
            if ratios:
                lines.append(
                    f"| {cpp_name} | {mode} | {len(ratios)} | {geomean(ratios):.3f} "
                    f"| {min(ratios):.2f} | {max(ratios):.2f} |"
                )
    return lines


def throughput_table(cpp: Table, rust: Table, case: str) -> list[str]:
    tables = {"C++": cpp, "Rust": rust}
    header = " | ".join(label for _, _, label in THROUGHPUT_COLUMNS)
    lines = [f"| Corpus | {header} |", "|---|" + "---:|" * len(THROUGHPUT_COLUMNS)]
    for corpus in CORPORA:
        values = []
        for lang, name, _ in THROUGHPUT_COLUMNS:
            xs = [
                x.gbps
                for (cell, searcher), x in tables[lang].items()
                if searcher == name and cell.mode == "corpora"
                and cell.input == corpus and cell.case == case
            ]
            values.append(f"{geomean(xs):.1f}" if xs else "-")
        lines.append(f"| {corpus} | " + " | ".join(values) + " |")
    return lines


def full_table(cpp: Table, rust: Table) -> list[str]:
    lines = [
        "| Mode | Input | Case | m | Algorithm | C++ ns | Rust ns | Rust / C++ |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    by_cell: dict[Cell, list[tuple[str, str]]] = defaultdict(list)
    for cpp_name, rust_name in PAIRS:
        for cell, name in cpp:
            if name == cpp_name and (cell, rust_name) in rust:
                by_cell[cell].append((cpp_name, rust_name))
    for cell in sorted(by_cell, key=lambda c: (c.mode, c.input, c.case, c.m)):
        for cpp_name, rust_name in by_cell[cell]:
            a, b = cpp[(cell, cpp_name)].ns, rust[(cell, rust_name)].ns
            lines.append(
                f"| {cell.mode} | {cell.input} | {cell.case} | {cell.m} | {cpp_name} "
                f"| {a:.1f} | {b:.1f} | {b / a:.2f} |"
            )
    return lines


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="?", type=Path, default=Path("rust/results"))
    parser.add_argument("--full", action="store_true", help="also list every cell")
    args = parser.parse_args()

    cpp = load(args.results / "cpp.csv")
    rust = load(args.results / "rust.csv")
    if check_needles(cpp, rust):
        return 1
    logger.info("%d C++ cells, %d Rust cells, same needles", len(cpp), len(rust))

    sections = [
        "### Rust / C++ time ratio (below 1: Rust faster)",
        "",
        *ratio_table(cpp, rust),
        "",
        "### Throughput, absent needles (GB/s, geomean over m = 8, 32, 256, 2048)",
        "",
        *throughput_table(cpp, rust, "absent"),
        "",
        "### Throughput, present needles (GB/s of haystack up to the match)",
        "",
        *throughput_table(cpp, rust, "present"),
    ]
    if args.full:
        sections += ["", "### Every cell", "", *full_table(cpp, rust)]
    sys.stdout.write("\n".join(sections) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
