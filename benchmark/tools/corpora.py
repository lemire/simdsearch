#!/usr/bin/env python3
"""Build the corpus set used by the data-diversity and haystack-size sweeps.

The default horspool file is one 141 KB English book, which is (a) LLC-resident
on every machine we test and (b) Latin-1-ish text with the byte-frequency skew
that anchor-and-verify filters like best. Neither property is universal, so this
script materialises two extra axes.

*Diversity* (`corpora/<name>.dat`, ~1 MB each). Low-entropy and non-Latin inputs
are not adversaries -- they are ordinary production workloads that happen to have
the byte-frequency structure the anchored filter depends on:

  english   the Project Gutenberg book, tiled to size (the horspool default)
  dna       |S| = 4, the classic small-alphabet workload
  protein   |S| = 20
  json      minified JSON: heavy punctuation, repeated key names
  base64    |S| = 64, near-uniform over a restricted range
  log       timestamped log lines with long repeated runs
  replog    log lines identical but for their counters (periodic text)
  source    real C/C++ source: identifiers, braces, long comment runs
            (data/cpp_source.txt, a fixed file shipped with the repository)
  chinese   UTF-8 CJK -- almost every byte >= 0x80
  russian   UTF-8 Cyrillic -- two-byte sequences, high bytes throughout
  influenza Pizza&Chili's influenza genomes (from NCBI's GenBank): real DNA,
            many similar strains, with header lines
  leaders   Pizza&Chili's world_leaders: the CIA's World Leaders lists,
            2003 to 2009, as text (a US government work, public domain)

The last three matter because UTF-8 CJK and Cyrillic spend most bytes at or
above 0x80, unlike the Latin book the default run uses. The StringZilla-style
selector's optional FilterHighBytes rule skips UTF-8 lead bytes; chinese and
russian are where that rule is tested.

*Size* (`corpora/size_<n>.dat`). The same English text tiled from 1 KB to 1 GB,
so the ranking can be checked once the haystack leaves cache and every searcher
converges toward memory bandwidth.

The two Pizza&Chili corpora are fetched (about 19 MB) and have no synthetic
substitute: offline, they are skipped. The non-Latin text is fetched from Project Gutenberg when a network is
available and synthesised deterministically otherwise; every synthetic corpus is seeded, so
runs are reproducible. Usage:

    python3 benchmark/tools/corpora.py --out corpora [--sizes] [--max-size 1G]
"""
import argparse
import os
import random
import string
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))          # benchmark/tools
ROOT = os.path.dirname(os.path.dirname(HERE))              # repository root

# 1 MB per diversity corpus. This experiment is about byte statistics, not about
# cache behaviour -- the size sweep below covers that separately -- and the
# benchmark's validation pass is O(n) per pattern, so a larger corpus costs a lot
# of wall clock without changing what the experiment measures.
TARGET = 1 << 20

# Project Gutenberg plain-text sources for the non-Latin corpora. Pinned by
# numeric ID; if the network is unavailable we fall back to a synthetic
# generator with the same byte statistics.
GUTENBERG = {
    # Aina Jushi, *Tou Peng Hsien Hua* (豆棚閒話), eBook #25328.
    "chinese": "https://www.gutenberg.org/files/25328/25328-0.txt",
    # Cyrillic-script Russian (P. N. Apostol, *Moskoviia v predstavlenii
    # inostrantsev XVI-XVII v.*, eBook #30774). Most Gutenberg "Russian"
    # entries are English translations -- this one is the original script,
    # ~86% of bytes >= 0x80.
    "russian": "https://www.gutenberg.org/files/30774/30774-0.txt",
}

# Pizza&Chili's repetitive corpus (pizzachili.dcc.uchile.cl/repcorpus.html).
# Each corpus is 1 MiB from the middle of the first 24 MiB of the file.
PIZZACHILI = {
    "influenza": "https://pizzachili.dcc.uchile.cl/repcorpus/real/influenza.gz",
    "leaders": "https://pizzachili.dcc.uchile.cl/repcorpus/real/world_leaders.gz",
}

# The C/C++ corpus: the benchmark's own sources at commit c52bda8 (driver,
# headers, tests; 261,785 bytes), concatenated once and shipped as a fixed file.
# Walking a checkout instead gave different bytes per commit and even per file
# system, since os.walk lists directories in on-disk order.
SOURCE_FILE = os.path.join(HERE, "..", "data", "cpp_source.txt")


def tile(buf: bytes, n: int) -> bytes:
    """Repeat buf to exactly n bytes."""
    if not buf:
        raise ValueError("empty source buffer")
    reps = (n + len(buf) - 1) // len(buf)
    return (buf * reps)[:n]


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:  # offline, 404, TLS, ...
        print(f"    (fetch failed: {e})", file=sys.stderr)
        return None


def gen_alphabet(alpha: str, n: int, seed: int) -> bytes:
    """n bytes drawn uniformly from `alpha`. Generated as a 1 MB block and tiled:
    the block is far longer than any needle we search for, so the tiling is
    invisible to a substring search but keeps generation fast."""
    rng = random.Random(seed)
    return tile(bytes(ord(rng.choice(alpha)) for _ in range(1 << 20)), n)


def gen_json(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    keys = ["id", "name", "value", "timestamp", "tags", "active", "count", "url"]
    out = [b"["]
    while sum(len(x) for x in out) < min(n, 1 << 20):
        rec = "{" + ",".join(
            f'"{k}":' + (f'"{"".join(rng.choices(string.ascii_lowercase, k=8))}"'
                         if rng.random() < 0.5 else str(rng.randint(0, 10**6)))
            for k in rng.sample(keys, 5)) + "},"
        out.append(rec.encode())
    out.append(b"]")
    return tile(b"".join(out), n)


def gen_log(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    msgs = ["connection established", "cache miss for key",
            "request completed in", "retrying after backoff",
            "----------------------------------------"]
    out = []
    while sum(len(x) for x in out) < min(n, 1 << 20):
        out.append(f"2026-07-27T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:"
                   f"{rng.randint(0,59):02d}Z {rng.choice(levels):5s} "
                   f"[worker-{rng.randint(0,15):02d}] {rng.choice(msgs)} "
                   f"{rng.randint(0,10**5)}\n".encode())
    return tile(b"".join(out), n)


def gen_replog(n: int) -> bytes:
    """Log lines identical but for their counters: periodic text, where a
    needle's anchors match on every line and candidates match deep."""
    lines = "".join(
        f"2026-10-07T12:{(i // 60) % 60:02d}:{i % 60:02d} INFO req id={i:06d} "
        f"status=200 path=/api/v1/items\n" for i in range(20000))
    return tile(lines.encode(), n)


def gen_source(n: int) -> bytes:
    """Real C/C++ source: the shipped fixed file, tiled to size."""
    return tile(open(SOURCE_FILE, "rb").read(), n)


def utf8_fallback(ranges, n: int, seed: int) -> bytes:
    """Synthetic UTF-8 over the given codepoint ranges (used when offline)."""
    rng = random.Random(seed)
    buf = bytearray()
    while len(buf) < (1 << 20):
        lo, hi = rng.choice(ranges)
        buf += chr(rng.randint(lo, hi)).encode("utf-8")
        if rng.random() < 0.12:
            buf += b" "
    return tile(bytes(buf), n)


def strip_gutenberg(raw: bytes) -> bytes:
    """Drop the Gutenberg licence header/footer, keeping the body."""
    s = raw
    for mark in (b"*** START OF TH", b"*** START OF THE PROJECT"):
        i = s.find(mark)
        if i >= 0:
            s = s[s.find(b"\n", i) + 1:]
            break
    for mark in (b"*** END OF TH", b"End of the Project Gutenberg"):
        i = s.find(mark)
        if i >= 0:
            s = s[:i]
            break
    return s.strip() or raw


def build_diversity(out: str, book: bytes):
    jobs = {
        "english": lambda: tile(book, TARGET),
        "dna":     lambda: gen_alphabet("ACGT", TARGET, 1),
        "protein": lambda: gen_alphabet("ACDEFGHIKLMNPQRSTVWY", TARGET, 2),
        "base64":  lambda: gen_alphabet(
            string.ascii_letters + string.digits + "+/", TARGET, 3),
        "json":    lambda: gen_json(TARGET, 4),
        "log":     lambda: gen_log(TARGET, 5),
        "replog":  lambda: gen_replog(TARGET),
    }
    for name, fn in jobs.items():
        path = os.path.join(out, name + ".dat")
        if os.path.exists(path) and os.path.getsize(path) == TARGET:
            print(f"  {name}: cached"); continue
        print(f"  {name} ...")
        open(path, "wb").write(fn())

    path = os.path.join(out, "source.dat")
    if not (os.path.exists(path) and os.path.getsize(path) == TARGET):
        print("  source ...")
        open(path, "wb").write(gen_source(TARGET))

    fallbacks = {
        "chinese": [(0x4E00, 0x9FFF)],
        "russian": [(0x0410, 0x044F)],
    }
    for name, url in GUTENBERG.items():
        path = os.path.join(out, name + ".dat")
        if os.path.exists(path) and os.path.getsize(path) == TARGET:
            print(f"  {name}: cached"); continue
        print(f"  {name} ...")
        raw = fetch(url)
        if raw and len(strip_gutenberg(raw)) > 50_000:
            body = strip_gutenberg(raw)
            hi = sum(1 for b in body[:100_000] if b >= 0x80)
            print(f"    ({len(body)} bytes, {100*hi/min(len(body),100_000):.0f}%"
                  f" bytes >= 0x80)")
            open(path, "wb").write(tile(body, TARGET))
        else:
            print("    (offline: synthesising UTF-8 instead)", file=sys.stderr)
            open(path, "wb").write(
                utf8_fallback(fallbacks[name], TARGET, 6))


def middle(raw: bytes, n: int) -> bytes:
    """n bytes from the middle of raw, starting on a UTF-8 character."""
    s = len(raw) // 2 - n // 2
    b = raw[s:s + n + 4]
    i = 0
    while i < 4 and 0x80 <= b[i] < 0xC0:
        i += 1
    return b[i:i + n]


def build_pizzachili(out: str):
    import zlib
    for name, url in PIZZACHILI.items():
        path = os.path.join(out, name + ".dat")
        if os.path.exists(path) and os.path.getsize(path) == TARGET:
            print(f"  {name}: cached"); continue
        print(f"  {name} ...")
        raw = fetch(url)
        if not raw:
            print(f"    (offline: {name} skipped)", file=sys.stderr)
            continue
        body = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw, 24 << 20)
        open(path, "wb").write(middle(body, TARGET))


def build_sizes(out: str, book: bytes, max_size: int):
    n = 1024
    while n <= max_size:
        path = os.path.join(out, f"size_{n}.dat")
        if not (os.path.exists(path) and os.path.getsize(path) == n):
            print(f"  size_{n} ...")
            open(path, "wb").write(tile(book, n))
        n *= 8


def write_manifest(out: str):
    """Record the size and SHA-256 of every corpus.

    Two of the corpora are fetched from Project Gutenberg and fall back to a
    seeded synthetic generator when the network is unavailable. That fallback is
    deliberate, but it means two readers can generate different bytes under the
    same command and compare numbers that were never comparable. The manifest
    makes which happened checkable after the fact rather than a line on stderr
    that scrolled past.
    """
    import hashlib
    lines = []
    for f in sorted(os.listdir(out)):
        if not f.endswith(".dat"):
            continue
        b = open(os.path.join(out, f), "rb").read()
        lines.append(f"{hashlib.sha256(b).hexdigest()}  {len(b):>9}  {f}")
    path = os.path.join(out, "MANIFEST.txt")
    with open(path, "w") as fh:
        fh.write("# sha256, bytes, corpus -- generated by corpora.py\n")
        fh.write("\n".join(lines) + "\n")
    print(f"manifest -> {path}")
    for l in lines:
        print("  " + l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "corpora"))
    ap.add_argument("--sizes", action="store_true",
                    help="also build the haystack-size sweep")
    ap.add_argument("--max-size", default="1G")
    ap.add_argument("--book", default=None,
                    help="path to the baseline English text (43-0.txt)")
    a = ap.parse_args()

    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    ms = a.max_size.upper()
    max_size = int(ms[:-1]) * mult[ms[-1]] if ms[-1] in mult else int(ms)

    os.makedirs(a.out, exist_ok=True)
    bookpath = a.book
    if not bookpath:
        for cand in (
            os.path.join(ROOT, "benchmark", "data", "43-0.txt"),
            os.path.join(HERE, "..", "data", "43-0.txt"),
            os.path.join(ROOT, "..", "simdsearch", "benchmark", "data",
                         "43-0.txt"),
        ):
            if os.path.exists(cand):
                bookpath = cand
                break
    if not bookpath or not os.path.exists(bookpath):
        sys.exit("could not find the baseline text; pass --book <path to 43-0.txt")
    book = open(bookpath, "rb").read()
    print(f"baseline text: {bookpath} ({len(book)} bytes)")

    print("diversity corpora:")
    build_diversity(a.out, book)
    build_pizzachili(a.out)
    write_manifest(a.out)
    if a.sizes:
        print("size sweep:")
        build_sizes(a.out, book, max_size)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
