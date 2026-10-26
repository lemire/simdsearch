//! The Rust counterpart of upstream's `benchmark/benchmarks/search_bench.cpp`:
//! the same modes, options, needles and CSV columns, so that its rows join
//! with those of the unmodified C++ benchmark. The four hardware-counter
//! columns are `nan`, as upstream prints them when counters are unavailable.
//!
//! ```text
//! bench_<name> [--offset K] [--rounds R] [--lengths L] [--header] corpora FILE... | adversarial
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};

/// A searcher under measurement. `prepare` runs once per needle, outside the
/// timed loop.
pub trait Searcher {
    const NAME: &'static str;
    type State;
    fn prepare(needle: &[u8]) -> Self::State;
    fn find(state: &Self::State, haystack: &[u8], needle: &[u8]) -> Option<usize>;
}

/// `std::mt19937_64`, as specified by the C++ standard.
#[derive(Debug, Clone)]
pub struct Mt19937_64 {
    state: [u64; 312],
    index: usize,
}

impl Mt19937_64 {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut state = [0; 312];
        state[0] = seed;
        for i in 1..312 {
            let prev = state[i - 1];
            state[i] = 6_364_136_223_846_793_005_u64
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        Self { state, index: 312 }
    }

    fn twist(&mut self) {
        const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
        const LOWER: u64 = 0x7FFF_FFFF;
        for i in 0..312 {
            let x = (self.state[i] & UPPER) | (self.state[(i + 1) % 312] & LOWER);
            let mag = if x & 1 == 0 { 0 } else { 0xB502_6F5A_A966_19E9 };
            self.state[i] = self.state[(i + 156) % 312] ^ (x >> 1) ^ mag;
        }
        self.index = 0;
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.index >= 312 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= (y >> 29) & 0x5555_5555_5555_5555;
        y ^= (y << 17) & 0x71D6_7FFF_EDA6_0000;
        y ^= (y << 37) & 0xFFF7_EEE0_0000_0000;
        y ^ (y >> 43)
    }

    /// `std::uniform_int_distribution<size_t>(0, max)(*this)` as libstdc++
    /// computes it: Lemire's nearly divisionless method.
    pub fn uniform(&mut self, max: usize) -> usize {
        let range = u64::try_from(max).unwrap_or(u64::MAX).wrapping_add(1);
        let mut product = u128::from(self.next_u64()) * u128::from(range);
        let low = product.to_le_bytes();
        let mut low = u64::from_le_bytes([
            low[0], low[1], low[2], low[3], low[4], low[5], low[6], low[7],
        ]);
        if low < range {
            let threshold = range.wrapping_neg() % range;
            while low < threshold {
                product = u128::from(self.next_u64()) * u128::from(range);
                let bytes = product.to_le_bytes();
                low = u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]);
            }
        }
        usize::try_from(product >> 64).unwrap_or(usize::MAX)
    }
}

/// A copy of some bytes placed `offset` bytes past a 64-byte boundary.
#[derive(Debug)]
pub struct AlignedBuffer {
    store: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuffer {
    #[must_use]
    pub fn new(bytes: &[u8], offset: usize) -> Self {
        let mut store = vec![0; bytes.len() + 128];
        let misalign = store.as_ptr().addr() % 64;
        let start = (offset % 64 + 64 - misalign) % 64;
        store[start..start + bytes.len()].copy_from_slice(bytes);
        Self {
            store,
            start,
            len: bytes.len(),
        }
    }

    #[must_use]
    pub fn get(&self) -> &[u8] {
        &self.store[self.start..self.start + self.len]
    }
}

const fn sequence_length(lead: u8) -> usize {
    match lead {
        0x00..=0x7F => 1,
        _ if lead >> 5 == 6 => 2,
        _ if lead >> 4 == 14 => 3,
        _ if lead >> 3 == 30 => 4,
        _ => 0,
    }
}

/// UTF-8 by `search_bench.cpp`'s rules: lead and continuation bytes only.
#[must_use]
pub fn valid_utf8(s: &[u8]) -> bool {
    let mut i = 0;
    while i < s.len() {
        let k = sequence_length(s[i]);
        if k == 0 || i + k > s.len() || s[i + 1..i + k].iter().any(|&c| c >> 6 != 2) {
            return false;
        }
        i += k;
    }
    true
}

#[must_use]
pub fn reverse_code_points(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = s.len();
    while i > 0 {
        let mut j = i - 1;
        while j > 0 && s[j] >> 6 == 2 {
            j -= 1;
        }
        out.extend_from_slice(&s[j..i]);
        i = j;
    }
    out
}

fn trim_to_utf8(s: &mut Vec<u8>) {
    for cut in 1..=3 {
        if valid_utf8(s) || s.len() <= cut {
            break;
        }
        if valid_utf8(&s[..s.len() - cut]) {
            s.truncate(s.len() - cut);
        }
    }
}

fn corpus_name(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name)
}

#[must_use]
pub fn reference(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memchr::memmem::find(haystack, needle)
}

struct Options {
    offset: usize,
    rounds: usize,
    lengths: Vec<usize>,
    header: bool,
    mode: String,
    files: Vec<String>,
}

fn parse_options(args: &[String]) -> Result<Options> {
    let mut opt = Options {
        offset: 0,
        rounds: 10,
        lengths: vec![8, 32, 256, 2048],
        header: false,
        mode: String::new(),
        files: Vec::new(),
    };
    let mut rest = args.iter().skip(1);
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--offset" => opt.offset = rest.next().context("--offset K")?.parse::<usize>()? % 64,
            "--rounds" => opt.rounds = rest.next().context("--rounds R")?.parse()?,
            "--lengths" => {
                opt.lengths = rest
                    .next()
                    .context("--lengths L")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--header" => opt.header = true,
            mode => {
                mode.clone_into(&mut opt.mode);
                break;
            }
        }
    }
    opt.files = rest.cloned().collect();
    ensure!(opt.rounds > 0, "--rounds must be positive");
    ensure!(!opt.lengths.contains(&0), "--lengths must be positive");
    ensure!(
        opt.mode == "corpora" || opt.mode == "adversarial",
        "usage: bench [--offset K] [--rounds R] [--lengths L] [--header] corpora FILE... | adversarial"
    );
    Ok(opt)
}

type FindFn<S> = fn(&<S as Searcher>::State, &[u8], &[u8]) -> Option<usize>;

struct Measurement {
    ns_per_search: f64,
    bytes_per_search: f64,
}

fn to_f64(x: usize) -> Result<f64> {
    Ok(f64::from(u32::try_from(x).context("count exceeds u32")?))
}

fn measure<S: Searcher>(
    haystack: &[u8],
    needles: &[Vec<u8>],
    rounds: usize,
    what: &str,
) -> Result<Measurement> {
    let states: Vec<S::State> = needles.iter().map(|n| S::prepare(n)).collect();
    let mut scanned = 0;
    for (needle, state) in needles.iter().zip(&states) {
        let want = reference(haystack, needle);
        scanned += want.map_or(haystack.len(), |at| at + needle.len());
        let got = S::find(state, haystack, needle);
        if got != want {
            bail!(
                "MISMATCH {}: {what} m={} got {got:?} want {want:?}",
                S::NAME,
                needle.len()
            );
        }
    }

    let find: FindFn<S> = black_box(S::find);
    let mut best = Duration::MAX;
    let mut sink = 0;
    for _ in 0..rounds {
        let start = Instant::now();
        for (needle, state) in needles.iter().zip(&states) {
            let x = find(state, black_box(haystack), needle);
            sink += x.map_or(0, |at| at + 1);
        }
        best = best.min(start.elapsed());
    }
    black_box(sink);
    let count = to_f64(needles.len())?;
    Ok(Measurement {
        ns_per_search: best.as_secs_f64() * 1e9 / count,
        bytes_per_search: to_f64(scanned)? / count,
    })
}

fn print_row<S: Searcher>(mode: &str, input: &str, case: &str, m: usize, x: &Measurement) {
    println!(
        "{mode},{input},{case},{m},{},{:.1},{:.0},nan,nan,nan,nan",
        S::NAME,
        x.ns_per_search,
        x.bytes_per_search
    );
}

#[must_use]
pub fn select_needles(h: &[u8], m: usize, present: bool, utf8: bool) -> Vec<Vec<u8>> {
    let m64 = u64::try_from(m).unwrap_or(u64::MAX);
    let mut rng = Mt19937_64::new(0x00C0_FFEE ^ m64 ^ if present { 0 } else { 0xABCD });
    let want = if present { 200 } else { 20 };
    let mut needles = Vec::new();
    let mut tries = 0;
    while needles.len() < want && tries < 200_000 {
        tries += 1;
        let at = rng.uniform(h.len() - m);
        let mut needle = h[at..at + m].to_vec();
        if utf8 && !valid_utf8(&needle) {
            continue;
        }
        if !present {
            needle = reverse_code_points(&needle);
            if reference(h, &needle).is_some() {
                continue;
            }
        }
        needles.push(needle);
    }
    if needles.len() < want {
        needles.clear();
    }
    needles
}

fn run_corpora<S: Searcher>(opt: &Options) -> Result<()> {
    for path in &opt.files {
        let mut h = std::fs::read(path).with_context(|| format!("cannot read {path}"))?;
        trim_to_utf8(&mut h);
        let utf8 = valid_utf8(&h);
        let hay = AlignedBuffer::new(&h, opt.offset);
        let name = corpus_name(path);
        for &m in &opt.lengths {
            if m > h.len() {
                continue;
            }
            for present in [true, false] {
                let needles = select_needles(&h, m, present, utf8);
                if needles.is_empty() {
                    continue;
                }
                let x = measure::<S>(hay.get(), &needles, opt.rounds, name)?;
                print_row::<S>(
                    "corpora",
                    name,
                    if present { "present" } else { "absent" },
                    m,
                    &x,
                );
            }
        }
    }
    Ok(())
}

fn periodic(period: &[u8], n: usize) -> Vec<u8> {
    period.iter().copied().cycle().take(n).collect()
}

/// The adversarial families of `search_bench.cpp`, 1 MiB, needle absent.
#[must_use]
pub fn adversarial(family: &str, m: usize) -> (Vec<u8>, Vec<u8>) {
    const N: usize = 1 << 20;
    match family {
        "tail" => {
            let mut p = vec![b'a'; m];
            p[m - 1] = b'b';
            (vec![b'a'; N], p)
        }
        "mid" => {
            let mut p = vec![b'a'; m];
            p[m / 2] = b'b';
            (vec![b'a'; N], p)
        }
        "ab" => {
            let mut p = periodic(b"ab", m);
            p[m / 2] = if p[m / 2] == b'a' { b'b' } else { b'a' };
            (periodic(b"ab", N), p)
        }
        "block" => {
            let mut period = vec![b'a'; m];
            period[m - 1] = b'b';
            (periodic(&period, N), vec![b'a'; m])
        }
        _ => {
            let per = if m < 256 { m.min(64) } else { 256 };
            let mut unit = vec![b'a'; per];
            unit[per - 1] = b'b';
            let p = periodic(&unit, m);
            let mut c = p.clone();
            c[m - 2] = b'c';
            (periodic(&c, N), p)
        }
    }
}

fn run_adversarial<S: Searcher>(opt: &Options) -> Result<()> {
    for family in ["tail", "mid", "ab", "block", "near"] {
        for m in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let (h, p) = adversarial(family, m);
            let hay = AlignedBuffer::new(&h, opt.offset);
            let x = measure::<S>(hay.get(), &[p], opt.rounds, family)?;
            print_row::<S>("adversarial", family, "absent", m, &x);
        }
    }
    Ok(())
}

/// Runs the benchmark for `S` with the process arguments.
///
/// # Errors
///
/// Bad arguments, an unreadable corpus, or an answer that differs from
/// `memchr::memmem::find`.
pub fn main<S: Searcher>() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let opt = parse_options(&args)?;
    if opt.header {
        println!(
            "mode,input,case,m,searcher,ns_per_search,bytes_per_search,cycles_per_search,\
             instructions_per_search,branches_per_search,branch_misses_per_search"
        );
    }
    if opt.mode == "corpora" {
        run_corpora::<S>(&opt)
    } else {
        run_adversarial::<S>(&opt)
    }
}
