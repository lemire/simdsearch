//! `NeedleHammer`: Guardzilla with a second, branch-free scan for inputs that
//! defeat the branch predictor, in O(n + m) time.
//!
//! Needs AVX-512 VBMI2 and VL in addition to F and BW.

use core::ops::ControlFlow;

use crate::anchors::{self, Offsets};
use crate::simd::{self, BLOCK, Block, low_mask};
use crate::verify::{Outcome, Verifier};
use crate::{byte, twoway};

/// Compile-time settings, for measurement. [`Defaults`] is the published
/// searcher.
pub trait Settings {
    /// Blocks in each of the first two windows of the first scan.
    const WINDOW: usize = 64;
    /// Blocks in each later window.
    const LONG_WINDOW: usize = 2048;
    /// The switch threshold, percent of blocks with a candidate.
    const BAND: usize = 16;
    /// Anchors of the second scan, 1 to 8.
    const ANCHORS: usize = 3;
    /// Whether the second scan is timed against the first.
    const TIMED: bool = true;
    /// Blocks scanned first by a lean Guardzilla loop.
    const PROLOGUE: usize = 16;
    /// 64-byte compares allowed per 64 haystack bytes scanned.
    const BUDGET: usize = 4;
}

/// The published settings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Defaults;

impl Settings for Defaults {}

/// First occurrence of `needle` in `haystack`.
///
/// The empty needle matches at 0. No byte outside either slice is read.
#[must_use]
pub fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    find_with::<Defaults>(haystack, needle)
}

/// [`find`] with the settings `S`.
#[must_use]
pub fn find_with<S: Settings>(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    const {
        assert!(S::ANCHORS >= 1 && S::ANCHORS <= anchors::MAX);
        assert!(S::WINDOW >= 1 && S::WINDOW <= 4096);
    };
    match needle.len() {
        0 => return Some(0),
        len if len > haystack.len() => return None,
        1 => return byte::find(haystack, needle[0]),
        _ => {}
    }
    let offsets = if needle.len() >= 3 {
        anchors::stringzilla(needle)
    } else {
        [0, 1, 1]
    };
    let head = simd::head_len(haystack);
    let end = head + BLOCK * S::PROLOGUE;
    let outcome = if S::PROLOGUE > 0 && end + BLOCK - 1 <= haystack.len() - needle.len() {
        prologue::<S>(haystack, needle, &offsets, head)
            .or_else(|| scan::<S>(haystack, needle, &offsets, end))
    } else {
        scan::<S>(haystack, needle, &offsets, 0)
    };
    match outcome {
        Some(Outcome::Match(at)) => Some(at),
        Some(Outcome::Exhausted(at)) => twoway::find_from(haystack, needle, at),
        None => None,
    }
}

type Triple = [usize; 3];
type Targets = [simd::V; 3];

const CHUNK: usize = 64;
const POSITIONS: usize = BLOCK * CHUNK + CHUNK / 4;

static IOTA: Block = {
    let mut iota = [0; BLOCK];
    let mut value: u8 = 0;
    while (value as usize) < BLOCK {
        iota[value as usize] = value;
        value += 1;
    }
    iota
};

struct Buffers {
    pos8: Vec<u8>,
    pos32: Vec<u32>,
}

const fn widen(offsets: &Triple) -> Offsets {
    let [a, b, c] = *offsets;
    [a, b, c, c, c, c, c, c]
}

fn filter(hay: &[u8], offsets: &Triple, targets: &Targets, at: usize, mut live: u64) -> u64 {
    for (&offset, &target) in offsets.iter().zip(targets) {
        live &= simd::eq8_masked(live, simd::load_masked(&hay[at + offset..], live), target);
    }
    live
}

#[inline]
fn candidates(x0: &Block, x1: &Block, x2: &Block, targets: &Targets) -> u64 {
    simd::eq8(simd::load(x0), targets[0])
        & simd::eq8(simd::load(x1), targets[1])
        & simd::eq8(simd::load(x2), targets[2])
}

fn prologue<S: Settings>(
    hay: &[u8],
    needle: &[u8],
    offsets: &Triple,
    head: usize,
) -> Option<Outcome> {
    let targets = offsets.map(|offset| simd::splat(needle[offset]));
    let mut verifier = Verifier::new(hay, needle, S::BUDGET);
    if head != 0 {
        let outcome = verifier.verify(0, filter(hay, offsets, &targets, 0, low_mask(head)));
        if outcome.is_some() {
            return outcome;
        }
    }
    let [l0, l1, l2, ..] = anchors::lanes(hay, head, &widen(offsets), 3, S::PROLOGUE);
    for (b, ((x0, x1), x2)) in l0.iter().zip(l1).zip(l2).enumerate() {
        let found = candidates(x0, x1, x2, &targets);
        if found != 0 {
            let outcome = verifier.verify(head + BLOCK * b, found);
            if outcome.is_some() {
                return outcome;
            }
        }
    }
    None
}

fn scan<S: Settings>(hay: &[u8], needle: &[u8], offsets: &Triple, from: usize) -> Option<Outcome> {
    let targets = offsets.map(|offset| simd::splat(needle[offset]));
    let span = hay.len() - needle.len() + 1;
    let mut verifier = Verifier::new(hay, needle, S::BUDGET);

    let start = if from == 0 {
        simd::head_len(hay).min(span)
    } else {
        from
    };
    if from == 0 && start != 0 {
        let outcome = verifier.verify(0, filter(hay, offsets, &targets, 0, low_mask(start)));
        if outcome.is_some() {
            return outcome;
        }
    }
    let blocks = (span - start) / BLOCK;
    let [l0, l1, l2, ..] = anchors::lanes(hay, start, &widen(offsets), 3, blocks);
    let (mut b, mut in_band, mut patience, mut windows) = (0, 0, 2, 0);
    while b < blocks {
        let size = if windows < 2 {
            S::WINDOW
        } else {
            S::LONG_WINDOW
        };
        windows += 1;
        let full = blocks - b > size;
        let end = if full { b + size } else { blocks };
        let timed = in_band + 1 >= patience;
        let clock = if timed { simd::rdtsc() } else { 0 };
        let mut hits = 0;
        let window = l0[b..end].iter().zip(&l1[b..end]).zip(&l2[b..end]);
        for (k, ((x0, x1), x2)) in window.enumerate() {
            let found = candidates(x0, x1, x2, &targets);
            if found != 0 {
                hits += 1;
                let outcome = verifier.verify(start + BLOCK * (b + k), found);
                if outcome.is_some() {
                    return outcome;
                }
            }
        }
        b = end;
        let band = hits * 100 > size * S::BAND && hits * 100 < size * (100 - S::BAND);
        in_band = if full && band { in_band + 1 } else { 0 };
        if in_band >= patience && b < blocks {
            let per_block = (simd::rdtsc() - clock) / size as u64;
            match second_scan::<S>(offsets, start, b, blocks, &mut verifier, per_block) {
                ControlFlow::Break(outcome) => return Some(outcome),
                ControlFlow::Continue(resume) => b = resume,
            }
            in_band = 0;
            patience *= 2;
        }
    }
    let tail = start + BLOCK * blocks;
    if tail < span {
        return verifier.verify(
            tail,
            filter(hay, offsets, &targets, tail, low_mask(span - tail)),
        );
    }
    None
}

#[inline(never)]
fn second_scan<S: Settings>(
    offsets: &Triple,
    start: usize,
    mut b: usize,
    blocks: usize,
    verifier: &mut Verifier<'_>,
    first_per_block: u64,
) -> ControlFlow<Outcome, usize> {
    let last = verifier.haystack().len() - verifier.needle().len();
    let mut buffers = Buffers {
        pos8: vec![0; POSITIONS + BLOCK],
        pos32: vec![0; POSITIONS + 16],
    };
    let mut chunk = 0;
    while b < blocks {
        let at = start + BLOCK * b;
        let full = blocks - b > CHUNK;
        let end = if full { b + CHUNK } else { blocks };
        let timed = S::TIMED && chunk % 16 == 0;
        let clock = if timed { simd::rdtsc() } else { 0 };
        if let Some(outcome) = compress_chunk::<S>(offsets, at, end - b, verifier, &mut buffers) {
            return ControlFlow::Break(outcome);
        }
        let end_byte = if full {
            at + BLOCK * CHUNK
        } else {
            last - (BLOCK - 2)
        };
        let timed_blocks = ((end_byte - at) / BLOCK) as u64;
        if timed && simd::rdtsc() - clock > first_per_block * timed_blocks {
            return ControlFlow::Continue(end);
        }
        b = end;
        chunk += 1;
    }
    ControlFlow::Continue(b)
}

fn gather_offset(offsets: &[usize], len: usize) -> usize {
    let (mut best, mut fewest) = (0, 4);
    let mut j = 0;
    while j + 4 <= len && fewest > 0 {
        let overlaps = offsets.iter().filter(|&&a| a.wrapping_sub(j) < 4).count();
        if overlaps < fewest {
            fewest = overlaps;
            best = j;
        }
        j += 1;
    }
    best
}

#[inline(never)]
fn compress_chunk<S: Settings>(
    first_offsets: &Triple,
    base: usize,
    blocks: usize,
    verifier: &mut Verifier<'_>,
    buffers: &mut Buffers,
) -> Option<Outcome> {
    let (hay, needle) = (verifier.haystack(), verifier.needle());
    let len = needle.len();
    let offsets = if S::ANCHORS == 3 {
        widen(first_offsets)
    } else {
        anchors::spaced(len, S::ANCHORS)
    };
    let targets = offsets.map(|offset| simd::splat(needle[offset]));
    let word_at = gather_offset(&offsets[..S::ANCHORS], len);
    let word = needle[word_at..]
        .first_chunk::<4>()
        .map_or(0, |w| i32::from_ne_bytes(*w));
    let want = simd::splat_i32(word);
    let gather: u16 = if len >= 4 { u16::MAX } else { 0 };
    let step = simd::splat(64);
    let zero = simd::zero();
    let one = simd::splat_i32(1);
    let lane15 = simd::splat_i32(15);

    let lanes = anchors::lanes(hay, base, &offsets, S::ANCHORS, blocks);
    let mut count = 0;
    let mut idx = simd::load(&IOTA);
    for b in 0..blocks {
        let mut found = u64::MAX;
        for (lane, &target) in lanes.iter().zip(&targets).take(S::ANCHORS) {
            found &= simd::eq8(simd::load(&lane[b]), target);
        }
        found |= u64::from(b & 3 == 3) << 63;
        simd::store(&mut buffers.pos8, count, simd::compress8(found, idx));
        count += found.count_ones() as usize;
        idx = simd::add8(idx, step);
    }

    let gather_base = &hay[base + word_at..];
    let mut kept = 0;
    let mut prev = simd::splat_i32(-1);
    let mut window = zero;
    let mut j = 0;
    while j < count {
        let lm: u16 = if count - j >= 16 {
            u16::MAX
        } else {
            (1 << (count - j)) - 1
        };
        let cur = simd::widen8to32(simd::load_masked_16(&buffers.pos8, j, lm));
        let starts = simd::ge32(simd::alignr32::<15>(cur, prev), cur);
        let mut f = simd::select32(starts, one);
        f = simd::add32(f, simd::alignr32::<15>(f, zero));
        f = simd::add32(f, simd::alignr32::<14>(f, zero));
        f = simd::add32(f, simd::alignr32::<12>(f, zero));
        f = simd::add32(f, simd::alignr32::<8>(f, zero));
        let spans = simd::add32(window, f);
        let pos = simd::add32(simd::shl32::<8>(spans), cur);
        window = simd::permute32(lane15, spans);
        prev = cur;
        let got = simd::gather_u32(lm & gather, pos, gather_base);
        let pass = (lm & !gather) | simd::eq32_masked(lm & gather, got, want);
        simd::store_u32(&mut buffers.pos32, kept, simd::compress32(pass, pos));
        kept += pass.count_ones() as usize;
        j += 16;
    }

    buffers.pos32[..kept]
        .iter()
        .find_map(|&x| verifier.check(base + x as usize))
}
