//! Guardzilla: `StringZilla`'s substring search, aligned and guarded, in
//! O(n + m) time.

use crate::anchors::{self, Lanes, Offsets};
use crate::simd::{self, BLOCK, low_mask};
use crate::verify::{Outcome, Verifier};
use crate::{byte, twoway};

/// Compile-time settings, for measurement. [`Defaults`] is the published
/// searcher.
pub trait Settings {
    /// Number of anchors, 1 to 8.
    const ANCHORS: usize = 3;
    /// Whether the scan steps on 64-byte boundaries of the haystack.
    const ALIGN: bool = true;
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
    const { assert!(S::ANCHORS >= 1 && S::ANCHORS <= anchors::MAX) };
    match needle.len() {
        0 => Some(0),
        len if len > haystack.len() => None,
        1 => byte::find(haystack, needle[0]),
        _ => match scan::<S>(haystack, needle) {
            Some(Outcome::Match(at)) => Some(at),
            Some(Outcome::Exhausted(at)) => twoway::find_from(haystack, needle, at),
            None => None,
        },
    }
}

fn filter<S: Settings>(
    hay: &[u8],
    offsets: &Offsets,
    targets: &[simd::V; anchors::MAX],
    at: usize,
    mut live: u64,
) -> u64 {
    for (&offset, &target) in offsets.iter().zip(targets).take(S::ANCHORS) {
        live &= simd::eq8_masked(live, simd::load_masked(&hay[at + offset..], live), target);
    }
    live
}

#[inline]
fn next_candidates<S: Settings>(
    lanes: &Lanes<'_>,
    targets: &[simd::V; anchors::MAX],
    from: usize,
    blocks: usize,
) -> Option<(usize, u64)> {
    (from..blocks).find_map(|b| {
        let mut found = u64::MAX;
        for (lane, &target) in lanes.iter().zip(targets).take(S::ANCHORS) {
            found &= simd::eq8(simd::load(&lane[b]), target);
        }
        (found != 0).then_some((b, found))
    })
}

fn scan<S: Settings>(hay: &[u8], needle: &[u8]) -> Option<Outcome> {
    let offsets = anchors::select(needle, S::ANCHORS);
    let targets = offsets.map(|offset| simd::splat(needle[offset]));
    let span = hay.len() - needle.len() + 1;
    let mut verifier = Verifier::new(hay, needle, S::BUDGET);

    let start = if S::ALIGN {
        simd::head_len(hay).min(span)
    } else {
        0
    };
    if start != 0 {
        let head = filter::<S>(hay, &offsets, &targets, 0, low_mask(start));
        let outcome = verifier.verify(0, head);
        if outcome.is_some() {
            return outcome;
        }
    }
    let blocks = (span - start) / BLOCK;
    let lanes = anchors::lanes(hay, start, &offsets, S::ANCHORS, blocks);
    let mut b = 0;
    while let Some((at, found)) = next_candidates::<S>(&lanes, &targets, b, blocks) {
        let outcome = verifier.verify(start + BLOCK * at, found);
        if outcome.is_some() {
            return outcome;
        }
        b = at + 1;
    }
    let tail = start + BLOCK * blocks;
    if tail < span {
        let found = filter::<S>(hay, &offsets, &targets, tail, low_mask(span - tail));
        return verifier.verify(tail, found);
    }
    None
}
