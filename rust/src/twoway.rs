//! The Crochemore-Perrin two-way algorithm: linear time, constant space.
//!
//! Guardzilla and `NeedleHammer` hand their search over to it when their work
//! budget runs out.

use crate::verify::{same_prefix, same_suffix};

/// First occurrence of `needle` in `haystack`, by two-way alone.
///
/// The empty needle matches at 0.
#[must_use]
pub fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    find_from(haystack, needle, 0)
}

const fn max_suffix(needle: &[u8], reverse: bool) -> (usize, usize) {
    let (mut suffix, mut j, mut k, mut period) = (usize::MAX, 0, 1, 1);
    while j + k < needle.len() {
        let a = needle[j + k];
        let b = needle[suffix.wrapping_add(k)];
        if if reverse { a > b } else { a < b } {
            j += k;
            k = 1;
            period = j.wrapping_sub(suffix);
        } else if a == b {
            if k == period {
                j += period;
                k = 1;
            } else {
                k += 1;
            }
        } else {
            suffix = j;
            j += 1;
            k = 1;
            period = 1;
        }
    }
    (suffix.wrapping_add(1), period)
}

pub(crate) fn find_from(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    let len = needle.len();
    let (c1, per1) = max_suffix(needle, false);
    let (c2, per2) = max_suffix(needle, true);
    let (crit, mut period) = if c1 > c2 { (c1, per1) } else { (c2, per2) };
    let periodic = needle
        .get(period..period + crit)
        .is_some_and(|shifted| shifted == &needle[..crit]);
    if !periodic {
        period = crit.max(len - crit) + 1;
    }
    let mut memory = 0;
    let mut at = from;
    while at + len <= hay.len() {
        let window = &hay[at..at + len];
        let mut i = crit.max(memory);
        let lim = (i + 4).min(len);
        while i < lim && needle[i] == window[i] {
            i += 1;
        }
        if i == lim {
            i += same_prefix(&needle[i..], &window[i..]);
        }
        if i < len {
            at += i - crit + 1;
            memory = 0;
            continue;
        }
        let mut k = crit;
        let low = if memory + 4 < crit { crit - 4 } else { memory };
        while k > low && needle[k - 1] == window[k - 1] {
            k -= 1;
        }
        if k == low {
            k -= same_suffix(&needle[memory..k], &window[memory..k]);
        }
        if k <= memory {
            return Some(at);
        }
        at += period;
        memory = if periodic { len - period } else { 0 };
    }
    None
}
