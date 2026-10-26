use crate::simd::{self, BLOCK, low_mask};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Match(usize),
    Exhausted(usize),
}

#[inline]
fn differs(a: &[u8], b: &[u8], mask: u64) -> u64 {
    simd::ne8_masked(mask, simd::load_masked(a, mask), simd::load_masked(b, mask))
}

pub fn same_prefix(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut k = 0;
    while k < len {
        let ne = differs(&a[k..], &b[k..], low_mask(len - k));
        if ne != 0 {
            return k + ne.trailing_zeros() as usize;
        }
        k += BLOCK;
    }
    len
}

pub fn same_suffix(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut k = len;
    while k > 0 {
        let c = k.min(BLOCK);
        k -= c;
        let ne = differs(&a[k..], &b[k..], low_mask(c));
        if ne != 0 {
            return len - k - (BLOCK - ne.leading_zeros() as usize);
        }
    }
    len
}

#[inline(never)]
fn rest_matches(a: &[u8], b: &[u8]) -> (bool, usize) {
    let m = b.len();
    let mut d = BLOCK;
    let mut compares = 0;
    while d < m {
        compares += 1;
        if differs(&a[d..], &b[d..], low_mask(m - d)) != 0 {
            return (false, compares);
        }
        d += BLOCK;
    }
    (true, compares)
}

pub struct Verifier<'a> {
    t: &'a [u8],
    p: &'a [u8],
    head_mask: u64,
    head: simd::V,
    budget: usize,
    slack: usize,
    work: usize,
}

impl<'a> Verifier<'a> {
    pub fn new(t: &'a [u8], p: &'a [u8], budget: usize) -> Self {
        let m = p.len();
        let head_mask = low_mask(m);
        Self {
            t,
            p,
            head_mask,
            head: simd::load_masked(p, head_mask),
            budget,
            slack: m / 8 + BLOCK,
            work: 0,
        }
    }

    pub const fn haystack(&self) -> &'a [u8] {
        self.t
    }

    pub const fn needle(&self) -> &'a [u8] {
        self.p
    }

    #[inline]
    pub fn check(&mut self, q: usize) -> Option<Outcome> {
        let m = self.p.len();
        let window = &self.t[q..q + m];
        let head = &window[..m.min(BLOCK)];
        let ne = simd::ne8_masked(
            self.head_mask,
            simd::load_masked(head, self.head_mask),
            self.head,
        );
        if ne != 0 {
            return None;
        }
        if m <= BLOCK {
            return Some(Outcome::Match(q));
        }
        let (matched, compares) = rest_matches(window, self.p);
        self.work += compares;
        if matched {
            return Some(Outcome::Match(q));
        }
        (self.work > self.budget * (q / BLOCK) + self.slack).then_some(Outcome::Exhausted(q))
    }

    #[inline]
    pub fn verify(&mut self, i: usize, mut candidates: u64) -> Option<Outcome> {
        while candidates != 0 {
            let outcome = self.check(i + candidates.trailing_zeros() as usize);
            if outcome.is_some() {
                return outcome;
            }
            candidates &= candidates - 1;
        }
        None
    }
}
