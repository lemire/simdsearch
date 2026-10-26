#![allow(dead_code, reason = "each test crate uses a subset of the helpers")]

pub type Find = fn(&[u8], &[u8]) -> Option<usize>;

pub struct Rng(u64);

impl Rng {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound as u64).unwrap_or(usize::MAX)
    }

    pub const fn byte(&mut self) -> u8 {
        self.next().to_le_bytes()[0]
    }

    pub const fn byte_below(&mut self, bound: u8) -> u8 {
        (self.next() % bound as u64).to_le_bytes()[0]
    }
}

pub struct Aligned {
    store: Vec<u8>,
    start: usize,
    len: usize,
}

impl Aligned {
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

    pub fn get(&self) -> &[u8] {
        &self.store[self.start..self.start + self.len]
    }
}

pub fn reference(hay: &[u8], needle: &[u8]) -> Option<usize> {
    memchr::memmem::find(hay, needle)
}

pub struct Checker {
    find: Find,
    pub cases: usize,
    pub fails: usize,
}

impl Checker {
    pub const fn new(find: Find) -> Self {
        Self {
            find,
            cases: 0,
            fails: 0,
        }
    }

    pub fn check(&mut self, hay: &[u8], needle: &[u8], what: &str) {
        self.cases += 1;
        let want = reference(hay, needle);
        let got = (self.find)(hay, needle);
        if got != want {
            if self.fails < 20 {
                eprintln!(
                    "FAIL {what}: n={} m={} want={want:?} got={got:?}",
                    hay.len(),
                    needle.len()
                );
            }
            self.fails += 1;
        }
    }
}

const QUICK: bool = cfg!(debug_assertions);

pub fn suite(find: Find) {
    let mut checker = Checker::new(find);
    let mut rng = Rng::new(20_261_013);
    for r in 0..if QUICK { 10_000 } else { 200_000 } {
        let alpha = if rng.below(5) == 0 {
            None
        } else {
            Some(1 + rng.byte_below(6))
        };
        let n = rng.below(if r % 10 == 0 { 20_000 } else { 1200 });
        let m = 1 + rng.below(if r % 7 == 0 { 3000 } else { 200 });
        let offset = rng.below(64);
        let draw = |rng: &mut Rng| match alpha {
            None => rng.byte(),
            Some(a) => b'a' + rng.byte_below(a),
        };
        let text: Vec<u8> = (0..n).map(|_| draw(&mut rng)).collect();
        let needle: Vec<u8> = if m <= n && rng.below(2) == 1 {
            let at = rng.below(n - m + 1);
            let mut needle = text[at..at + m].to_vec();
            if rng.below(3) == 0 {
                let k = rng.below(m);
                needle[k] ^= 1 + rng.byte_below(255);
            }
            needle
        } else {
            (0..m).map(|_| draw(&mut rng)).collect()
        };
        checker.check(Aligned::new(&text, offset).get(), &needle, "random");
    }

    let base: Vec<u8> = (0..4096 + 64)
        .map(|i: usize| b'a' + ((i * 7 + i / 13) % 23).to_le_bytes()[0])
        .collect();
    for offset in 0..64 {
        let text = &base[offset..offset + 4096];
        for m in [2, 5, 63, 64, 65, 200] {
            for at in (0..=text.len() - m).step_by(61) {
                checker.check(text, &text[at..at + m], "alignment");
            }
        }
    }

    let n = if QUICK { 1 << 13 } else { 1 << 18 };
    for m in [16, 64, 65, 128, 1000, if QUICK { 2000 } else { 4096 }] {
        let mut hay = vec![b'a'; n];
        let mut needle = vec![b'a'; m];
        needle[m - 1] = b'b';
        checker.check(&hay, &needle, "tail");
        needle = vec![b'a'; m];
        needle[m / 2] = b'b';
        checker.check(&hay, &needle, "mid");
        for (i, byte) in hay.iter_mut().enumerate() {
            *byte = if i % m == m - 1 { b'b' } else { b'a' };
        }
        needle = vec![b'a'; m];
        checker.check(&hay, &needle, "block");
        hay[n - 3 * m] = b'a';
        checker.check(&hay, &needle, "block-found");
    }

    assert_eq!(
        checker.fails, 0,
        "{} of {} cases failed",
        checker.fails, checker.cases
    );
}
