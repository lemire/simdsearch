mod common;

use common::{Aligned, Rng, reference};

#[test]
fn conventions() {
    assert_eq!(simdsearch::find(b"hello", b""), Some(0));
    assert_eq!(simdsearch::find(b"", b""), Some(0));
    assert_eq!(simdsearch::find(b"", b"a"), None);
    assert_eq!(simdsearch::find(b"the quick brown fox", b"brown"), Some(10));
    assert_eq!(simdsearch::find(b"the quick brown fox", b"green"), None);
    assert_eq!(simdsearch::find(b"abcabcabd", b"abd"), Some(6));
    assert_eq!(
        simdsearch::find(&[b'a', 0, b'b', 0, 0, b'c'], &[0, 0, b'c']),
        Some(3)
    );
}

#[test]
fn exact_size_buffers_full_byte_range() {
    let mut rng = Rng::new(2026);
    for _ in 0..60_000 {
        let n = rng.below(2500);
        let m = rng.below(80);
        let (text_offset, needle_offset) = (rng.below(64), rng.below(64));
        let binary = rng.below(2) == 1;
        let draw = |rng: &mut Rng| {
            if binary {
                rng.byte()
            } else {
                b'a' + rng.byte_below(3)
            }
        };
        let text: Vec<u8> = (0..n).map(|_| draw(&mut rng)).collect();
        let needle: Vec<u8> = if m <= n && rng.below(2) == 1 {
            let at = rng.below(n - m + 1);
            let mut needle = text[at..at + m].to_vec();
            if m > 0 && rng.below(3) == 0 {
                let k = rng.below(m);
                needle[k] ^= 1 + rng.byte_below(255);
            }
            needle
        } else {
            (0..m).map(|_| draw(&mut rng)).collect()
        };
        let hay = Aligned::new(&text, text_offset);
        let needle = Aligned::new(&needle, needle_offset);
        let want = reference(hay.get(), needle.get());
        assert_eq!(
            simdsearch::find(hay.get(), needle.get()),
            want,
            "n={n} m={m}"
        );
        assert_eq!(
            simdsearch::needlehammer::find(hay.get(), needle.get()),
            want,
            "n={n} m={m}"
        );
        assert_eq!(
            simdsearch::twoway::find(hay.get(), needle.get()),
            want,
            "n={n} m={m}"
        );
    }
}
