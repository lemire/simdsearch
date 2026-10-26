use crate::simd::{self, BLOCK, low_mask};

fn masked(s: &[u8], v: simd::V) -> u64 {
    let mask = low_mask(s.len());
    simd::eq8_masked(mask, simd::load_masked(s, mask), v)
}

pub fn find(t: &[u8], byte: u8) -> Option<usize> {
    let v = simd::splat(byte);
    let h = simd::head_len(t).min(t.len());
    let (head, body) = t.split_at(h);
    let hit = masked(head, v);
    if hit != 0 {
        return Some(hit.trailing_zeros() as usize);
    }
    let (blocks, tail) = body.as_chunks::<BLOCK>();
    for (b, block) in blocks.iter().enumerate() {
        let hit = simd::eq8(simd::load(block), v);
        if hit != 0 {
            return Some(h + BLOCK * b + hit.trailing_zeros() as usize);
        }
    }
    let hit = masked(tail, v);
    (hit != 0).then(|| h + BLOCK * blocks.len() + hit.trailing_zeros() as usize)
}
