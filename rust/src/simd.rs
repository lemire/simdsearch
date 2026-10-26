#![allow(
    unsafe_code,
    reason = "the one module that calls AVX-512 intrinsics; every other module is safe"
)]

#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
use core::arch::x86_64::_mm512_maskz_compress_epi8;
use core::arch::x86_64::{
    __m128i, __m512i, _mm_maskz_loadu_epi8, _mm512_add_epi8, _mm512_add_epi32, _mm512_alignr_epi32,
    _mm512_cmpeq_epi8_mask, _mm512_cmpge_epi32_mask, _mm512_cvtepu8_epi32, _mm512_loadu_si512,
    _mm512_mask_cmpeq_epi8_mask, _mm512_mask_cmpeq_epi32_mask, _mm512_mask_cmplt_epu32_mask,
    _mm512_mask_cmpneq_epi8_mask, _mm512_mask_i32gather_epi32, _mm512_maskz_compress_epi32,
    _mm512_maskz_loadu_epi8, _mm512_maskz_mov_epi32, _mm512_permutexvar_epi32, _mm512_set1_epi8,
    _mm512_set1_epi32, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_storeu_si512, _rdtsc,
};

pub type V = __m512i;

pub const BLOCK: usize = 64;

pub type Block = [u8; BLOCK];

macro_rules! intrinsics {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $intrinsic:ident;)*) => {$(
        #[inline]
        pub fn $name($($arg: $ty),*) -> $ret {
            // SAFETY: lib.rs refuses to compile unless the target enables
            // AVX-512F and BW (and this module VBMI2 and VL where used), and
            // the intrinsic does not access memory.
            unsafe { $intrinsic($($arg),*) }
        }
    )*};
}

intrinsics! {
    splat_u8(b: i8) -> V = _mm512_set1_epi8;
    splat_i32(x: i32) -> V = _mm512_set1_epi32;
    zero() -> V = _mm512_setzero_si512;
    eq8(a: V, b: V) -> u64 = _mm512_cmpeq_epi8_mask;
    eq8_masked(k: u64, a: V, b: V) -> u64 = _mm512_mask_cmpeq_epi8_mask;
    ne8_masked(k: u64, a: V, b: V) -> u64 = _mm512_mask_cmpneq_epi8_mask;
    add8(a: V, b: V) -> V = _mm512_add_epi8;
    add32(a: V, b: V) -> V = _mm512_add_epi32;
    ge32(a: V, b: V) -> u16 = _mm512_cmpge_epi32_mask;
    eq32_masked(k: u16, a: V, b: V) -> u16 = _mm512_mask_cmpeq_epi32_mask;
    lt32u_masked(k: u16, a: V, b: V) -> u16 = _mm512_mask_cmplt_epu32_mask;
    widen8to32(a: __m128i) -> V = _mm512_cvtepu8_epi32;
    select32(k: u16, a: V) -> V = _mm512_maskz_mov_epi32;
    compress32(k: u16, a: V) -> V = _mm512_maskz_compress_epi32;
    permute32(idx: V, a: V) -> V = _mm512_permutexvar_epi32;
}

#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
intrinsics! {
    compress8(k: u64, a: V) -> V = _mm512_maskz_compress_epi8;
}

#[inline]
pub fn alignr32<const N: i32>(a: V, b: V) -> V {
    // SAFETY: AVX-512F is enabled (lib.rs); no memory access.
    unsafe { _mm512_alignr_epi32::<N>(a, b) }
}

#[inline]
pub fn shl32<const N: u32>(a: V) -> V {
    // SAFETY: AVX-512F is enabled (lib.rs); no memory access.
    unsafe { _mm512_slli_epi32::<N>(a) }
}

#[inline]
pub fn splat(byte: u8) -> V {
    splat_u8(i8::from_ne_bytes([byte]))
}

#[inline]
pub const fn low_mask(k: usize) -> u64 {
    if k >= BLOCK { u64::MAX } else { (1 << k) - 1 }
}

#[inline]
pub fn head_len(s: &[u8]) -> usize {
    s.as_ptr().addr().wrapping_neg() & (BLOCK - 1)
}

#[inline]
pub fn load(block: &Block) -> V {
    // SAFETY: `block` is 64 readable bytes; the load is unaligned.
    unsafe { _mm512_loadu_si512(block.as_ptr().cast()) }
}

#[inline]
pub fn load_masked(s: &[u8], mask: u64) -> V {
    // SAFETY: the mask is cut to the first min(64, s.len()) bytes, all
    // readable; masked-off bytes are not accessed.
    unsafe { _mm512_maskz_loadu_epi8(mask & low_mask(s.len()), s.as_ptr().cast()) }
}

#[cfg(all(target_feature = "avx512vbmi2", target_feature = "avx512vl"))]
#[inline]
pub fn load_masked_16(s: &[u8], at: usize, mask: u16) -> __m128i {
    let bytes = &s[at..at + 16];
    // SAFETY: `bytes` is 16 readable bytes.
    unsafe { _mm_maskz_loadu_epi8(mask, bytes.as_ptr().cast()) }
}

#[inline]
pub fn store(s: &mut [u8], at: usize, v: V) {
    let bytes = &mut s[at..at + BLOCK];
    // SAFETY: `bytes` is 64 writable bytes; the store is unaligned.
    unsafe { _mm512_storeu_si512(bytes.as_mut_ptr().cast(), v) }
}

#[inline]
pub fn store_u32(s: &mut [u32], at: usize, v: V) {
    let words = &mut s[at..at + 16];
    // SAFETY: `words` is 64 writable bytes; the store is unaligned.
    unsafe { _mm512_storeu_si512(words.as_mut_ptr().cast(), v) }
}

#[inline]
pub fn gather_u32(mask: u16, offsets: V, base: &[u8]) -> V {
    let limit = i32::try_from(base.len().saturating_sub(3)).unwrap_or(i32::MAX);
    let in_bounds = lt32u_masked(mask, offsets, splat_i32(limit));
    // SAFETY: only lanes whose offset satisfies offset + 4 <= base.len() are
    // gathered, so every 4-byte read lies inside `base`; offsets are below
    // i32::MAX, so their sign extension is the same value.
    unsafe { _mm512_mask_i32gather_epi32::<1>(zero(), in_bounds, offsets, base.as_ptr().cast()) }
}

#[inline]
pub fn rdtsc() -> u64 {
    // SAFETY: RDTSC is present on every x86-64 processor and has no
    // preconditions.
    unsafe { _rdtsc() }
}
