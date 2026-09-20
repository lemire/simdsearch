#pragma once
// Crochemore-Perrin two-way with vectorized comparison loops (AVX-512BW, or
// NEON on AArch64).
//
// The scalar two-way (kmp_twoway.h) spends its time in two byte loops,
//     while (i < m && nd[i] == hay[i + j]) ++i;          // right half, forward
//     while (d > memory && nd[d-1] == hay[d-1+j]) --d;   // left half, backward
// This version compares 64 bytes at a time in both (first mismatch by tzcnt /
// lzcnt, the needle chunk around the critical position held in a register),
// after a four-byte scalar prefix: in the regime where nearly every window
// mismatches within a byte or two, a vector verify is pure per-window overhead
// (measured 4x slower than the scalar loop on alternating text), so the first
// bytes are tested the way the scalar algorithm tests them and the vector path
// is entered only once they match.
//
// No filter: this is the linear-time fallback a filtering kernel resumes with
// once its work budget is spent, and it must stay linear on every input. The
// shift logic is exactly the scalar algorithm's.
#include <cstddef>
#include <cstdint>
#include <utility>
#include "twoway_prep.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

namespace twoway_simd {

// Mask of the k low bits, k <= 64.
static inline __mmask64 low_mask(size_t k) { return k >= 64 ? ~(__mmask64)0 : ((__mmask64)1 << k) - 1; }
// Leading zero count with lzcnt semantics (64 for zero).
static inline size_t lz64(uint64_t x) { return x ? (size_t)__builtin_clzll(x) : 64; }

// Leading equal bytes of a[0..len) vs b[0..len).
static inline size_t lcp(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t k = 0;
    for (; k + 64 <= len; k += 64) {
        __mmask64 ne = _mm512_cmpneq_epi8_mask(_mm512_loadu_si512(a + k),
                                               _mm512_loadu_si512(b + k));
        if (ne) return k + (size_t)__builtin_ctzll(ne);
    }
    if (k < len) {
        __mmask64 msk = low_mask(len - k);
        __mmask64 ne = _mm512_cmpneq_epi8_mask(_mm512_maskz_loadu_epi8(msk, a + k),
                                               _mm512_maskz_loadu_epi8(msk, b + k));
        if (ne) return k + (size_t)__builtin_ctzll(ne);
    }
    return len;
}

// Trailing equal bytes of a[0..len) vs b[0..len), scanning from the end.
// The tail compare loads k < 64 bytes with a zero mask; lanes [k, 64) of BOTH
// operands are then zero, hence equal, so `ne` has no bit at or above k and
// lzcnt counts (64 - k) unset high lanes plus the trailing equal bytes. The
// `len - 64 + lzcnt` below relies on that; a compare-under-mask form would
// need `len - k + (lzcnt - (64 - k))` instead.
static inline size_t lcs(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t k = len;
    while (k >= 64) {
        __mmask64 ne = _mm512_cmpneq_epi8_mask(_mm512_loadu_si512(a + k - 64),
                                               _mm512_loadu_si512(b + k - 64));
        if (ne) return (len - k) + lz64(ne);
        k -= 64;
    }
    if (k) {
        __mmask64 msk = low_mask(k);
        __mmask64 ne = _mm512_cmpneq_epi8_mask(_mm512_maskz_loadu_epi8(msk, a),
                                               _mm512_maskz_loadu_epi8(msk, b));
        if (ne) return len - 64 + lz64(ne);
    }
    return len;
}

struct prep {
    twoway_prep tw;
    __m512i rv;      // needle[crit .. crit+rl), zero padded
    __mmask64 rm;    // rl low bits
    size_t rl;
    __m512i lv;      // needle[crit-ll .. crit) in lanes [0, ll)
    __mmask64 lm;
    size_t ll;
    bool fits;       // m <= 64: whole needle in nv, so one compare verifies
    __m512i nv;
    __mmask64 nm;

    void build(const char* needle, size_t m) {
        tw.build(needle, m);
        if (m == 0) return;
        const uint8_t* nd = (const uint8_t*)needle;
        size_t crit = tw.crit;
        rl = m - crit < 64 ? m - crit : 64;
        rm = low_mask(rl);
        rv = _mm512_maskz_loadu_epi8(rm, nd + crit);
        ll = crit < 64 ? crit : 64;
        lm = low_mask(ll);
        lv = _mm512_maskz_loadu_epi8(lm, nd + crit - ll);
        fits = m <= 64;
        nm = low_mask(m);
        nv = _mm512_maskz_loadu_epi8(nm, nd);
    }
};

// Right half of window j, starting at max(crit, memory). Returns m on a full
// match, else the index of the first mismatch.
static inline __attribute__((always_inline)) size_t
right_scan(const prep& P, const uint8_t* hay, size_t n, const uint8_t* nd,
           size_t m, size_t j, size_t memory) {
    const size_t crit = P.tw.crit;
    size_t i0 = memory > crit ? memory : crit;
    {   // scalar prefix, see the header comment
        size_t lim = i0 + 4 < m ? i0 + 4 : m;
        size_t i = i0;
        while (i < lim && nd[i] == hay[j + i]) ++i;
        if (i < lim) return i;
        if (i == m) return m;
        i0 = i;
    }
    if (i0 - crit >= P.rl) return i0 + lcp(nd + i0, hay + j + i0, m - i0);
    __mmask64 hm = low_mask(n - j - crit);
    __m512i hv = _mm512_maskz_loadu_epi8(hm, hay + j + crit);
    __mmask64 ne = _mm512_mask_cmpneq_epi8_mask(P.rm, P.rv, hv);
    ne &= ~0ULL << (i0 - crit);
    if (ne) return crit + (size_t)__builtin_ctzll(ne);
    size_t i = crit + P.rl;
    if (i < m) i += lcp(nd + i, hay + j + i, m - i);
    return i;
}

// Left half: does nd[memory..crit) == hay[j+memory .. j+crit)?
static inline __attribute__((always_inline)) bool
left_ok(const prep& P, const uint8_t* hay, const uint8_t* nd, size_t j, size_t memory) {
    const size_t crit = P.tw.crit;
    if (memory >= crit) return true;
    const size_t need = crit - memory;
    __m512i hv = _mm512_maskz_loadu_epi8(P.lm, hay + j + crit - P.ll);
    __mmask64 ne = _mm512_cmpneq_epi8_mask(P.lv, hv);
    size_t t = lz64(ne) - (64 - P.ll);  // trailing equal bytes in chunk
    if (t >= need) return true;
    if (t < P.ll) return false;
    size_t rem = need - P.ll;
    return lcs(nd + memory, hay + j + memory, rem) == rem;
}

// Verify window j in two-way order. Returns m on a full match; the index of
// the first right-half mismatch when < m; or LEFT_FAIL when the right half
// matched but the left half did not.
static constexpr size_t LEFT_FAIL = SIZE_MAX;

template <bool FITS>
static inline __attribute__((always_inline)) size_t
verify(const prep& P, const uint8_t* hay, size_t n, const uint8_t* nd, size_t m,
       size_t j, size_t memory) {
    const size_t crit = P.tw.crit;
    if constexpr (FITS) {
        {   // scalar prefix
            size_t i0 = memory > crit ? memory : crit;
            size_t lim = i0 + 4 < m ? i0 + 4 : m;
            size_t i = i0;
            while (i < lim && nd[i] == hay[j + i]) ++i;
            if (i < lim) return i;
        }
        // Whole needle in one register: one load, one compare, then the mask is
        // split into right-half bits (>= max(crit, memory)) and left-half bits
        // ([memory, crit)). memory <= m - 1 and crit <= m - 1, so the shifts
        // below are by less than 64.
        __mmask64 hm = low_mask(n - j);
        __mmask64 ne = _mm512_mask_cmpneq_epi8_mask(P.nm, P.nv, _mm512_maskz_loadu_epi8(hm, hay + j));
        size_t i0 = memory > crit ? memory : crit;
        __mmask64 r = ne & (~0ULL << i0);
        if (r) return (size_t)__builtin_ctzll(r);
        __mmask64 l = ne & low_mask(crit) & (~0ULL << memory);
        return l ? LEFT_FAIL : m;
    } else {
        size_t i = right_scan(P, hay, n, nd, m, j, memory);
        if (i < m) return i;
        return left_ok(P, hay, nd, j, memory) ? m : LEFT_FAIL;
    }
}

template <bool FITS>
static inline std::pair<bool, size_t>
search_impl(const prep& P, const char* text, size_t n, const char* pat, size_t m) {
    const uint8_t* hay = (const uint8_t*)text;
    const uint8_t* nd = (const uint8_t*)pat;
    const size_t crit = P.tw.crit, period = P.tw.period;
    const bool periodic = P.tw.periodic;
    const size_t last = n - m;
    size_t j = 0, memory = 0;
    // The first right-half byte is tested with a plain branch so that the
    // common "mismatch at once, shift by one" path stays speculative instead
    // of serialising on a vector compare.
    while (j <= last) {
        size_t i0 = memory > crit ? memory : crit;
        if (hay[j + i0] != nd[i0]) {
            j += i0 - crit + 1;
            memory = 0;
            continue;
        }
        size_t i = verify<FITS>(P, hay, n, nd, m, j, memory);
        if (i == m) return {true, j};
        if (i == LEFT_FAIL) {
            j += period;
            memory = periodic ? m - period : 0;
        } else {
            j += i - crit + 1;
            memory = 0;
        }
    }
    return {false, 0};
}

static inline std::pair<bool, size_t>
search(const prep& P, const char* text, size_t n, const char* pat, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    if (P.fits) return search_impl<true>(P, text, n, pat, m);
    return search_impl<false>(P, text, n, pat, m);
}

// Stateless entry point: preprocess, then search text[from..n).
static inline std::pair<bool, size_t>
search_from(const char* text, size_t n, const char* pat, size_t m, size_t from) {
    if (from + m > n) return {false, 0};
    prep P; P.build(pat, m);
    auto [f, idx] = search(P, text + from, n - from, pat, m);
    if (!f) return {false, 0};
    return {true, from + idx};
}

}  // namespace twoway_simd

#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace twoway_simd {

// Lane bits of a 0x00/0xFF vector, one bit per byte lane at bit 4k+3.
static inline uint64_t lane_mask(uint8x16_t v) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0) & 0x8888888888888888ull;
}

// Leading equal bytes of a[0..len) vs b[0..len): 16 bytes per compare, the
// first differing lane by ctz of the mismatch mask; the last partial block
// byte by byte.
static inline size_t lcp(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t k = 0;
    for (; k + 16 <= len; k += 16) {
        const uint64_t ne = lane_mask(vmvnq_u8(vceqq_u8(vld1q_u8(a + k), vld1q_u8(b + k))));
        if (ne) return k + ((size_t)__builtin_ctzll(ne) >> 2);
    }
    while (k < len && a[k] == b[k]) ++k;
    return k;
}

// Trailing equal bytes of a[0..len) vs b[0..len), scanning from the end.
static inline size_t lcs(const uint8_t* a, const uint8_t* b, size_t len) {
    size_t k = len;
    while (k >= 16) {
        const uint64_t ne = lane_mask(vmvnq_u8(vceqq_u8(vld1q_u8(a + k - 16), vld1q_u8(b + k - 16))));
        if (ne) {
            const size_t lane = (size_t)(63 - __builtin_clzll(ne)) >> 2;   // highest differing lane
            return (len - k) + (15 - lane);
        }
        k -= 16;
    }
    size_t t = 0;
    while (t < k && a[k - 1 - t] == b[k - 1 - t]) ++t;
    return (len - k) + t;
}

struct prep {
    twoway_prep tw;
    void build(const char* needle, size_t m) { tw.build(needle, m); }
};

// Verify window j in two-way order: m on a full match, the index of the first
// right-half mismatch when < m, or LEFT_FAIL when the right half matched but
// the left half did not. The first four bytes of the right half are compared
// scalar-ly, as in the AVX-512 version: on inputs where nearly every window
// mismatches at once a vector compare is pure per-window overhead.
static constexpr size_t LEFT_FAIL = SIZE_MAX;

static inline size_t verify(const prep& P, const uint8_t* hay, const uint8_t* nd, size_t m,
                            size_t j, size_t memory) {
    const size_t crit = P.tw.crit;
    size_t i = memory > crit ? memory : crit;
    const size_t lim = i + 4 < m ? i + 4 : m;
    while (i < lim && nd[i] == hay[j + i]) ++i;
    if (i < lim) return i;
    if (i < m) i += lcp(nd + i, hay + j + i, m - i);
    if (i < m) return i;
    if (memory >= crit) return m;
    const size_t need = crit - memory;
    return lcs(nd + memory, hay + j + memory, need) == need ? m : LEFT_FAIL;
}

static inline std::pair<bool, size_t>
search(const prep& P, const char* text, size_t n, const char* pat, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const uint8_t* hay = (const uint8_t*)text;
    const uint8_t* nd = (const uint8_t*)pat;
    const size_t crit = P.tw.crit, period = P.tw.period;
    const bool periodic = P.tw.periodic;
    const size_t last = n - m;
    size_t j = 0, memory = 0;
    while (j <= last) {
        const size_t i0 = memory > crit ? memory : crit;
        if (hay[j + i0] != nd[i0]) { j += i0 - crit + 1; memory = 0; continue; }
        const size_t i = verify(P, hay, nd, m, j, memory);
        if (i == m) return {true, j};
        if (i == LEFT_FAIL) { j += period; memory = periodic ? m - period : 0; }
        else { j += i - crit + 1; memory = 0; }
    }
    return {false, 0};
}

static inline std::pair<bool, size_t>
search_from(const char* text, size_t n, const char* pat, size_t m, size_t from) {
    if (from + m > n) return {false, 0};
    prep P; P.build(pat, m);
    auto [f, idx] = search(P, text + from, n - from, pat, m);
    if (!f) return {false, 0};
    return {true, from + idx};
}

}  // namespace twoway_simd

#else
#error "twoway_simd.h: no SIMD backend (needs AVX-512F+BW or AArch64 NEON)"
#endif

// Stateless entry point in the style of the other searchers.
inline std::pair<bool, size_t> twoway_simd_search(const char* text, size_t n, const char* pattern, size_t m) {
    return twoway_simd::search_from(text, n, pattern, m, 0);
}
