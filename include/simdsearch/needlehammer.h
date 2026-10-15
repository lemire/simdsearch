#pragma once
// NeedleHammer: Guardzilla with a second scan for inputs that defeat the
// branch predictor.
//
// Guardzilla takes one branch per 64 haystack positions: whether any position
// passed the three-anchor filter. When candidates are rare or nearly
// everywhere, the branch is predicted; when a fair share of the blocks have
// one, at an irregular rate (DNA, some log files), it is not, and on Intel
// Emerald Rapids every misprediction costs about 25 cycles, as much as six
// blocks of scanning. NeedleHammer counts the blocks with a candidate (in the
// branch taken for them) over windows of 64 blocks (4 KiB). When two consecutive
// windows have a share r of such blocks with min(r, 1 - r) above 16%, the
// expected misprediction cost per block (25 cycles times min(r, 1 - r)) is
// above the cost of the second scan (about 4 cycles per block), and the
// search continues with that scan, one window at a time, for as long as it
// takes fewer cycles (RDTSC, on a sample of the windows) than the last window
// of the first scan did. Periodic input can be in the band yet predictable; there the second
// scan is slower, the first scan resumes, and it waits twice as many windows
// before it tries again. The second scan:
//
//  1. Over a window, the filter's mask compresses (VPCOMPRESSB) a byte
//     vector of positions that wraps every 256 bytes; the last block of every
//     256-byte window always emits position 255, so that each window leaves a
//     mark. No branch depends on the data.
//  2. The positions are decoded 16 at a time: a new window starts wherever a
//     position is not above the previous one (a prefix sum). A gather then
//     compares 4 more needle bytes at each candidate and VPCOMPRESSD keeps the
//     candidates that pass.
//  3. Those are compared against the needle in order, with Guardzilla's guard.
//
// The idea of compressing the candidates, and the sentinel, are Geoff
// Langdale's (Hyperscan). Needs AVX-512F, AVX-512BW and AVX-512 VBMI2 (Ice
// Lake, Sapphire Rapids and later; Zen 4 and later). O(n + m) time.
//
// Settings, for measurement only:
//   NEEDLEHAMMER_WINDOW  blocks per window (default 64)
//   NEEDLEHAMMER_BAND    the switch threshold, percent (default 16)
//   NEEDLEHAMMER_ANCHORS anchors of the second scan, 1 to 8 (default 3: the
//                        first scan's; other counts are evenly spaced)
//   NEEDLEHAMMER_TIMED   0 to keep the second scan without timing it (default 1)
#include "guardzilla.h"

#if !defined(__AVX512VBMI2__)
#error "NeedleHammer needs AVX-512 VBMI2 (-mavx512vbmi2)"
#endif

#ifndef NEEDLEHAMMER_WINDOW
#define NEEDLEHAMMER_WINDOW 64
#endif
#ifndef NEEDLEHAMMER_BAND
#define NEEDLEHAMMER_BAND 16
#endif
#ifndef NEEDLEHAMMER_ANCHORS
#define NEEDLEHAMMER_ANCHORS 3
#endif
#ifndef NEEDLEHAMMER_TIMED
#define NEEDLEHAMMER_TIMED 1
#endif
static_assert(NEEDLEHAMMER_ANCHORS >= 1 && NEEDLEHAMMER_ANCHORS <= 8, "NEEDLEHAMMER_ANCHORS: 1 to 8");
static_assert(NEEDLEHAMMER_WINDOW >= 1 && NEEDLEHAMMER_WINDOW <= 4096, "NEEDLEHAMMER_WINDOW");

namespace needlehammer {
namespace detail {
using guardzilla::detail::low_mask;
using guardzilla::detail::rest_matches;

// The second scan over the full blocks from the 64-byte-aligned position i
// to end, at most NEEDLEHAMMER_WINDOW of them: 0 if none matches, 1 + q if q
// does, n + 2 + q if the guard trips at q.
__attribute__((noinline)) inline size_t compress_chunk(const uint8_t* t, size_t n, const uint8_t* p, size_t m,
                                                       const size_t o3[3], size_t i, size_t end, size_t& work) {
    constexpr int A = NEEDLEHAMMER_ANCHORS;
    const size_t slack = m / 8 + 64;
    size_t o[A];
    if (A == 3) {
        for (int j = 0; j < A; ++j) o[j] = o3[j];
    } else {
        const size_t a = size_t(A) < m ? size_t(A) : m;
        for (size_t j = 0; j < size_t(A); ++j) o[j] = a == 1 ? m - 1 : (j < a ? j : a - 1) * (m - 1) / (a - 1);
    }
    __m512i v[A];
    for (int j = 0; j < A; ++j) v[j] = _mm512_set1_epi8(char(p[o[j]]));
    const __mmask64 head_mask = low_mask(m);
    const __m512i head = _mm512_maskz_loadu_epi8(head_mask, p);
    // Candidate q: 1 + q if it matches, n + 2 + q if the guard trips, else 0.
    auto check = [&](size_t q) -> size_t {
        if (_mm512_mask_cmpneq_epi8_mask(head_mask, _mm512_maskz_loadu_epi8(head_mask, t + q), head)) return 0;
        if (m <= 64 || rest_matches(t + q, p, m, &work)) return q + 1;
        if (work > GUARDZILLA_BUDGET * (q / 64) + slack) return n + 2 + q;
        return 0;
    };
    // The 4 needle bytes of the gather: those overlapping the fewest anchors.
    size_t k = 0;
    int fewest = 4;
    for (size_t j = 0; m >= 4 && j + 4 <= m && fewest; ++j) {
        int ov = 0;
        for (int a = 0; a < A; ++a) ov += o[a] - j < 4;
        if (ov < fewest) fewest = ov, k = j;
    }
    uint32_t word = 0;
    std::memcpy(&word, p + k, m >= 4 ? 4 : 0);
    const __m512i want = _mm512_set1_epi32(int(word));
    const __mmask16 gather = m >= 4 ? __mmask16(0xFFFF) : __mmask16(0);
    const __m512i iota = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44,
                                         43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24,
                                         23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                                         1, 0);
    const __m512i step = _mm512_set1_epi8(64), zero = _mm512_setzero_si512(), one = _mm512_set1_epi32(1),
                  lane15 = _mm512_set1_epi32(15);
    constexpr size_t kMax = 64 * NEEDLEHAMMER_WINDOW + NEEDLEHAMMER_WINDOW / 4;   // positions and sentinels
    alignas(64) uint8_t pos8[kMax + 64];
    alignas(64) uint32_t pos32[kMax + 16];
    size_t r;
    {
        // 1. Filter and compress, one 64-byte block at a time.
        const size_t base = i;
        uint8_t* out = pos8;
        __m512i idx = iota;
        for (uint64_t b = 0; i < end; i += 64, ++b) {
            __mmask64 s = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[0]), v[0]);
            for (int j = 1; j < A; ++j) s &= _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[j]), v[j]);
            s |= uint64_t((b & 3) == 3) << 63;   // the sentinel, position 255 of the window
            _mm512_storeu_si512(out, _mm512_maskz_compress_epi8(s, idx));
            out += _mm_popcnt_u64(s);
            idx = _mm512_add_epi8(idx, step);
        }
        // 2. Decode 16 positions at a time, gather, keep those that pass.
        const size_t count = size_t(out - pos8);
        uint32_t* kept = pos32;
        __m512i prev = _mm512_set1_epi32(-1), window = zero;
        for (size_t j = 0; j < count; j += 16) {
            const __mmask16 lm = count - j >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << (count - j)) - 1);
            const __m512i cur = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(lm, pos8 + j));
            // A new window wherever a position is not above the one before.
            __m512i f = _mm512_maskz_mov_epi32(_mm512_cmpge_epi32_mask(_mm512_alignr_epi32(cur, prev, 15), cur), one);
            f = _mm512_add_epi32(f, _mm512_alignr_epi32(f, zero, 15));
            f = _mm512_add_epi32(f, _mm512_alignr_epi32(f, zero, 14));
            f = _mm512_add_epi32(f, _mm512_alignr_epi32(f, zero, 12));
            f = _mm512_add_epi32(f, _mm512_alignr_epi32(f, zero, 8));
            const __m512i w = _mm512_add_epi32(window, f);
            const __m512i pos = _mm512_add_epi32(_mm512_slli_epi32(w, 8), cur);
            window = _mm512_permutexvar_epi32(lane15, w);
            prev = cur;
            const __m512i got = _mm512_mask_i32gather_epi32(zero, lm & gather, pos, t + base + k, 1);
            const __mmask16 pass = (lm & ~gather) | _mm512_mask_cmpeq_epi32_mask(lm & gather, got, want);
            _mm512_storeu_si512(kept, _mm512_maskz_compress_epi32(pass, pos));
            kept += _mm_popcnt_u32(pass);
        }
        // 3. Compare in order.
        for (const uint32_t* x = pos32; x < kept; ++x)
            if ((r = check(base + *x))) return r;
    }
    return 0;
}

// The second scan from the aligned position i, one window at a time; the
// first window, and one in 16 after it, are timed against `first_cost`, the
// cycles of the first scan's last window. Returns {r, i}: r as
// compress_chunk() when nonzero, else the position where the first scan
// resumes (the end of the full blocks, or where the second scan was slower).
__attribute__((noinline)) inline std::pair<size_t, size_t> second_scan(const uint8_t* t, size_t n, const uint8_t* p,
                                                                        size_t m, const size_t o[3], size_t i,
                                                                        size_t& work, uint64_t first_cost) {
    const size_t last = n - m;
    for (size_t k = 0; i + 63 <= last; ++k) {
        const size_t end = last - 63 - i >= 64 * NEEDLEHAMMER_WINDOW ? i + 64 * NEEDLEHAMMER_WINDOW : last - 62;
        const bool timed = NEEDLEHAMMER_TIMED && k % 16 == 0;
        const uint64_t start = timed ? __rdtsc() : 0;
        const size_t r = compress_chunk(t, n, p, m, o, i, end, work);
        if (r) return {r, i};
        i = end;
        if (timed && __rdtsc() - start > first_cost) break;
    }
    return {0, i};
}

// The first match among candidates [0, n - m], n if there is none, or
// n + 1 + q if the work budget ran out at candidate q. Requires m >= 2.
// Guardzilla's scan, in windows of NEEDLEHAMMER_WINDOW blocks, counting the
// blocks with a candidate in the branch taken for them. After `patience`
// consecutive windows in the band (the last of them timed), the search goes
// on with second_scan(); when that turns out slower, the first scan resumes
// and the patience doubles.
inline size_t scan(const uint8_t* t, size_t n, const uint8_t* p, size_t m) {
    size_t o[3] = {0, 1, 1};
    if (m >= 3) guardzilla::detail::sz_anchors(p, m, o);
    const __m512i v0 = _mm512_set1_epi8(char(p[o[0]])), v1 = _mm512_set1_epi8(char(p[o[1]])),
                  v2 = _mm512_set1_epi8(char(p[o[2]]));
    const __mmask64 head_mask = low_mask(m);
    const __m512i head = _mm512_maskz_loadu_epi8(head_mask, p);
    const size_t last = n - m, slack = m / 8 + 64;
    size_t work = 0;   // 64-byte compares beyond the first 64 needle bytes

    // As in Guardzilla.
    auto filter = [&](size_t i, __mmask64 live) {
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + i + o[0]), v0);
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + i + o[1]), v1);
        return _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + i + o[2]), v2);
    };
    auto verify = [&](size_t i, __mmask64 w) -> size_t {
        for (; w; w &= w - 1) {
            const size_t q = i + size_t(__builtin_ctzll(w));
            if (_mm512_mask_cmpneq_epi8_mask(head_mask, _mm512_maskz_loadu_epi8(head_mask, t + q), head)) continue;
            if (m <= 64 || rest_matches(t + q, p, m, &work)) return q + 1;
            if (work > GUARDZILLA_BUDGET * (q / 64) + slack) return n + 2 + q;
        }
        return 0;
    };

    size_t r;
    const size_t h = (64 - (uintptr_t(t) & 63)) & 63;   // a masked head up to the first 64-byte boundary
    size_t i = h < last + 1 ? h : last + 1;
    if (i && (r = verify(0, filter(0, low_mask(i))))) return r - 1;
    size_t in_band = 0, patience = 2;
    while (i + 63 <= last) {
        const bool full = last - 63 - i >= 64 * NEEDLEHAMMER_WINDOW;
        const size_t end = full ? i + 64 * NEEDLEHAMMER_WINDOW : last - 62;
        const bool timed = in_band + 1 >= patience;   // this window could complete the streak
        const uint64_t start = timed ? __rdtsc() : 0;
        size_t hits = 0;
        for (; i < end; i += 64) {
            const __mmask64 s = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[0]), v0) &
                                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[1]), v1) &
                                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[2]), v2);
            if (s) {
                ++hits;
                if ((r = verify(i, s))) return r - 1;
            }
        }
        const bool band = hits * 100 > NEEDLEHAMMER_WINDOW * NEEDLEHAMMER_BAND &&
                          hits * 100 < NEEDLEHAMMER_WINDOW * (100 - NEEDLEHAMMER_BAND);
        in_band = full && band ? in_band + 1 : 0;
        if (in_band >= patience && i + 63 <= last) {
            const auto x = second_scan(t, n, p, m, o, i, work, __rdtsc() - start);
            if (x.first) return x.first - 1;
            i = x.second;
            in_band = 0;
            patience *= 2;
        }
    }
    if (i <= last && (r = verify(i, filter(i, low_mask(last - i + 1))))) return r - 1;
    return n;
}

}  // namespace detail

// First occurrence of pattern[0..m) in text[0..n): {true, index} or
// {false, 0}. The empty needle matches at 0. No byte outside either buffer is
// read. O(n + m) time.
inline std::pair<bool, size_t> find(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const auto* t = reinterpret_cast<const uint8_t*>(text);
    const auto* p = reinterpret_cast<const uint8_t*>(pattern);
    size_t r;
    if (m == 1) {
        const void* x = std::memchr(t, p[0], n);
        r = x ? size_t(static_cast<const uint8_t*>(x) - t) : n;
    } else {
        r = detail::scan(t, n, p, m);
        if (r > n) r = guardzilla::detail::twoway(t, n, p, m, r - n - 1);   // the budget ran out
    }
    return r < n ? std::pair<bool, size_t>{true, r} : std::pair<bool, size_t>{false, 0};
}

}  // namespace needlehammer
