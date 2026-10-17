#pragma once
// NeedleHammer: Guardzilla with a second scan for inputs that defeat the
// branch predictor.
//
// Guardzilla takes one branch per 64 haystack positions: whether any position
// passed the three-anchor filter. When candidates are rare or nearly
// everywhere, the branch is predicted. When a fair share of the blocks have
// one, at an irregular rate (DNA, some log files), it is not, and every
// misprediction costs about 27 cycles on Granite Rapids and Zen 5, as much as
// seven to ten blocks of scanning.
//
// NeedleHammer runs Guardzilla's scan and counts the blocks with a candidate,
// in the branch taken for them, per window: two windows of 64 blocks (4 KiB)
// for a quick decision, then windows of 2048 blocks (128 KiB), since leaving
// the loop at the end of a window costs a misprediction. When two consecutive
// windows have a share r of such blocks with min(r, 1 - r) above 16%, the
// expected cost of the mispredictions (27 cycles times min(r, 1 - r) per
// block) exceeds the extra cost of the second scan (4.5 to 7.1 cycles per
// block), and the search goes on with the second scan. Periodic input can be
// in the band yet predictable, so the switch is checked: the last window of
// the first scan and a sample of the second scan's chunks are timed (RDTSC),
// and when the second scan is slower, the first resumes and waits for twice
// as many windows in the band before it switches again.
//
// The second scan, over chunks of 64 blocks:
//  1. The filter's mask compresses (VPCOMPRESSB) a byte vector of positions
//     that wraps every 256 bytes; the last block of every 256-byte span always
//     emits position 255, so that each span leaves a mark. No branch depends
//     on the data.
//  2. The positions are decoded 16 at a time: a new span starts wherever a
//     position is not above the previous one (a prefix sum). A gather then
//     compares 4 more needle bytes at each candidate, and VPCOMPRESSD keeps the
//     candidates that pass.
//  3. Those are compared with the needle in order, with Guardzilla's guard.
//
// A search that ends early costs little but setup, and the windowed scan
// costs more to enter than Guardzilla's (about 36 more instructions per call).
// So the first 16 blocks (1 KiB) are scanned by a lean copy of Guardzilla's
// loop, and the windowed scan starts after them.
//
// Compressing the candidates, and the sentinel, are Geoff Langdale's ideas
// (Hyperscan). Needs AVX-512F, BW, VL and VBMI2 (Ice Lake, Sapphire
// Rapids and later; Zen 4 and later). O(n + m) time.
//
// Settings, for measurement only:
//   NEEDLEHAMMER_WINDOW  blocks in each of the first two windows (default 64)
//   NEEDLEHAMMER_LONG_WINDOW  blocks in each later window (default 2048)
//   NEEDLEHAMMER_BAND    the switch threshold, percent (default 16)
//   NEEDLEHAMMER_ANCHORS anchors of the second scan, 1 to 8 (default 3: the
//                        first scan's; other counts are evenly spaced)
//   NEEDLEHAMMER_TIMED   0 to keep the second scan without timing it (default 1)
//   NEEDLEHAMMER_PROLOGUE blocks scanned first by a lean Guardzilla loop
//                        (default 16; 0 for none)
#include "guardzilla.h"

#if !(defined(__AVX512VBMI2__) && defined(__AVX512VL__))
#error "NeedleHammer needs AVX-512 VBMI2 and VL (-mavx512vbmi2 -mavx512vl)"
#endif

#ifndef NEEDLEHAMMER_WINDOW
#define NEEDLEHAMMER_WINDOW 64
#endif
#ifndef NEEDLEHAMMER_LONG_WINDOW
#define NEEDLEHAMMER_LONG_WINDOW 2048
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
#ifndef NEEDLEHAMMER_PROLOGUE
#define NEEDLEHAMMER_PROLOGUE 16
#endif
static_assert(NEEDLEHAMMER_ANCHORS >= 1 && NEEDLEHAMMER_ANCHORS <= 8, "NEEDLEHAMMER_ANCHORS: 1 to 8");
static_assert(NEEDLEHAMMER_WINDOW >= 1 && NEEDLEHAMMER_WINDOW <= 4096, "NEEDLEHAMMER_WINDOW");

// Out of line, and opaque to GCC's interprocedural passes: when GCC sees into
// the functions that the main loop calls, it stops keeping the anchors in
// registers in that loop (GCC 14).
#if defined(__GNUC__) && !defined(__clang__)
#define NEEDLEHAMMER_OPAQUE __attribute__((noinline, noipa))
#else
#define NEEDLEHAMMER_OPAQUE __attribute__((noinline))
#endif

namespace needlehammer {
namespace detail {
using guardzilla::detail::low_mask;
using guardzilla::detail::rest_matches;

// The blocks of a chunk of the second scan (4 KiB).
constexpr size_t kChunk = 64;

// The second scan over the full blocks from the 64-byte-aligned position i
// to end, at most kChunk of them: 0 if none matches, 1 + q if q
// does, n + 2 + q if the guard trips at q.
NEEDLEHAMMER_OPAQUE inline size_t compress_chunk(const uint8_t* t, size_t n, const uint8_t* p, size_t m,
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
    constexpr size_t kMax = 64 * kChunk + kChunk / 4;   // positions and sentinels
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

// The second scan from the aligned position i, in chunks of kChunk blocks;
// the first chunk, and one in 16 after it, are timed against the cycles per
// block of the first scan's last window. Returns {r, i}: r as
// compress_chunk() when nonzero, else the position where the first scan
// resumes (the end of the full blocks, or where the second scan was slower).
NEEDLEHAMMER_OPAQUE inline std::pair<size_t, size_t> second_scan(const uint8_t* t, size_t n, const uint8_t* p,
                                                                  size_t m, const size_t o[3], size_t i,
                                                                  size_t& work, uint64_t first_per_block) {
    const size_t last = n - m;
    for (size_t k = 0; i + 63 <= last; ++k) {
        const size_t end = last - 63 - i >= 64 * kChunk ? i + 64 * kChunk : last - 62;
        const bool timed = NEEDLEHAMMER_TIMED && k % 16 == 0;
        const uint64_t start = timed ? __rdtsc() : 0;
        const size_t r = compress_chunk(t, n, p, m, o, i, end, work);
        if (r) return {r, i};
        if (timed && __rdtsc() - start > first_per_block * ((end - i) / 64)) return {0, end};
        i = end;
    }
    return {0, i};
}

// Guardzilla's scan of the candidates [0, end), end <= n - m + 1, with the
// anchors o: the first blocks, where NeedleHammer's larger scan() costs more to
// enter than the blocks cost to scan. The first match, `end` if none, or
// n + 1 + q if the budget ran out at candidate q.
inline size_t prologue(const uint8_t* t, size_t n, const uint8_t* p, size_t m, const size_t* o, size_t end) {
    const __m512i v0 = _mm512_set1_epi8(char(p[o[0]])), v1 = _mm512_set1_epi8(char(p[o[1]])),
                  v2 = _mm512_set1_epi8(char(p[o[2]]));
    const __mmask64 head_mask = low_mask(m);
    const __m512i head = _mm512_maskz_loadu_epi8(head_mask, p);
    const size_t slack = m / 8 + 64;
    size_t work = 0;
    auto verify = [&](size_t i, __mmask64 w) -> size_t {
        for (; w; w &= w - 1) {
            const size_t q = i + size_t(__builtin_ctzll(w));
            if (_mm512_mask_cmpneq_epi8_mask(head_mask, _mm512_maskz_loadu_epi8(head_mask, t + q), head)) continue;
            if (m <= 64 || rest_matches(t + q, p, m, &work)) return q + 1;
            if (work > GUARDZILLA_BUDGET * (q / 64) + slack) return n + 2 + q;
        }
        return 0;
    };
    const size_t h = (64 - (uintptr_t(t) & 63)) & 63;
    size_t i = h, r;
    if (i) {
        __mmask64 live = low_mask(i);
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + o[0]), v0);
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + o[1]), v1);
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + o[2]), v2);
        if ((r = verify(0, live))) return r - 1;
    }
    for (; i < end; i += 64) {
        const __mmask64 s = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[0]), v0) &
                            _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[1]), v1) &
                            _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o[2]), v2);
        if (s && (r = verify(i, s))) return r - 1;
    }
    return end;
}

// The first match among candidates [from, n - m], n if there is none, or
// n + 1 + q if the work budget ran out at candidate q. Requires m >= 2.
// Guardzilla's scan, in windows of NEEDLEHAMMER_WINDOW blocks, counting the
// blocks with a candidate in the branch taken for them. After `patience`
// consecutive windows in the band (the last of them timed), the search goes
// on with second_scan(); when that turns out slower, the first scan resumes
// and the patience doubles. The inner loop's exit is mispredicted once per
// window, so after the first two, which give a quick decision, the windows
// are long.
//
// The anchors are o[0..3); candidates start at `from`: 0, or a 64-byte
// boundary of the haystack.
inline size_t scan(const uint8_t* t, size_t n, const uint8_t* p, size_t m, const size_t* o_in, size_t from) {
    size_t o[3] = {o_in[0], o_in[1], o_in[2]};
    // Copies: o escapes to second_scan(), and read through it, the offsets
    // would be reloaded at every block.
    const size_t o0 = o[0], o1 = o[1], o2 = o[2];
    const __m512i v0 = _mm512_set1_epi8(char(p[o0])), v1 = _mm512_set1_epi8(char(p[o1])),
                  v2 = _mm512_set1_epi8(char(p[o2]));
    const __mmask64 head_mask = low_mask(m);
    const __m512i head = _mm512_maskz_loadu_epi8(head_mask, p);
    const size_t last = n - m, slack = m / 8 + 64;
    size_t work = 0;   // 64-byte compares beyond the first 64 needle bytes

    // As in Guardzilla.
    auto filter = [&](size_t i, __mmask64 live) {
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + i + o0), v0);
        live &= _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + i + o1), v1);
        return _mm512_mask_cmpeq_epi8_mask(live, _mm512_maskz_loadu_epi8(live, t + i + o2), v2);
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
    size_t i = from ? from : h < last + 1 ? h : last + 1;
    if (!from && i && (r = verify(0, filter(0, low_mask(i))))) return r - 1;
    size_t in_band = 0, patience = 2, windows = 0;
    while (i + 63 <= last) {
        const size_t blocks = windows++ < 2 ? NEEDLEHAMMER_WINDOW : NEEDLEHAMMER_LONG_WINDOW;
        const bool full = last - 63 - i >= 64 * blocks;
        const size_t end = full ? i + 64 * blocks : last - 62;
        const bool timed = in_band + 1 >= patience;   // this window could complete the streak
        const uint64_t start = timed ? __rdtsc() : 0;
        size_t hits = 0;
        for (; i < end; i += 64) {
            const __mmask64 s = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o0), v0) &
                                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o1), v1) &
                                _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(t + i + o2), v2);
            if (s) {
                ++hits;
                if ((r = verify(i, s))) return r - 1;
            }
        }
        const bool band = hits * 100 > blocks * NEEDLEHAMMER_BAND && hits * 100 < blocks * (100 - NEEDLEHAMMER_BAND);
        in_band = full && band ? in_band + 1 : 0;
        if (in_band >= patience && i + 63 <= last) {
            const uint64_t per_block = (__rdtsc() - start) / blocks;
            const auto x = second_scan(t, n, p, m, o, i, work, per_block);
            if (x.first) return x.first - 1;
            i = x.second;   // the second scan reached the end, or was slower
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
        size_t o[3] = {0, 1, 1};
        if (m >= 3) guardzilla::detail::sz_anchors(p, m, o);
        // A search that ends within the first blocks costs little but setup:
        // those blocks go to the lean prologue(), and scan() takes the rest.
        const size_t h = (64 - (uintptr_t(t) & 63)) & 63;
        const size_t end = h + 64 * NEEDLEHAMMER_PROLOGUE;   // candidates [0, end)
        if (NEEDLEHAMMER_PROLOGUE && end + 63 <= n - m) {
            r = detail::prologue(t, n, p, m, o, end);
            if (r == end) r = detail::scan(t, n, p, m, o, end);
        } else {
            r = detail::scan(t, n, p, m, o, 0);
        }
        if (r > n) r = guardzilla::detail::twoway(t, n, p, m, r - n - 1);   // the budget ran out
    }
    return r < n ? std::pair<bool, size_t>{true, r} : std::pair<bool, size_t>{false, 0};
}

}  // namespace needlehammer
