#pragma once
// Needle-Hammer: one wide anchored kernel with a work-counting guard.
//
// The filter tests every candidate position on three or four needle bytes
// chosen for selectivity (first, middle, last, and a quarter point or a byte
// that is rare in the needle), 256 positions per iteration, then narrows the
// survivors byte by byte. Three anchors are used while the haystack shows
// they suffice; a fourth is added when survivors become frequent. Narrowing
// rounds are counted, and once they exceed a budget proportional to the
// haystack the search resumes with a linear-time two-way, so every input is
// searched in linear time. Needles of one to three bytes take a dedicated
// stride loop with no verification at all.
//
// Dispatch, in order:
//   m == 0                     match at 0
//   m <= 3                     short_search: M compares per chunk, a survivor is a match
//   n < m                      no match
//   n < 1024 or n < m + 255    masked 64-byte windows with the anchors (small haystack)
//   m == 4                     the four anchors are bytes 0..3: avx512_naive_search256_body
//   m <= 36                    wide kernel, no guard (narrowing bounded by construction)
//   otherwise                  wide kernel, guard budget n/64 rounds; a three-anchor
//                              start escalates to four, then resumes with two-way
//
// Requires AVX-512F/BW/VL/DQ and BMI2. Shares avx512_align_head,
// avx512_naive_search_body and avx512_naive_search256_body with the earlier
// kernels in avx512search.h.
#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include "avx512search.h"
#include "twoway_simd.h"

namespace needle_hammer {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Guard budget: narrowing rounds allowed before giving up, as a fraction of
// the haystack. One round is one broadcast plus four masked compares over a
// 256-byte block, about five cycles; the fallback two-way runs at 2-40 GB/s
// depending on the input, so n/64 rounds (~0.1 cycle per haystack byte) keeps
// a search that gives up within a small factor of the fallback alone.
static constexpr size_t kBudgetDen = 64;

// Below this needle length the guard is absent. A block admits at most m - 4
// narrowing rounds, so the whole haystack admits at most n(m - 4)/256 rounds,
// which for m <= 36 is at most n/8: bounded by construction, and cheaper than
// what two-way costs on the short periodic needles that would trip a guard
// (8 GB/s of narrowing against 2 GB/s of two-way on the block shape at L=16).
static constexpr size_t kFreeBelow = 36;

// Haystacks shorter than this take masked 64-byte windows rather than the
// 256-byte stride, whose alignment head and loop bound need room to pay off.
static constexpr size_t kMinWide = 1024;

// Needles up to this length start with three anchors; longer ones start with
// four. Measured on Emerald Rapids and Zen 5, a three-anchor start wins or
// ties at every length once survivors escalate it to four when needed, so
// there is no cap by default. Overridable at compile time
// (-DNH2_THREE_ANCHOR_MAX=128, say) so the switch point can be re-fitted
// per machine rather than trusted.
#ifndef NH2_THREE_ANCHOR_MAX
#define NH2_THREE_ANCHOR_MAX SIZE_MAX
#endif
static constexpr size_t kThreeAnchorMax = NH2_THREE_ANCHOR_MAX;

// Cost of a surviving lane, in filter-compare units, for the escalation rule
// below: a mispredicted branch plus, for long needles, the restart of the far
// load streams the anchors at m/2 and m-1 keep in flight.
static inline size_t survivor_cost(size_t m) { return 20 + m / 32; }

// ---------------------------------------------------------------------------
// Anchor selection
// ---------------------------------------------------------------------------

struct anchors {
    size_t o[4];   // offsets into the needle, ascending; o[3] is the spare when k == 3
    int k;         // anchors in use: 3 or 4
};

// Positional anchors: first, quarter, middle and last byte, each walked
// inward (at most 64 steps) so the four byte values are distinct where
// possible. The walk rules are StringZilla's: the middle anchor moves only
// while it equals the first byte, since it is the anchor most likely to sit on
// an anomaly and a stricter rule walks it off that anomaly on a two-letter
// alphabet; the last moves while it equals first or middle; the quarter point
// moves while it equals the first byte.
static inline void positional(const unsigned char* s, size_t m, size_t o[4]) {
    o[0] = 0; o[1] = m / 4; o[2] = m / 2; o[3] = m - 1;
    { int w = 64; while (w-- && s[o[1]] == s[o[0]] && o[1] + 1 < o[2]) ++o[1]; }
    { int w = 64; while (w-- && s[o[2]] == s[o[0]] && o[2] + 1 < o[3]) ++o[2]; }
    { int w = 64; while (w-- && (s[o[3]] == s[o[2]] || s[o[3]] == s[o[0]]) && o[3] > o[2] + 1) --o[3]; }
}

// Select the anchors for a needle of m >= 5 bytes.
//
// After the positional choice, one pass over the needle (four vector compares
// per 64 bytes) counts the bytes whose value is NOT one of the four anchor
// values. If those are few (at most max(1, m/8)) but not none, the first of
// them is a byte rare in the needle -- and, the needle being a substring of
// the text it matches, rare at that offset in any window that could match --
// so it replaces the quarter point and all four anchors are used. That
// catches a single odd byte wherever it sits. If there are none, the needle
// is made of the anchor values alone (a tiny alphabet, or a run of one byte):
// four anchors, since each cuts candidates by little. Otherwise the alphabet
// is wide and three anchors -- first, middle, last -- are enough to start
// with; the quarter point is kept as the spare the kernel adds if survivors
// turn out to be frequent.
static inline anchors select(const char* pattern, size_t m) {
    const unsigned char* s = (const unsigned char*)pattern;
    anchors a;
    positional(s, m, a.o);
    const __m512i v0 = _mm512_set1_epi8((char)s[a.o[0]]), v1 = _mm512_set1_epi8((char)s[a.o[1]]);
    const __m512i v2 = _mm512_set1_epi8((char)s[a.o[2]]), v3 = _mm512_set1_epi8((char)s[a.o[3]]);
    const size_t limit = m / 8 > 1 ? m / 8 : 1;
    size_t outside = 0, first_out = m;
    for (size_t k = 0; k < m; k += 64) {
        const size_t rem = m - k;
        const __mmask64 lanes = rem >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << rem) - 1);
        const __m512i x = _mm512_maskz_loadu_epi8(lanes, s + k);
        const __mmask64 in = _mm512_cmpeq_epi8_mask(x, v0) | _mm512_cmpeq_epi8_mask(x, v1)
                           | _mm512_cmpeq_epi8_mask(x, v2) | _mm512_cmpeq_epi8_mask(x, v3);
        const __mmask64 out = lanes & ~in;
        if (out) {
            if (first_out == m) first_out = k + (size_t)__builtin_ctzll(out);
            outside += (size_t)__builtin_popcountll(out);
            if (outside > limit) break;
        }
    }
    if (outside == 0) {
        a.k = 4;                              // needle made of the anchor values alone
    } else if (outside <= limit) {
        a.o[1] = first_out;                   // a rare byte: the best anchor there is
        a.k = 4;
    } else if (m > kThreeAnchorMax || m <= 8) {
        a.k = 4;
    } else {
        // three to start: first, middle, last; the quarter point is the spare
        const size_t quarter = a.o[1];
        a.o[1] = a.o[2]; a.o[2] = a.o[3]; a.o[3] = quarter;
        a.k = 3;
    }
    std::sort(a.o, a.o + a.k);
    return a;
}

// ---------------------------------------------------------------------------
// Masked windows: 64 candidate positions per step, every load masked to the
// live lanes, the anchors tested first and the remaining needle bytes in
// order. Used for the alignment head, the remainder past the stride loop,
// and whole small haystacks. Candidate p is live only while p + m <= n, so
// every load text + p + k with k < m stays inside the buffer.
// ---------------------------------------------------------------------------
template <int K>
static inline __attribute__((always_inline)) std::pair<bool, size_t>
masked_windows(const char* text, size_t n, const char* pattern, size_t m,
               size_t first, size_t last_exclusive, const size_t* o, const __m512i* pv) {
    for (size_t i = first; i < last_exclusive; i += 64) {
        const size_t cand = last_exclusive - i;
        const __mmask64 active = (cand >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << cand) - 1);
        __mmask64 f = active;
        for (int k = 0; k < K && f != 0; ++k)
            f = _mm512_mask_cmpeq_epi8_mask(f, _mm512_maskz_loadu_epi8(f, text + i + o[k]), pv[k]);
        for (size_t k = 0; k < m && f != 0; ++k) {
            bool is_anchor = false;
            for (int q = 0; q < K; ++q) is_anchor |= (k == o[q]);
            if (is_anchor) continue;
            f = _mm512_mask_cmpeq_epi8_mask(f, _mm512_maskz_loadu_epi8(f, text + i + k),
                                            _mm512_set1_epi8((char)pattern[k]));
        }
        if (f != 0) return {true, i + (size_t)__builtin_ctzll(f)};
    }
    return {false, 0};
}

// ---------------------------------------------------------------------------
// The wide kernel
// ---------------------------------------------------------------------------
struct result {
    bool found;
    size_t index;
    int state;       // 0 done; 1 budget exhausted; 2 escalate to four anchors
    size_t resume;   // first position not yet ruled out, for states 1 and 2
    size_t rounds;   // narrowing rounds counted so far (Guarded only)
};

// 256 positions per iteration as four 64-byte chunks; each chunk loads the
// haystack at the chunk base plus each anchor offset and ANDs the compares.
// Loads reach text + i + 192 + 63 + o[K-1] <= text + i + 255 + m - 1, inside
// the buffer by the loop bound. A lone survivor in the block is verified in
// place; several are narrowed together, one needle byte per round, the rounds
// counted when Guarded. With K == 3 the block's survivors are also counted
// and the kernel returns state 2 when they are costing more than a fourth
// anchor would (the same rule for K == 4 would only ever say "keep four").
template <bool Guarded, int K>
static inline result
wide(const char* text, size_t n, const char* pattern, size_t m,
     const anchors& a, size_t budget_rounds, size_t rounds, size_t start) {
    static_assert(K == 3 || K == 4);
    const bool fits = (m <= 64);
    const __mmask64 nmask = fits ? ((m == 64) ? ~(__mmask64)0 : (((__mmask64)1 << m) - 1)) : (__mmask64)0;
    const __m512i nvec = fits ? _mm512_maskz_loadu_epi8(nmask, pattern) : _mm512_setzero_si512();
    const size_t o0 = a.o[0], o1 = a.o[1], o2 = a.o[2], o3 = a.o[K - 1];
    const __m512i p0 = _mm512_set1_epi8((char)pattern[o0]);
    const __m512i p1 = _mm512_set1_epi8((char)pattern[o1]);
    const __m512i p2 = _mm512_set1_epi8((char)pattern[o2]);
    const __m512i p3 = _mm512_set1_epi8((char)pattern[o3]);
    const size_t oo[4] = {o0, o1, o2, o3};
    const __m512i pv[4] = {p0, p1, p2, p3};
    // One base pointer per anchor, indexed by the block position: keeps the
    // loop on base+index addressing with a single induction variable.
    const char* const t0 = text + o0; const char* const t1 = text + o1;
    const char* const t2 = text + o2; const char* const t3 = text + o3;
    size_t i = start;
    if (start == 0) {
        // Walk to a 64-byte boundary so the offset-0 load of every chunk is
        // aligned (a quarter of the loads; the others split lines whatever
        // the base). One masked window covers the positions skipped.
        const size_t head = avx512_align_head(text, n, m);
        if (head) {
            auto r = masked_windows<K>(text, n, pattern, m, 0, head, oo, pv);
            if (r.first) return {true, r.second, 0, 0, rounds};
            i = head;
        }
    }
    size_t blocks = 0, survivors = 0;
#define NH2_CHUNK(OFF)                                                                            \
    (K == 4                                                                                       \
     ? ((_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t0+i+(OFF))), p0)                \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t1+i+(OFF))), p1))               \
       & (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t2+i+(OFF))), p2)               \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t3+i+(OFF))), p3)))              \
     : ((_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t0+i+(OFF))), p0)                \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t1+i+(OFF))), p1))               \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t2+i+(OFF))), p2)))
    const size_t last_block = n - m - 255;   // n >= m + 255 here
    for (; i <= last_block; i += 256) {
        __mmask64 fA = NH2_CHUNK(0), fB = NH2_CHUNK(64);
        __mmask64 fC = NH2_CHUNK(128), fD = NH2_CHUNK(192);
        if constexpr (K == 3) ++blocks;
        if ((fA | fB | fC | fD) == 0) continue;
        if constexpr (K == 3) {
            survivors += (size_t)__builtin_popcountll(fA) + (size_t)__builtin_popcountll(fB)
                       + (size_t)__builtin_popcountll(fC) + (size_t)__builtin_popcountll(fD);
            // Dropping the fourth anchor saves about two cycles per block;
            // escalate once the survivors it would have removed cost more,
            // after a short warm-up.
            if (survivors * survivor_cost(m) > 2 * blocks + 128) return {false, 0, 2, i, rounds};
        }
        const int nz = (fA != 0) + (fB != 0) + (fC != 0) + (fD != 0);
        const __mmask64 one = fA | fB | fC | fD;
        if (nz == 1 && (one & (one - 1)) == 0) {
            const size_t off = (fA != 0) ? 0 : (fB != 0) ? 64 : (fC != 0) ? 128 : 192;
            const size_t b = off + (size_t)__builtin_ctzll(one);
            if (fits) {
                if (_mm512_mask_cmpneq_epi8_mask(nmask, _mm512_maskz_loadu_epi8(nmask, text + i + b), nvec) == 0)
                    return {true, i + b, 0, 0, rounds};
            } else if (_mm512_cmpneq_epi8_mask(_mm512_loadu_si512((const void*)(text + i + b)),
                                               _mm512_loadu_si512((const void*)pattern)) == 0
                       && std::memcmp(text + i + b + 64, pattern + 64, m - 64) == 0) {
                return {true, i + b, 0, 0, rounds};
            }
            continue;
        }
        for (size_t k = 0; k < m; ++k) {
            if (k == o0 || k == o1 || k == o2 || k == o3) continue;
            if ((fA | fB | fC | fD) == 0) break;
            const __m512i pk = _mm512_set1_epi8((char)pattern[k]);
            fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text+i+k+  0)), pk);
            fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text+i+k+ 64)), pk);
            fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text+i+k+128)), pk);
            fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text+i+k+192)), pk);
            if constexpr (Guarded) ++rounds;
        }
        if (fA) return {true, i +   0 + (size_t)__builtin_ctzll(fA), 0, 0, rounds};
        if (fB) return {true, i +  64 + (size_t)__builtin_ctzll(fB), 0, 0, rounds};
        if (fC) return {true, i + 128 + (size_t)__builtin_ctzll(fC), 0, 0, rounds};
        if (fD) return {true, i + 192 + (size_t)__builtin_ctzll(fD), 0, 0, rounds};
        // Every position in [i, i + 256) is ruled out here, so a give-up
        // resumes past the block.
        if constexpr (Guarded) { if (rounds > budget_rounds) return {false, 0, 1, i + 256, rounds}; }
    }
#undef NH2_CHUNK
    auto r = masked_windows<K>(text, n, pattern, m, i, n - m + 1, oo, pv);
    return {r.first, r.second, 0, 0, rounds};
}

// ---------------------------------------------------------------------------
// Short needles, m in {1, 2, 3}: a survivor is a match, no verification.
// ---------------------------------------------------------------------------
template <int M>
static inline __attribute__((always_inline)) std::pair<bool, size_t>
short_search(const char* text, size_t n, const char* pattern) {
    static_assert(M >= 1 && M <= 3);
    if constexpr (M == 1) {
        // memchr: one masked compare per 64 bytes from the first byte on.
        const __m512i v = _mm512_set1_epi8((char)pattern[0]);
        for (size_t i = 0; i < n; i += 64) {
            const size_t cand = n - i;
            const __mmask64 active = (cand >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << cand) - 1);
            const __mmask64 eq = _mm512_mask_cmpeq_epi8_mask(active, _mm512_maskz_loadu_epi8(active, text + i), v);
            if (eq) return {true, i + (size_t)__builtin_ctzll(eq)};
        }
        return {false, 0};
    } else {
        const size_t m = M;
        if (n < m) return {false, 0};
        size_t i = 0;
        const __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
        const __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
        const __m512i p2 = _mm512_set1_epi8((char)pattern[M > 2 ? 2 : 1]);
        auto chunk = [&](size_t off) -> __mmask64 {
            __mmask64 f = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+off)), p0)
                        & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+off+1)), p1);
            if constexpr (M > 2) f &= _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+off+2)), p2);
            return f;
        };
        if (n >= m + 255) {
            // Most short-needle searches on text end within the first 64
            // bytes: look there first with one unaligned window, then align.
            {
                const __mmask64 f = chunk(0);
                if (f) return {true, (size_t)__builtin_ctzll(f)};
                i = 64;
            }
            const size_t head = avx512_align_head(text + i, n - i, m);
            if (head) {
                auto r = avx512_masked_head(text + i, pattern, m, head);
                if (r.first) return {true, i + r.second};
                i += head;
            }
            for (; i + m + 255 <= n; i += 256) {
                const __mmask64 fA = chunk(0), fB = chunk(64), fC = chunk(128), fD = chunk(192);
                if ((fA | fB | fC | fD) == 0) continue;
                if (fA) return {true, i +   0 + (size_t)__builtin_ctzll(fA)};
                if (fB) return {true, i +  64 + (size_t)__builtin_ctzll(fB)};
                if (fC) return {true, i + 128 + (size_t)__builtin_ctzll(fC)};
                return {true, i + 192 + (size_t)__builtin_ctzll(fD)};
            }
        }
        for (; i + m <= n; i += 64) {
            const size_t cand = n - m - i + 1;
            __mmask64 f = (cand >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << cand) - 1);
            for (size_t k = 0; k < m && f != 0; ++k)
                f = _mm512_mask_cmpeq_epi8_mask(f, _mm512_maskz_loadu_epi8(f, text + i + k),
                                                _mm512_set1_epi8((char)pattern[k]));
            if (f != 0) return {true, i + (size_t)__builtin_ctzll(f)};
        }
        return {false, 0};
    }
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

// Out of line so that the short-needle entry does not pay this function's
// prologue (callee-saved registers and a 64-byte-aligned frame for the vector
// arrays) on a search that ends in its first window.
template <bool Guarded>
[[gnu::noinline]] static std::pair<bool, size_t>
search_long(const char* text, size_t n, const char* pattern, size_t m) {
    if (n < m) return {false, 0};
    if (n < kMinWide || n < m + 255) {
        if (m == 4) return avx512_naive_search_body(text, n, pattern, m);
        const anchors a = select(pattern, m);
        __m512i pv[4];
        for (int k = 0; k < a.k; ++k) pv[k] = _mm512_set1_epi8((char)pattern[a.o[k]]);
        return a.k == 3 ? masked_windows<3>(text, n, pattern, m, 0, n - m + 1, a.o, pv)
                        : masked_windows<4>(text, n, pattern, m, 0, n - m + 1, a.o, pv);
    }
    if (m == 4) return avx512_naive_search256_body(text, n, pattern, m);
    const anchors a = select(pattern, m);
    const size_t budget = Guarded ? n / kBudgetDen + 1 : ~(size_t)0;
    const bool guarded = Guarded && m > kFreeBelow;
    result r;
    if (a.k == 4) {
        r = guarded ? wide<true, 4>(text, n, pattern, m, a, budget, 0, 0)
                    : wide<false, 4>(text, n, pattern, m, a, budget, 0, 0);
    } else {
        r = guarded ? wide<true, 3>(text, n, pattern, m, a, budget, 0, 0)
                    : wide<false, 3>(text, n, pattern, m, a, budget, 0, 0);
        if (r.state != 0) {
            // Survivors too frequent (2), or the budget spent with three
            // anchors (1): continue with four. A budget give-up gets a fresh
            // budget, so the bound is 2n/64 rounds plus the two-way pass.
            anchors a4 = a; a4.k = 4; std::sort(a4.o, a4.o + 4);
            const size_t carried = (r.state == 1) ? 0 : r.rounds;
            r = guarded ? wide<true, 4>(text, n, pattern, m, a4, budget, carried, r.resume)
                        : wide<false, 4>(text, n, pattern, m, a4, budget, carried, r.resume);
        }
    }
    if (r.state == 0) return {r.found, r.index};
    return twoway_simd::search_from(text, n, pattern, m, r.resume);
}

// Inlined into the entry point so that a one-byte search is the memchr loop
// and a switch, with no call in between.
template <bool Guarded = true>
[[gnu::always_inline]] static inline std::pair<bool, size_t>
search(const char* text, size_t n, const char* pattern, size_t m) {
    switch (m) {
        case 0: return {true, 0};
        case 1: return short_search<1>(text, n, pattern);
        case 2: return short_search<2>(text, n, pattern);
        case 3: return short_search<3>(text, n, pattern);
        default: return search_long<Guarded>(text, n, pattern, m);
    }
}

}  // namespace needle_hammer

// Entry points in the style of the other kernels.
std::pair<bool, size_t> avx512_needle_hammer(const char* text, size_t n, const char* pattern, size_t m) {
    return needle_hammer::search<true>(text, n, pattern, m);
}
// The same kernel with the guard compiled out, to measure what the guard costs.
std::pair<bool, size_t> avx512_needle_hammer_unguarded(const char* text, size_t n, const char* pattern, size_t m) {
    return needle_hammer::search<false>(text, n, pattern, m);
}
