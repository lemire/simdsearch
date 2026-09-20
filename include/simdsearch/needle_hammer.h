#pragma once
// Needle-Hammer: one wide anchored kernel with a work-counting guard.
//
// The filter tests every candidate position on two to four needle bytes
// chosen for selectivity (first and last byte, then the middle and a quarter
// point, or a byte that is rare in the needle), 256 positions per iteration,
// then narrows the survivors byte by byte. Two anchors are used while the
// haystack shows they suffice; a third and a fourth are added, one at a
// time, when survivors become frequent. Narrowing rounds are counted in the
// block loop and in the windows that cover the ends of the haystack, and
// once they exceed a budget proportional to the haystack the search resumes
// with a linear-time two-way, so every input is searched in linear time.
// Needles of one to three bytes take a dedicated stride loop with no
// verification at all.
//
// Dispatch, in order:
//   m == 0                     match at 0
//   m <= 3                     short_search: M compares per chunk, a survivor is a match
//   n < m                      no match
//   n < 1024 or n < m + 255    masked 64-byte windows with the anchors (small haystack)
//   m == 4                     the four anchors are bytes 0..3: avx512_naive_search256_body
//   m <= 36                    wide kernel, no guard (narrowing bounded by construction)
//   otherwise                  wide kernel, guard budget n/64 rounds; a two-anchor
//                              start escalates to three and four, then resumes with two-way
//
// Two backends: AVX-512 (F/BW), where a window is 64 positions and a block
// 256, and AArch64 NEON, where a window is 16 positions and a block 64. The
// NEON kernel keeps candidates as 0x00/0xFF lanes, tests "any lane alive"
// with shrn + fcmp, and covers the ends of the haystack with overlap windows
// instead of masked loads, as neonsearch.h does. Block-derived constants are
// scaled to the narrower window. What does not depend on the
// register width -- the anchor structure and its positional choice, the
// kernel result, the escalation driver and the public entry points -- is
// written once, above and below the two backends.
//
// Public API (any backend): needle_hammer::find(text, n, pattern, m) returns
// {found, index}; needle_hammer::find(haystack, needle) on string_views
// returns the index or std::string_view::npos. See the README.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include "twoway_simd.h"

namespace needle_hammer {

// ---------------------------------------------------------------------------
// Shared by both backends
// ---------------------------------------------------------------------------

struct anchors {
    size_t o[4];   // offsets into the needle, ascending; unused slots are spares
    int k;         // anchors in use: 2, 3 or 4
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

// Sort the first k anchor offsets (k <= 4). An insertion sort, not
// std::sort: four elements do not need introsort, and GCC's -Warray-bounds
// cannot see that introsort's 16-element threshold is unreachable here.
static inline void sort_anchors(size_t* o, int k) {
    for (int i = 1; i < k; ++i) {
        const size_t v = o[i];
        int j = i;
        while (j > 0 && o[j - 1] > v) { o[j] = o[j - 1]; --j; }
        o[j] = v;
    }
}

// One more anchor: the next spare joins the active set, kept sorted.
static inline anchors escalate(anchors a) {
    ++a.k;
    sort_anchors(a.o, a.k);
    return a;
}

// What a kernel pass reports back to the driver.
struct result {
    bool found = false;
    size_t index = 0;
    int state = 0;         // 0 done; 1 budget exhausted; 2 escalate to one more anchor
    size_t resume = 0;     // first position not yet ruled out, for states 1 and 2
    size_t rounds = 0;     // narrowing rounds counted so far (Guarded only)
    size_t survivors = 0;  // positions that passed the anchors so far (K < 4 only)
};

// Cost of a surviving lane, in filter-compare units, for the escalation rule:
// a mispredicted branch plus, for long needles, the restart of the far load
// streams the anchors at m/2 and m-1 keep in flight.
static inline size_t survivor_cost(size_t m) { return 20 + m / 32; }

// The escalation driver. `run(anchors, carried_rounds, from)` is one pass of
// a kernel with the given anchors from position `from`. Survivors too
// frequent (state 2): add the next anchor and continue from the block that
// showed it, carrying the rounds already spent. The budget spent with fewer
// than four anchors (state 1): continue with all four and a fresh budget, so
// the bound is 2(n/kBudgetDen + 1) rounds plus the two-way pass. Out of
// budget with four anchors: resume with the linear-time two-way from the
// first position not ruled out.
template <class Run>
static inline std::pair<bool, size_t>
drive(Run run, anchors an, const char* text, size_t n, const char* pattern, size_t m) {
    result r = run(an, 0, 0);
    while (r.state != 0 && an.k < 4) {
        if (r.state == 1) { an.k = 4; sort_anchors(an.o, 4); r = run(an, 0, r.resume); }
        else              { an = escalate(an); r = run(an, r.rounds, r.resume); }
    }
    if (r.state == 0) return {r.found, r.index};
    return twoway_simd::search_from(text, n, pattern, m, r.resume);
}

}  // namespace needle_hammer

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#include "avx512_naive.h"

namespace needle_hammer {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Guard budget: narrowing rounds allowed before giving up, as a fraction of
// the haystack. One round is one broadcast plus four masked compares over a
// 256-byte block, about five cycles; the fallback two-way runs at 2-40 GB/s
// depending on the input, so n/64 rounds (~0.1 cycle per haystack byte) keeps
// a search that gives up within a small factor of the fallback alone.
#ifndef NH2_BUDGET_DEN
#define NH2_BUDGET_DEN 64
#endif
static constexpr size_t kBudgetDen = NH2_BUDGET_DEN;

// Below this needle length the guard is absent. A block admits at most m - 2
// narrowing rounds, so the block loop admits at most n(m - 2)/256 rounds,
// about n/7.5 for m = 36; the masked windows (alignment head, remainder,
// small haystacks) admit up to m rounds per 64 positions, n(m - 2)/64 in the
// worst case, still bounded by construction. Both are cheaper than what
// two-way costs on the short periodic needles that would trip a guard
// (8 GB/s of narrowing against 2 GB/s of two-way on the block shape at L=16).
static constexpr size_t kFreeBelow = 36;

// Haystacks shorter than this take masked 64-byte windows rather than the
// 256-byte stride, whose alignment head and loop bound need room to pay off.
static constexpr size_t kMinWide = 1024;

// Needles longer than this start with four anchors whatever kStartAnchors
// says. Measured on Emerald Rapids and Zen 5, a start with fewer anchors wins
// or ties at every length once survivors escalate it when needed, so there
// is no cap by default. Overridable at compile time
// (-DNH2_THREE_ANCHOR_MAX=128, say) so the switch point can be re-fitted per
// machine rather than trusted.
#ifndef NH2_THREE_ANCHOR_MAX
#define NH2_THREE_ANCHOR_MAX SIZE_MAX
#endif
static constexpr size_t kThreeAnchorMax = NH2_THREE_ANCHOR_MAX;
// Anchors a wide-alphabet needle starts with (2, 3 or 4). Default 2: a third
// compare per block costs more than the survivors it removes except on DNA,
// where the escalation adds it within the first blocks. A four-byte needle
// takes the naive kernel instead.
#ifndef NH2_START_ANCHORS
#define NH2_START_ANCHORS 2
#endif
static constexpr int kStartAnchors = NH2_START_ANCHORS;

// ---------------------------------------------------------------------------
// Anchor selection
// ---------------------------------------------------------------------------

// Select the anchors for a needle of m >= 5 bytes.
//
// After the positional choice, one pass over the needle (four vector compares
// per 64 bytes) counts the bytes whose value is NOT one of the four positional
// values. If those are few (at most max(1, m/8)) but not none, the first of
// them is a byte rare in the needle -- and, the needle being a substring of
// the text it matches, rare at that offset in any window that could match --
// so it replaces the quarter point and all four anchors are used. That
// catches a single odd byte wherever it sits. Otherwise the kernel starts
// with kStartAnchors (default 2: first and last); the middle and the quarter
// point are the spares it adds, in that order, if survivors turn out to be
// frequent. DNA, a unary run and a two-letter needle take that same start:
// the first blocks escalate when the two anchors do not cut.
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
    const int start = kStartAnchors;
    if (outside != 0 && outside <= limit) {
        a.o[1] = first_out;                   // a rare byte: the best anchor there is
        a.k = 4;
    } else if (m > kThreeAnchorMax) {
        a.k = 4;
    } else {
        // start anchors, then the spares in the order the escalation adds
        // them: first, last, middle, quarter (first, middle, last when
        // starting with three).
        const size_t quarter = a.o[1], middle = a.o[2];
        a.o[1] = a.o[3]; a.o[2] = middle; a.o[3] = quarter;
        a.k = start;
        if (a.k == 3) std::swap(a.o[1], a.o[2]);
        if (a.k == 4) std::swap(a.o[1], a.o[2]);
    }
    sort_anchors(a.o, a.k);
    return a;
}

// ---------------------------------------------------------------------------
// Masked windows: 64 candidate positions per step, every load masked to the
// live lanes, the anchors tested first and the remaining needle bytes in
// order. Used for the alignment head, the remainder past the stride loop,
// and whole small haystacks. Candidate p is live only while p + m <= n, so
// every load text + p + k with k < m stays inside the buffer.
// ---------------------------------------------------------------------------

// Masked windows over [first, last_exclusive). With fewer than four anchors
// the survivors of the anchor filter are counted and the escalation rule
// applies here as in the block loop, with `allowance` standing for the
// blocks scanned so far: a window whose anchors are common in the haystack
// would otherwise narrow through the whole needle before the block loop
// has seen a single block.
template <bool Guarded, int K>
static inline __attribute__((always_inline)) result
masked_windows(const char* text, [[maybe_unused]] size_t n, const char* pattern, size_t m,
               size_t first, size_t last_exclusive, const size_t* o, const __m512i* pv,
               size_t budget_rounds, size_t rounds, size_t survivors, size_t allowance) {
    for (size_t i = first; i < last_exclusive; i += 64) {
        const size_t cand = last_exclusive - i;
        const __mmask64 active = (cand >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << cand) - 1);
        // Every load is masked by `active`, not by the live mask f: a load
        // whose mask depends on the previous compare makes each round wait
        // for the one before (load latency plus compare latency, about eight
        // cycles), and on an adversarial input a window narrows through the
        // whole needle. With the constant mask the loads and compares are
        // independent and only the AND chains, at a cycle a round.
        __mmask64 f = active;
        for (int k = 0; k < K && f != 0; ++k)
            f &= _mm512_mask_cmpeq_epi8_mask(active, _mm512_maskz_loadu_epi8(active, text + i + o[k]), pv[k]);
        if constexpr (K < 4) {
            if (f != 0) {
                survivors += (size_t)__builtin_popcountll(f);
                if (survivors * survivor_cost(m) > allowance) return {false, 0, 2, i, rounds, survivors};
            }
        }
        // The remaining needle bytes in order, as the segments between the
        // sorted anchor offsets, so no round tests whether it sits on an
        // anchor. The rounds count against the same budget as the block
        // loop's: a window is at most 64 positions, but on an adversarial
        // input it narrows through the whole needle, m rounds.
        size_t k = 0;
        for (int q = 0; q <= K && f != 0; ++q) {
            const size_t end = (q < K) ? o[q] : m;
            for (; k < end && f != 0; ++k) {
                f &= _mm512_mask_cmpeq_epi8_mask(active, _mm512_maskz_loadu_epi8(active, text + i + k),
                                                 _mm512_set1_epi8((char)pattern[k]));
                if constexpr (Guarded) {
                    if (++rounds > budget_rounds) return {false, 0, 1, i, rounds, survivors};
                }
            }
            k = end + 1;
        }
        if (f != 0) return {true, i + (size_t)__builtin_ctzll(f), 0, 0, rounds, survivors};
    }
    return {false, 0, 0, 0, rounds, survivors};
}

// ---------------------------------------------------------------------------
// The wide kernel
// ---------------------------------------------------------------------------
// 256 positions per iteration as four 64-byte chunks; each chunk loads the
// haystack at the chunk base plus each anchor offset and ANDs the compares.
// Loads reach text + i + 192 + 63 + o[K-1] <= text + i + 255 + m - 1, inside
// the buffer by the loop bound. A lone survivor in the block is verified in
// place; several are narrowed together, one needle byte per round, the rounds
// counted when Guarded. With K < 4 the block's survivors are also counted
// and the kernel returns state 2 when they are costing more than another
// anchor would (the same rule for K == 4 would only ever say "keep four").
// Out of line so the broadcasts stay in registers; inlined into the
// dispatcher, GCC reloads them from the stack each block.
template <bool Guarded, int K>
[[gnu::noinline]] static result
wide(const char* text, size_t n, const char* pattern, size_t m,
     const anchors& a, size_t budget_rounds, size_t rounds, size_t start) {
    static_assert(K >= 2 && K <= 4);
    const bool fits = (m <= 64);
    const __mmask64 nmask = fits ? ((m == 64) ? ~(__mmask64)0 : (((__mmask64)1 << m) - 1)) : (__mmask64)0;
    const __m512i nvec = fits ? _mm512_maskz_loadu_epi8(nmask, pattern) : _mm512_setzero_si512();
    const size_t o0 = a.o[0], o1 = a.o[1], o2 = a.o[K > 2 ? 2 : 1], o3 = a.o[K - 1];
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
    size_t survivors = 0;
    if (start == 0) {
        // Walk to a 64-byte boundary so the offset-0 load of every chunk is
        // aligned (a quarter of the loads; the others split lines whatever
        // the base). One masked window covers the positions skipped.
        const size_t head = avx512_align_head(text, n, m);
        if (head) {
            const result r = masked_windows<Guarded, K>(text, n, pattern, m, 0, head, oo, pv, budget_rounds, rounds, 0, 128);
            if (r.found || r.state != 0) return r;
            rounds = r.rounds;
            survivors = r.survivors;
            i = head;
        }
    }
    // Survivors seen so far with fewer than four anchors. The number of
    // blocks scanned is not counted: it is (i - first_block) / 256, and a
    // counter incremented every block was spilled to the stack by the
    // compiler and read-modify-written per iteration, which Zen 5 charged
    // 25% of the loop for.
    const size_t first_block = i;
#define NH2_CHUNK(OFF)                                                                            \
    (K == 4                                                                                       \
     ? ((_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t0+i+(OFF))), p0)                \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t1+i+(OFF))), p1))               \
       & (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t2+i+(OFF))), p2)               \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t3+i+(OFF))), p3)))              \
     : K == 3                                                                                     \
     ? ((_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t0+i+(OFF))), p0)                \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t1+i+(OFF))), p1))               \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t2+i+(OFF))), p2))               \
     : (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t0+i+(OFF))), p0)                 \
       & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(t1+i+(OFF))), p1)))
    const size_t last_block = n - m - 255;   // n >= m + 255 here
    for (; i <= last_block; i += 256) {
        __mmask64 fA = NH2_CHUNK(0), fB = NH2_CHUNK(64);
        __mmask64 fC = NH2_CHUNK(128), fD = NH2_CHUNK(192);
        if ((fA | fB | fC | fD) == 0) continue;
        if constexpr (K < 4) {
            survivors += (size_t)__builtin_popcountll(fA) + (size_t)__builtin_popcountll(fB)
                       + (size_t)__builtin_popcountll(fC) + (size_t)__builtin_popcountll(fD);
            // Each anchor dropped saves about two cycles per block; escalate
            // once the survivors the next one would remove cost more, after
            // a short warm-up. blocks = (i - first_block) / 256.
            if (survivors * survivor_cost(m) > ((i - first_block) >> 7) + 128) return {false, 0, 2, i, rounds, survivors};
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
                                               _mm512_loadu_si512((const void*)pattern)) == 0) {
                // The first 64 bytes match: compare the rest 64 bytes at a
                // time and charge the bytes compared to the budget, at one
                // round per 256 (a round is four 64-byte compares). Without
                // the charge, one late-failing near-match per block -- the
                // needle's period placed once per block -- costs m bytes of
                // comparison per block that no counter sees. On ordinary
                // text a lone survivor that matches 64 bytes is the match.
                const size_t k = 64 + twoway_simd::lcp((const uint8_t*)text + i + b + 64,
                                                       (const uint8_t*)pattern + 64, m - 64);
                if (k == m) return {true, i + b, 0, 0, rounds};
                if constexpr (Guarded) {
                    rounds += (k + 255) >> 8;
                    if (rounds > budget_rounds) return {false, 0, 1, i + 256, rounds};
                }
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
            // The budget is tested per round: on ordinary text this loop
            // runs for a small fraction of the blocks, so the test costs
            // nothing there, and on an adversarial input a single block can
            // run m rounds, many times the budget. A give-up resumes at the
            // block's start, since its positions are not all ruled out.
            if constexpr (Guarded) {
                if (++rounds > budget_rounds) return {false, 0, 1, i, rounds};
            }
        }
        if (fA) return {true, i +   0 + (size_t)__builtin_ctzll(fA), 0, 0, rounds};
        if (fB) return {true, i +  64 + (size_t)__builtin_ctzll(fB), 0, 0, rounds};
        if (fC) return {true, i + 128 + (size_t)__builtin_ctzll(fC), 0, 0, rounds};
        if (fD) return {true, i + 192 + (size_t)__builtin_ctzll(fD), 0, 0, rounds};
    }
#undef NH2_CHUNK
    return masked_windows<Guarded, K>(text, n, pattern, m, i, n - m + 1, oo, pv, budget_rounds, rounds,
                                      survivors, ((i - first_block) >> 7) + 128);
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
        // A macro, not a lambda: GCC outlined the lambda for M == 3, and a
        // call per chunk with the broadcasts passed through memory cost the
        // three-byte needle 3x.
#define NH_SHORT_CHUNK(OFF)                                                                    \
        (M > 2                                                                                 \
         ? (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF))), p0)        \
            & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+1)), p1)    \
            & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+2)), p2))   \
         : (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF))), p0)        \
            & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+1)), p1)))
        if (n >= m + 255) {
            // Most short-needle searches on text end within the first 64
            // bytes: look there first with one unaligned window, then align.
            {
                const __mmask64 f = NH_SHORT_CHUNK(0);
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
                const __mmask64 fA = NH_SHORT_CHUNK(0), fB = NH_SHORT_CHUNK(64);
                const __mmask64 fC = NH_SHORT_CHUNK(128), fD = NH_SHORT_CHUNK(192);
                if ((fA | fB | fC | fD) == 0) continue;
                if (fA) return {true, i +   0 + (size_t)__builtin_ctzll(fA)};
                if (fB) return {true, i +  64 + (size_t)__builtin_ctzll(fB)};
                if (fC) return {true, i + 128 + (size_t)__builtin_ctzll(fC)};
                return {true, i + 192 + (size_t)__builtin_ctzll(fD)};
            }
        }
#undef NH_SHORT_CHUNK
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
    const bool small = n < kMinWide || n < m + 255;
    if (m == 4) return small ? avx512_naive_search_body(text, n, pattern, m)
                             : avx512_naive_search256_body(text, n, pattern, m);
    const anchors a = select(pattern, m);
    const size_t budget = Guarded ? n / kBudgetDen + 1 : ~(size_t)0;
    const bool guarded = Guarded && m > kFreeBelow;
    if (small) {
        return drive([&](const anchors& an, size_t carried, size_t from) -> result {
            __m512i pv[4];
            for (int k = 0; k < an.k; ++k) pv[k] = _mm512_set1_epi8((char)pattern[an.o[k]]);
            switch (an.k) {
                case 2: return guarded ? masked_windows<true, 2>(text, n, pattern, m, from, n - m + 1, an.o, pv, budget, carried, 0, 128)
                                       : masked_windows<false, 2>(text, n, pattern, m, from, n - m + 1, an.o, pv, budget, carried, 0, 128);
                case 3: return guarded ? masked_windows<true, 3>(text, n, pattern, m, from, n - m + 1, an.o, pv, budget, carried, 0, 128)
                                       : masked_windows<false, 3>(text, n, pattern, m, from, n - m + 1, an.o, pv, budget, carried, 0, 128);
                default: return guarded ? masked_windows<true, 4>(text, n, pattern, m, from, n - m + 1, an.o, pv, budget, carried, 0, 128)
                                        : masked_windows<false, 4>(text, n, pattern, m, from, n - m + 1, an.o, pv, budget, carried, 0, 128);
            }
        }, a, text, n, pattern, m);
    }
    return drive([&](const anchors& an, size_t carried, size_t from) -> result {
        switch (an.k) {
            case 2: return guarded ? wide<true, 2>(text, n, pattern, m, an, budget, carried, from)
                                   : wide<false, 2>(text, n, pattern, m, an, budget, carried, from);
            case 3: return guarded ? wide<true, 3>(text, n, pattern, m, an, budget, carried, from)
                                   : wide<false, 3>(text, n, pattern, m, an, budget, carried, from);
            default: return guarded ? wide<true, 4>(text, n, pattern, m, an, budget, carried, from)
                                    : wide<false, 4>(text, n, pattern, m, an, budget, carried, from);
        }
    }, a, text, n, pattern, m);
}

}  // namespace needle_hammer

#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

namespace needle_hammer {

// ---------------------------------------------------------------------------
// Constants (NEON: window W = 16, block B = 64)
// ---------------------------------------------------------------------------
static constexpr size_t kW = 16;
static constexpr size_t kB = 64;
// A narrowing round covers 64 positions here, a quarter of the AVX-512 block,
// so the same instruction budget is four times as many rounds: n/16.
#ifndef NH2_BUDGET_DEN
#define NH2_BUDGET_DEN 16
#endif
static constexpr size_t kBudgetDen = NH2_BUDGET_DEN;
// Below this needle length the guard is absent: a block admits at most m - 2
// rounds, so the block loop admits at most n(m - 2)/64, under n/10 for m <= 8,
// and the single windows at most n(m - 2)/16; bounded by construction.
static constexpr size_t kFreeBelow = 8;
// Small haystacks take single windows rather than the block loop.
static constexpr size_t kMinWide = 512;
#ifndef NH2_THREE_ANCHOR_MAX
#define NH2_THREE_ANCHOR_MAX SIZE_MAX
#endif
static constexpr size_t kThreeAnchorMax = NH2_THREE_ANCHOR_MAX;
// Anchors a wide-alphabet needle starts with, as on AVX-512. At 128-bit
// width the filter's own compares are the cost -- a window is 16 positions,
// so each anchor is a load, a compare and an AND per 16 positions -- and two
// anchors (first and last, walked apart) win on text by the same margin
// three lose by; the escalation rule adds the third and the fourth when
// survivors show they are needed, one at a time. Measured on an Apple M4 Max
// against the memchr crate's two-byte NEON filter. Unlike AVX-512, a
// four-byte needle starts with two anchors here too: there is no separate
// four-byte kernel, and four anchors on a 16-lane window cost more than the
// verification they save.
#ifndef NH2_START_ANCHORS
#define NH2_START_ANCHORS 2
#endif
static constexpr int kStartAnchors = NH2_START_ANCHORS;

static inline uint64_t lane_mask(uint8x16_t v) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0) & 0x8888888888888888ull;
}
static inline bool any_lane(uint8x16_t v) {
    return vget_lane_f64(vreinterpret_f64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0) != 0.0;
}
static inline uint8x16_t load(const char* p) { return vld1q_u8((const uint8_t*)p); }
static inline size_t lane_index(uint64_t mask) { return (size_t)__builtin_ctzll(mask) >> 2; }

// ---------------------------------------------------------------------------
// Anchor selection. Same rarity pass as AVX-512 (sixteen bytes at a time,
// last partial block through a padded copy), but a needle made only of the
// positional values starts with four anchors when m > 8 (three when those
// values are exactly two) rather than escalating from two. At 16 lanes the
// extra compares are cheap relative to verifying DNA-like survivors. A wide
// alphabet still starts with kStartAnchors (default 2).
// ---------------------------------------------------------------------------
static inline anchors select(const char* pattern, size_t m) {
    const unsigned char* s = (const unsigned char*)pattern;
    anchors a;
    positional(s, m, a.o);
    const uint8x16_t v0 = vdupq_n_u8(s[a.o[0]]), v1 = vdupq_n_u8(s[a.o[1]]);
    const uint8x16_t v2 = vdupq_n_u8(s[a.o[2]]), v3 = vdupq_n_u8(s[a.o[3]]);
    const size_t limit = m / 8 > 1 ? m / 8 : 1;
    size_t outside = 0, first_out = m;
    for (size_t k = 0; k < m; k += 16) {
        uint8x16_t x;
        if (k + 16 <= m) {
            x = load(pattern + k);
        } else {
            alignas(16) unsigned char buf[16];
            std::memset(buf, s[a.o[0]], 16);     // padding counts as "inside"
            std::memcpy(buf, s + k, m - k);
            x = vld1q_u8(buf);
        }
        const uint8x16_t in = vorrq_u8(vorrq_u8(vceqq_u8(x, v0), vceqq_u8(x, v1)),
                                       vorrq_u8(vceqq_u8(x, v2), vceqq_u8(x, v3)));
        const uint64_t out = lane_mask(vmvnq_u8(in));
        if (out) {
            if (first_out == m) first_out = k + lane_index(out);
            outside += (size_t)__builtin_popcountll(out);
            if (outside > limit) break;
        }
    }
    const unsigned char b0 = s[a.o[0]], b1 = s[a.o[1]], b2 = s[a.o[2]], b3 = s[a.o[3]];
    const int distinct = 1 + (b1 != b0) + (b2 != b0 && b2 != b1) + (b3 != b0 && b3 != b1 && b3 != b2);
    if (outside == 0 && distinct != 2 && m > 8) {
        a.k = 4;
    } else if (outside != 0 && outside <= limit) {
        a.o[1] = first_out; a.k = 4;
    } else if (m > kThreeAnchorMax) {
        a.k = 4;
    } else if (outside == 0 && m > 8) {
        // two values: first, middle, last carry both; the quarter is the spare
        const size_t quarter = a.o[1];
        a.o[1] = a.o[2]; a.o[2] = a.o[3]; a.o[3] = quarter;
        a.k = 3;
    } else {
        // wide alphabet: start with kStartAnchors; the spares follow in the
        // order the escalation adds them -- first, last, then middle, then
        // the quarter point.
        const size_t quarter = a.o[1], middle = a.o[2];
        a.o[1] = a.o[3]; a.o[2] = middle; a.o[3] = quarter;
        a.k = kStartAnchors;
        if (a.k == 3) std::swap(a.o[1], a.o[2]);   // first, middle, last; spare quarter
    }
    sort_anchors(a.o, a.k);
    return a;
}

// ---------------------------------------------------------------------------
// Single windows with the anchors, for the remainder past the block loop and
// for whole small haystacks. NEON has no masked load, so the last window is
// an overlap window: it starts at the last in-bounds position and lanes below
// `first` -- already scanned, known not to match -- are ignored.
// ---------------------------------------------------------------------------
// Windows over [first, n - m]. With fewer than four anchors the survivors
// of the anchor filter are counted and the escalation rule applies here as
// in the block loop, `allowance` standing for the blocks scanned so far.
template <bool Guarded, int K>
static inline result
windows(const char* text, size_t n, const char* pattern, size_t m,
        size_t first, const size_t* o, const uint8x16_t* pv, size_t budget_rounds, size_t rounds,
        size_t survivors, size_t allowance) {
    if (n < m || first > n - m) return {false, 0, 0, 0, rounds, survivors};
    const size_t positions = n - m + 1;
    if (positions < kW) {
        // fewer than a window of candidates: compare each in place
        for (size_t p = first; p < positions; ++p)
            if (std::memcmp(text + p, pattern, m) == 0) return {true, p, 0, 0, rounds, survivors};
        return {false, 0, 0, 0, rounds, survivors};
    }
    const size_t last = positions - kW;   // last window start with every load in bounds
    // Returns the lane mask of the window's survivors, or 0; sets `stop` to
    // 1 when the narrowing rounds, counted against the same budget as the
    // block loop's, ran out, or to 2 when the survivors call for another
    // anchor (the window is then not resolved).
    int stop = 0;
    auto window = [&](size_t i) -> uint64_t {
        uint8x16_t f = vceqq_u8(load(text + i + o[0]), pv[0]);
        for (int k = 1; k < K; ++k) f = vandq_u8(f, vceqq_u8(load(text + i + o[k]), pv[k]));
        if (!any_lane(f)) return 0;
        if constexpr (K < 4) {
            survivors += (size_t)__builtin_popcountll(lane_mask(f));
            if (survivors * survivor_cost(m) > allowance) { stop = 2; return 0; }
        }
        // the remaining needle bytes in order, as the segments between the
        // sorted anchor offsets
        size_t k = 0;
        for (int q = 0; q <= K; ++q) {
            const size_t end = (q < K) ? o[q] : m;
            for (; k < end; ++k) {
                f = vandq_u8(f, vceqq_u8(load(text + i + k), vdupq_n_u8((uint8_t)pattern[k])));
                if (!any_lane(f)) return 0;
                if constexpr (Guarded) {
                    if (++rounds > budget_rounds) { stop = 1; return 0; }
                }
            }
            k = end + 1;
        }
        return lane_mask(f);
    };
    size_t i = first;
    for (; i <= last; i += kW) {
        const uint64_t r = window(i);
        if (r) return {true, i + lane_index(r), 0, 0, rounds, survivors};
        if (stop) return {false, 0, stop, i, rounds, survivors};
    }
    if (i < positions) {
        uint64_t r = window(last);
        if (stop) return {false, 0, stop, i, rounds, survivors};
        while (r) {
            const size_t b = last + lane_index(r);
            if (b >= i) return {true, b, 0, 0, rounds, survivors};
            r &= r - 1;
        }
    }
    return {false, 0, 0, 0, rounds, survivors};
}

// ---------------------------------------------------------------------------
// The wide kernel: 64 positions per iteration as four 16-lane windows.
// ---------------------------------------------------------------------------

template <bool Guarded, int K>
[[gnu::noinline]] static result
wide(const char* text, size_t n, const char* pattern, size_t m,
     const anchors& a, size_t budget_rounds, size_t rounds, size_t start) {
    static_assert(K >= 2 && K <= 4);
    const bool fits = (m <= kW);
    const size_t o0 = a.o[0], o1 = a.o[1], o2 = a.o[K > 2 ? 2 : 1], o3 = a.o[K - 1];
    const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[o0]), p1 = vdupq_n_u8((uint8_t)pattern[o1]);
    const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[o2]), p3 = vdupq_n_u8((uint8_t)pattern[o3]);
    const size_t oo[4] = {o0, o1, o2, o3};
    const uint8x16_t pv[4] = {p0, p1, p2, p3};
    const char* const t0 = text + o0; const char* const t1 = text + o1;
    const char* const t2 = text + o2; const char* const t3 = text + o3;
    // needle staged once for m <= 16
    alignas(16) uint8_t nbuf[16] = {};
    if (fits) std::memcpy(nbuf, pattern, m);
    const uint8x16_t nvec = vld1q_u8(nbuf);
    const uint64_t nbits = 0x8888888888888888ull & ((m >= 16) ? ~(uint64_t)0 : (((uint64_t)1 << (4 * m)) - 1));
    // For a needle past one register the comparison runs 16 bytes at a time
    // and the bytes compared are charged to the budget, one round per 64 (a
    // round is four 16-byte compares): see the AVX-512 kernel.
    auto verify = [&](size_t pos) -> bool {
        if (fits && pos + kW <= n) return (lane_mask(vceqq_u8(load(text + pos), nvec)) & nbits) == nbits;
        const size_t k = twoway_simd::lcp((const uint8_t*)text + pos, (const uint8_t*)pattern, m);
        if constexpr (Guarded) rounds += (k + 63) >> 6;
        return k == m;
    };
    size_t i = start;
    const size_t first_block = i;
    size_t survivors = 0;
#define NH2N_CHUNK(OFF)                                                                       \
    (K == 4 ? vandq_u8(vandq_u8(vceqq_u8(load(t0 + i + (OFF)), p0), vceqq_u8(load(t1 + i + (OFF)), p1)),  \
                       vandq_u8(vceqq_u8(load(t2 + i + (OFF)), p2), vceqq_u8(load(t3 + i + (OFF)), p3)))  \
     : K == 3 ? vandq_u8(vandq_u8(vceqq_u8(load(t0 + i + (OFF)), p0), vceqq_u8(load(t1 + i + (OFF)), p1)),  \
                       vceqq_u8(load(t2 + i + (OFF)), p2))                                          \
              : vandq_u8(vceqq_u8(load(t0 + i + (OFF)), p0), vceqq_u8(load(t1 + i + (OFF)), p1)))
    for (; i + m + kB - 1 <= n; i += kB) {
        uint8x16_t fA = NH2N_CHUNK(0),  fB = NH2N_CHUNK(16);
        uint8x16_t fC = NH2N_CHUNK(32), fD = NH2N_CHUNK(48);
        uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
        if (!any_lane(any)) continue;
        const uint64_t mA = lane_mask(fA), mB = lane_mask(fB), mC = lane_mask(fC), mD = lane_mask(fD);
        if constexpr (K < 4) {
            survivors += (size_t)__builtin_popcountll(mA) + (size_t)__builtin_popcountll(mB)
                       + (size_t)__builtin_popcountll(mC) + (size_t)__builtin_popcountll(mD);
            if (survivors * survivor_cost(m) > ((i - first_block) >> 5) + 128) return {false, 0, 2, i, rounds, survivors};
        }
        const int nz = (mA != 0) + (mB != 0) + (mC != 0) + (mD != 0);
        const uint64_t one = mA | mB | mC | mD;
        if (nz == 1 && (one & (one - 1)) == 0) {
            const size_t off = (mA != 0) ? 0 : (mB != 0) ? 16 : (mC != 0) ? 32 : 48;
            const size_t b = i + off + lane_index(one);
            if (verify(b)) return {true, b, 0, 0, rounds};
            if constexpr (Guarded) { if (rounds > budget_rounds) return {false, 0, 1, i + kB, rounds}; }
            continue;
        }
        for (size_t k = 0; k < m; ++k) {
            if (k == o0 || k == o1 || k == o2 || k == o3) continue;
            any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
            if (!any_lane(any)) break;
            const uint8x16_t pk = vdupq_n_u8((uint8_t)pattern[k]);
            fA = vandq_u8(fA, vceqq_u8(load(text + i + k +  0), pk));
            fB = vandq_u8(fB, vceqq_u8(load(text + i + k + 16), pk));
            fC = vandq_u8(fC, vceqq_u8(load(text + i + k + 32), pk));
            fD = vandq_u8(fD, vceqq_u8(load(text + i + k + 48), pk));
            // Budget tested per round; a give-up resumes at the block's
            // start, since its positions are not all ruled out.
            if constexpr (Guarded) {
                if (++rounds > budget_rounds) return {false, 0, 1, i, rounds};
            }
        }
        uint64_t r;
        if ((r = lane_mask(fA))) return {true, i +  0 + lane_index(r), 0, 0, rounds};
        if ((r = lane_mask(fB))) return {true, i + 16 + lane_index(r), 0, 0, rounds};
        if ((r = lane_mask(fC))) return {true, i + 32 + lane_index(r), 0, 0, rounds};
        if ((r = lane_mask(fD))) return {true, i + 48 + lane_index(r), 0, 0, rounds};
    }
#undef NH2N_CHUNK
    return windows<Guarded, K>(text, n, pattern, m, i, oo, pv, budget_rounds, rounds,
                               survivors, ((i - first_block) >> 5) + 128);
}

// ---------------------------------------------------------------------------
// Short needles, m in {1, 2, 3}: the filter is the whole needle.
// ---------------------------------------------------------------------------
template <int M>
static inline std::pair<bool, size_t>
short_search(const char* text, size_t n, const char* pattern) {
    static_assert(M >= 1 && M <= 3);
    const size_t m = M;
    if (n < m) return {false, 0};
    const size_t positions = n - m + 1;
    const uint8x16_t q0 = vdupq_n_u8((uint8_t)pattern[0]);
    const uint8x16_t q1 = vdupq_n_u8((uint8_t)pattern[M > 1 ? 1 : 0]);
    const uint8x16_t q2 = vdupq_n_u8((uint8_t)pattern[M > 2 ? 2 : 0]);
    if (positions < kW) {
        for (size_t p = 0; p < positions; ++p)
            if (std::memcmp(text + p, pattern, m) == 0) return {true, p};
        return {false, 0};
    }
#define NH2N_SHORT(BASE)                                                                    \
    (M == 1 ? vceqq_u8(load(BASE), q0)                                                      \
     : M == 2 ? vandq_u8(vceqq_u8(load(BASE), q0), vceqq_u8(load((BASE) + 1), q1))          \
              : vandq_u8(vandq_u8(vceqq_u8(load(BASE), q0), vceqq_u8(load((BASE) + 1), q1)), \
                         vceqq_u8(load((BASE) + 2), q2)))
    const size_t last = positions - kW;
    size_t i = 0;
    for (; i + kB - 1 <= last; i += kB) {
        const uint8x16_t fA = NH2N_SHORT(text + i), fB = NH2N_SHORT(text + i + 16);
        const uint8x16_t fC = NH2N_SHORT(text + i + 32), fD = NH2N_SHORT(text + i + 48);
        if (!any_lane(vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD)))) continue;
        uint64_t r;
        if ((r = lane_mask(fA))) return {true, i +  0 + lane_index(r)};
        if ((r = lane_mask(fB))) return {true, i + 16 + lane_index(r)};
        if ((r = lane_mask(fC))) return {true, i + 32 + lane_index(r)};
        r = lane_mask(fD);       return {true, i + 48 + lane_index(r)};
    }
    for (; i <= last; i += kW) {
        const uint8x16_t f = NH2N_SHORT(text + i);
        if (any_lane(f)) return {true, i + lane_index(lane_mask(f))};
    }
    if (i < positions) {
        uint64_t r = lane_mask(NH2N_SHORT(text + last));
        while (r) {
            const size_t b = last + lane_index(r);
            if (b >= i) return {true, b};
            r &= r - 1;
        }
    }
#undef NH2N_SHORT
    return {false, 0};
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------
template <bool Guarded>
[[gnu::noinline]] static std::pair<bool, size_t>
search_long(const char* text, size_t n, const char* pattern, size_t m) {
    if (n < m) return {false, 0};
    const anchors a = select(pattern, m);
    const size_t budget = Guarded ? n / kBudgetDen + 1 : ~(size_t)0;
    const bool guarded = Guarded && m > kFreeBelow;
    if (n < kMinWide || n < m + kB - 1) {
        return drive([&](const anchors& an, size_t carried, size_t from) -> result {
            uint8x16_t pv[4];
            for (int k = 0; k < an.k; ++k) pv[k] = vdupq_n_u8((uint8_t)pattern[an.o[k]]);
            switch (an.k) {
                case 2: return guarded ? windows<true, 2>(text, n, pattern, m, from, an.o, pv, budget, carried, 0, 128)
                                       : windows<false, 2>(text, n, pattern, m, from, an.o, pv, budget, carried, 0, 128);
                case 3: return guarded ? windows<true, 3>(text, n, pattern, m, from, an.o, pv, budget, carried, 0, 128)
                                       : windows<false, 3>(text, n, pattern, m, from, an.o, pv, budget, carried, 0, 128);
                default: return guarded ? windows<true, 4>(text, n, pattern, m, from, an.o, pv, budget, carried, 0, 128)
                                        : windows<false, 4>(text, n, pattern, m, from, an.o, pv, budget, carried, 0, 128);
            }
        }, a, text, n, pattern, m);
    }
    return drive([&](const anchors& an, size_t carried, size_t from) -> result {
        switch (an.k) {
            case 2: return guarded ? wide<true, 2>(text, n, pattern, m, an, budget, carried, from)
                                   : wide<false, 2>(text, n, pattern, m, an, budget, carried, from);
            case 3: return guarded ? wide<true, 3>(text, n, pattern, m, an, budget, carried, from)
                                   : wide<false, 3>(text, n, pattern, m, an, budget, carried, from);
            default: return guarded ? wide<true, 4>(text, n, pattern, m, an, budget, carried, from)
                                    : wide<false, 4>(text, n, pattern, m, an, budget, carried, from);
        }
    }, a, text, n, pattern, m);
}

}  // namespace needle_hammer

#else
#error "needle_hammer.h: no SIMD backend (needs AVX-512F+BW or AArch64 NEON)"
#endif

// ---------------------------------------------------------------------------
// Entry points, the same on both backends
// ---------------------------------------------------------------------------
namespace needle_hammer {

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

// First occurrence of pattern[0..m) in text[0..n): {true, index} or
// {false, 0}. The empty needle matches at 0. Neither buffer needs a NUL
// terminator and both may contain any byte value; no byte outside
// [text, text + n) or [pattern, pattern + m) is ever read.
inline std::pair<bool, size_t> find(const char* text, size_t n, const char* pattern, size_t m) {
    return search<true>(text, n, pattern, m);
}
// The same on string_views: the index, or std::string_view::npos.
inline size_t find(std::string_view haystack, std::string_view needle) {
    const auto [found, index] = search<true>(haystack.data(), haystack.size(), needle.data(), needle.size());
    return found ? index : std::string_view::npos;
}
// The same kernel with the guard compiled out. Faster by a few percent on
// text; no longer linear in the worst case. For measurement, not for use on
// input you do not control.
inline std::pair<bool, size_t> find_unguarded(const char* text, size_t n, const char* pattern, size_t m) {
    return search<false>(text, n, pattern, m);
}

}  // namespace needle_hammer

// Entry points in the style of the other kernels, for the benchmark table.
#if defined(__AVX512F__) && defined(__AVX512BW__)
inline std::pair<bool, size_t> avx512_needle_hammer(const char* text, size_t n, const char* pattern, size_t m) {
    return needle_hammer::find(text, n, pattern, m);
}
inline std::pair<bool, size_t> avx512_needle_hammer_unguarded(const char* text, size_t n, const char* pattern, size_t m) {
    return needle_hammer::find_unguarded(text, n, pattern, m);
}
#else
inline std::pair<bool, size_t> neon_needle_hammer(const char* text, size_t n, const char* pattern, size_t m) {
    return needle_hammer::find(text, n, pattern, m);
}
inline std::pair<bool, size_t> neon_needle_hammer_unguarded(const char* text, size_t n, const char* pattern, size_t m) {
    return needle_hammer::find_unguarded(text, n, pattern, m);
}
#endif
