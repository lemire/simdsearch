#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

// The anchor selector plus the scalar and library baselines, shared with the
// NEON backend.
#include "common_search.h"
// The alignment head and the naive four-byte-prefix kernels, shared with
// needle_hammer.h.
#include "avx512_naive.h"


inline std::pair<bool, size_t> avx512_naive_search(const char* text, size_t n,
                                               const char* pattern, size_t m) {
    return avx512_naive_search_body(text, n, pattern, m);
}

inline std::pair<bool, size_t> avx512_naive_search256(const char* text, size_t n,
                                                  const char* pattern, size_t m) {
    return avx512_naive_search256_body(text, n, pattern, m);
}


// Find-all variant of avx512_naive_search. Instead of returning at the first
// match and discarding the rest of the mask, each 64-byte block builds the same
// per-position match mask and then enumerates EVERY set bit -- calling
// callback(index) for each occurrence -- before advancing to the next block.
// This pays the pattern-byte broadcasts and the block scan once per block
// regardless of how many matches it holds, instead of re-entering a first-match
// search (with its own setup) once per match. Occurrences are reported in
// increasing index order and include overlapping ones. callback must be a
// callable taking a single size_t index.
template <typename F>
inline void avx512_naive_search_all(const char* text, size_t n, const char* pattern,
                             size_t m, F callback) {
    // Empty needle: match at every index in [0, n], matching the first-match
    // loop baseline (returns {true, 0} then advances one byte). Unsupported to
    // leave undefined -- callers comparing loop vs block would disagree.
    if (m == 0) {
        for (size_t i = 0; i <= n; ++i) callback(i);
        return;
    }
    if (n < m) return;

    size_t i = 0;
    // SIMD chunk reads bytes [i, i + 63 + (m - 1)], so require i + m + 63 <= n.
    if (n >= m + 63) {
        if (m >= 4) {
            // Broadcasts of the first four pattern bytes are loop-invariant, so
            // hoist them out of the block loop (amortized across all blocks).
            const __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
            const __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
            const __m512i p2 = _mm512_set1_epi8((char)pattern[2]);
            const __m512i p3 = _mm512_set1_epi8((char)pattern[3]);
            for (; i + m + 63 <= n; i += 64) {
                __mmask64 found = _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512((const void*)(text + i)), p0);
                found = _mm512_mask_cmpeq_epi8_mask(
                    found, _mm512_loadu_si512((const void*)(text + i + 1)), p1);
                found = _mm512_mask_cmpeq_epi8_mask(
                    found, _mm512_loadu_si512((const void*)(text + i + 2)), p2);
                found = _mm512_mask_cmpeq_epi8_mask(
                    found, _mm512_loadu_si512((const void*)(text + i + 3)), p3);
                for (size_t j = 4; j < m && found; ++j) {
                    __m512i pj = _mm512_set1_epi8((char)pattern[j]);
                    found = _mm512_mask_cmpeq_epi8_mask(
                        found, _mm512_loadu_si512((const void*)(text + i + j)), pj);
                }
                // Enumerate every match in this block, lowest index first.
                while (found) {
                    callback(i + (size_t)__builtin_ctzll(found));
                    found &= found - 1;  // clear the lowest set bit
                }
            }
        } else {
            const __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
            for (; i + m + 63 <= n; i += 64) {
                __mmask64 found = _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512((const void*)(text + i)), p0);
                for (size_t j = 1; j < m && found; ++j) {
                    __m512i pj = _mm512_set1_epi8((char)pattern[j]);
                    found = _mm512_mask_cmpeq_epi8_mask(
                        found, _mm512_loadu_si512((const void*)(text + i + j)), pj);
                }
                while (found) {
                    callback(i + (size_t)__builtin_ctzll(found));
                    found &= found - 1;
                }
            }
        }
    }

    // Scalar tail for positions the SIMD loop could not safely cover.
    for (; i + m <= n; ++i)
        if (std::memcmp(text + i, pattern, m) == 0) callback(i);
}




// Returns {found, index} of first occurrence (matches avx512_naive_search
// interface). Faithful port of StringZilla's sz_find_skylake. Three needle
// bytes (first/middle/last, chosen by sz_locate_needle_anomalies) are
// broadcast and compared against the haystack; ANDing the three masks leaves
// only candidates that match at all three anchors, which a specialised compare
// then verifies in full (three paths by needle length; see below). Every load is a predicated _mm512_maskz_loadu_epi8, so a
// single masked loop covers the body, the tail, and haystacks shorter than one
// 64-byte window with no scalar fallback (masked-off lanes are never touched,
// so the loads stay in bounds at the end of the haystack).
// Vectorised equality, porting StringZilla's sz_equal_skylake. Used to confirm a
// surviving candidate when the needle is too long to sit in one register.
// Deliberately not std::memcmp: the library call costs about ten cycles of
// call and setup per candidate, and on an input where the filter stops
// discriminating that fixed cost is paid once per haystack position.
static inline bool sz_equal_avx512(const char* a, const char* b, size_t length) {
    while (length >= 64) {
        if (_mm512_cmpneq_epi8_mask(_mm512_loadu_si512((const void*)a),
                                    _mm512_loadu_si512((const void*)b)) != 0)
            return false;
        a += 64; b += 64; length -= 64;
    }
    if (length) {
        const __mmask64 m = (length >= 64) ? ~(__mmask64)0
                                           : (((__mmask64)1 << length) - 1);
        return _mm512_mask_cmpneq_epi8_mask(m, _mm512_maskz_loadu_epi8(m, a),
                                            _mm512_maskz_loadu_epi8(m, b)) == 0;
    }
    return true;
}

// The anchored kernel, once, with a verification budget in bytes. Both entry
// points below call this same function, so they run the same instructions and
// differ only in the budget they pass -- which is what makes the guarded and
// unguarded searchers comparable.
//
// The budget is not a template parameter, and deliberately so. Templating the
// body on whether the counter is compiled in produces two machine-code bodies,
// and GCC generates markedly better code for the guarded one: with budgets set
// so that neither can bail, the guarded instantiation measures substantially
// faster than the unguarded one for identical work. One body removes that
// instead of leaving it to be explained.
//
// Unguarded callers pass SIZE_MAX. The counter cannot reach it: doing so would
// need ~2^64 bytes of verification, which no reachable input supplies.
//
// Verification follows StringZilla's three length-specialised paths rather than
// collapsing to a single std::memcmp call. The collapse is expensive exactly
// where it is least affordable: on an input that defeats the filter, every
// haystack position becomes a candidate, so the per-candidate constant is
// multiplied by n. The three paths are
//
//   m <= 3   no verification at all. The anchors are 0, m/2 and m-1, which for
//            m <= 3 covers every byte of the needle, so a surviving mask bit is
//            already a confirmed match.
//   m < 64   the whole needle is preloaded into one register once, outside the
//            loop; each candidate costs a masked load and a masked compare.
//   m >= 64  sz_equal_avx512, 64 bytes per iteration.
template <bool FilterHighBytes = false>
static inline simd_guarded_result avx512_stringzilla_body(
    const char* haystack, size_t h_len, const char* needle, size_t n_len,
    size_t budget_bytes) {
    if (n_len == 0) return {true, 0, false, 0};
    if (h_len < n_len) return {false, 0, false, 0};

    // Single-byte needle: one masked broadcast compare per 64-byte window. It
    // verifies nothing, so it cannot exhaust a budget and needs no guard.
    if (n_len == 1) {
        __m512i n_vec = _mm512_set1_epi8((char)needle[0]);
        for (size_t i = 0; i < h_len; i += 64) {
            size_t cand = h_len - i;  // bytes (= candidate positions) remaining
            __mmask64 active = (cand >= 64) ? ~(__mmask64)0
                                            : (((__mmask64)1 << cand) - 1);
            __mmask64 eq = _mm512_mask_cmpeq_epi8_mask(
                active, _mm512_maskz_loadu_epi8(active, haystack + i), n_vec);
            if (eq != 0) return {true, i + (size_t)__builtin_ctzll(eq), false, 0};
        }
        return {false, 0, false, 0};
    }

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies_t<FilterHighBytes>(needle, n_len, off_first,
                                                  off_mid, off_last);
    __m512i first = _mm512_set1_epi8((char)needle[off_first]);
    __m512i mid = _mm512_set1_epi8((char)needle[off_mid]);
    __m512i last = _mm512_set1_epi8((char)needle[off_last]);

    // Needles that fit in one register are loaded once, here, not per candidate.
    const bool anchors_cover_needle = (n_len <= 3);
    const bool needle_fits_register = (n_len < 64);
    const __mmask64 needle_mask =
        needle_fits_register ? (((__mmask64)1 << n_len) - 1) : (__mmask64)0;
    const __m512i needle_vec =
        needle_fits_register ? _mm512_maskz_loadu_epi8(needle_mask, needle)
                             : _mm512_setzero_si512();

    // Each iteration handles up to 64 candidate start positions [i, i+63].
    // active masks to the candidates that actually exist; off_last <= n_len-1,
    // so the masked-off lanes are exactly the ones that would read past h_len.
    size_t verified = 0;
    for (size_t i = 0; i + n_len <= h_len; i += 64) {
        size_t cand = h_len - n_len + 1 - i;  // candidate positions remaining
        __mmask64 active = (cand >= 64) ? ~(__mmask64)0
                                        : (((__mmask64)1 << cand) - 1);
        __mmask64 mask = _mm512_mask_cmpeq_epi8_mask(
            active, _mm512_maskz_loadu_epi8(active, haystack + i + off_first), first);
        mask = _mm512_mask_cmpeq_epi8_mask(
            mask, _mm512_maskz_loadu_epi8(mask, haystack + i + off_mid), mid);
        mask = _mm512_mask_cmpeq_epi8_mask(
            mask, _mm512_maskz_loadu_epi8(mask, haystack + i + off_last), last);

        // The anchors already tested every byte; no verification to do.
        if (anchors_cover_needle) {
            if (mask != 0)
                return {true, i + (size_t)__builtin_ctzll(mask), false, 0};
            continue;
        }

        while (mask != 0) {
            size_t b = (size_t)__builtin_ctzll(mask);
            verified += n_len;
            bool equal;
            if (needle_fits_register) {
                equal = _mm512_mask_cmpneq_epi8_mask(
                            needle_mask,
                            _mm512_maskz_loadu_epi8(needle_mask, haystack + i + b),
                            needle_vec) == 0;
            } else {
                equal = sz_equal_avx512(haystack + i + b, needle, n_len);
            }
            if (equal) return {true, i + b, false, 0};
            // Budget tested after the compare, for the same reason as the wide
            // kernel: the candidate is already paid for, so a match is reported
            // rather than thrown away. Overshoot is one verification.
            //
            // Resume at b + 1, which is what `resume` means: the first position
            // not yet ruled out. Everything below it in this window is ruled out
            // too -- the lanes that are not mask bits failed the anchors, and
            // the mask bits below b were verified and failed.
            if (verified > budget_bytes) return {false, 0, true, i + b + 1};
            mask &= mask - 1;  // clear the lowest set bit and continue
        }
    }

    return {false, 0, false, 0};
}

// The kernel as the algorithm table sees it: the shared body with a budget it
// can never exhaust, so this is the original algorithm with no guard behaviour.
inline std::pair<bool, size_t> avx512_stringzilla_find(const char* haystack, size_t h_len,
                                                const char* needle, size_t n_len)
{
    auto r = avx512_stringzilla_body<false>(haystack, h_len, needle, n_len,
                                            ~(size_t)0);  // unreachable budget
    return {r.found, r.index};
}

// The same kernel with upstream's UTF-8 lead-byte rule enabled, so the cost of
// that choice can be measured rather than assumed. Not the default: see
// sz_needle_anomalies.h.
inline std::pair<bool, size_t> avx512_stringzilla_find_hifilter(const char* haystack,
                                                         size_t h_len,
                                                         const char* needle,
                                                         size_t n_len)
{
    auto r = avx512_stringzilla_body<true>(haystack, h_len, needle, n_len,
                                           ~(size_t)0);
    return {r.found, r.index};
}


// A second way to combine the two ideas, for reference: keep StringZilla's
// three-anchor filter but run it with the 256-byte stride of the naive kernel,
// so each anchor broadcast is reused across four 64-byte chunks and the
// per-window mask bookkeeping is paid once per 256 bytes instead of once per
// 64. Unlike the length-dispatched needle-hammer above there is no threshold -- the
// filter cost is already independent of m -- so this is length-insensitive
// everywhere, not just above a cutoff.
//
// The main loop needs i + m + 255 <= n (the last anchor of the last chunk reads
// text[i + 192 + 63 + off_last], and off_last <= m - 1). Whatever it cannot
// cover is handed to avx512_stringzilla_find, whose masked loads already handle
// short haystacks with no scalar fallback.
inline std::pair<bool, size_t> avx512_stringzilla256_find(const char* text, size_t n,
                                                   const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    // One-byte needles have no three distinct anchors; the 64-byte kernel
    // already handles them with a single broadcast compare per window.
    if (m == 1 || n < m + 255) return avx512_stringzilla_find(text, n, pattern, m);

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies(pattern, m, off_first, off_mid, off_last);
    const bool n_fits = (m <= 64);
    const __mmask64 n_mask = n_fits ? ((m == 64) ? ~(__mmask64)0
                                                 : (((__mmask64)1 << m) - 1))
                                    : (__mmask64)0;
    const __m512i n_vec = n_fits ? _mm512_maskz_loadu_epi8(n_mask, pattern)
                                 : _mm512_setzero_si512();
    const __m512i first = _mm512_set1_epi8((char)pattern[off_first]);
    const __m512i mid = _mm512_set1_epi8((char)pattern[off_mid]);
    const __m512i last = _mm512_set1_epi8((char)pattern[off_last]);

    size_t i = 0;
    for (; i + m + 255 <= n; i += 256) {
        const char* a = text + i + off_first;
        __mmask64 fA = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(a +   0)), first);
        __mmask64 fB = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(a +  64)), first);
        __mmask64 fC = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(a + 128)), first);
        __mmask64 fD = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(a + 192)), first);
        if ((fA | fB | fC | fD) == 0) continue;

        const char* b = text + i + off_mid;
        fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(b +   0)), mid);
        fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(b +  64)), mid);
        fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(b + 128)), mid);
        fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(b + 192)), mid);
        if ((fA | fB | fC | fD) == 0) continue;

        const char* c = text + i + off_last;
        fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(c +   0)), last);
        fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(c +  64)), last);
        fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(c + 128)), last);
        fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(c + 192)), last);
        if ((fA | fB | fC | fD) == 0) continue;

        // Verify survivors in increasing index order: chunk by chunk, and
        // within a chunk lowest set bit first, so the first memcmp that
        // succeeds is the leftmost occurrence.
        const __mmask64 masks[4] = {fA, fB, fC, fD};
        for (size_t k = 0; k < 4; ++k) {
            __mmask64 mk = masks[k];
            while (mk != 0) {
                size_t off = i + 64 * k + (size_t)__builtin_ctzll(mk);
                bool eq;
                if (n_fits) {
                    eq = _mm512_mask_cmpneq_epi8_mask(
                             n_mask, _mm512_maskz_loadu_epi8(n_mask, text + off),
                             n_vec) == 0;
                } else {
                    eq = sz_equal_avx512(text + off, pattern, m);
                }
                if (eq) return {true, off};
                mk &= mk - 1;
            }
        }
    }

    // Remainder: hand the uncovered suffix to the 64-byte masked kernel.
    auto [f, idx] = avx512_stringzilla_find(text + i, n - i, pattern, m);
    if (!f) return {false, 0};
    return {true, i + idx};
}


// ===========================================================================
// Needle-hammer at 256-bit (AVX2) and 128-bit (SSE2) register width
// ===========================================================================
//
// The same three kernels and the same length dispatch as the AVX-512 version
// above, rebuilt on narrower vectors, so the scheme can be measured against
// its own register width instead of only against other algorithms. Everything
// below is width-generic: an Ops traits struct supplies the vector type, the
// window width W, and the three primitives (broadcast, unaligned load, compare
// -> bitmap), and one templated body serves both widths.
//
// The one structural difference from AVX-512 is where the match state lives.
// AVX-512 keeps it in a dedicated __mmask64 and folds "compare only where a
// candidate is still alive, then AND" into a single instruction
// (_mm512_mask_cmpeq_epi8_mask). AVX2 and SSE2 have no mask registers, so the
// state is a plain integer bitmap produced by cmpeq + movemask and narrowed
// with &= -- one extra instruction per pattern-byte round, and the compare is
// done for every lane whether or not it is still alive. The algorithm is
// otherwise identical.
//
// The other difference is the tail. AVX-512's anchored kernel uses predicated
// loads, so a single masked loop covers the body, the tail and haystacks
// shorter than one window with no scalar fallback. Neither narrower ISA has
// byte-granular masked loads, so all three kernels here stop the vector loop
// while a full window is in bounds and finish with a scalar memcmp loop -- the
// same shape avx512_naive_search256 already uses.

// 256-bit lanes (AVX2). The bitmap is 32 bits, one per candidate position.
struct x86_ops256 {
    using vec_t = __m256i;
    static constexpr size_t kWidth = 32;
    static inline vec_t broadcast(char c) { return _mm256_set1_epi8(c); }
    static inline vec_t load(const char* p) {
        return _mm256_loadu_si256((const __m256i*)p);
    }
    static inline uint32_t eq_mask(vec_t a, vec_t b) {
        return (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b));
    }
};

// 128-bit lanes (SSE2, baseline on x86-64). The bitmap uses the low 16 bits of
// the same uint32_t; the upper half stays zero, so ctz and the &= narrowing
// need no separate handling.
struct x86_ops128 {
    using vec_t = __m128i;
    static constexpr size_t kWidth = 16;
    static inline vec_t broadcast(char c) { return _mm_set1_epi8(c); }
    static inline vec_t load(const char* p) {
        return _mm_loadu_si128((const __m128i*)p);
    }
    static inline uint32_t eq_mask(vec_t a, vec_t b) {
        return (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(a, b));
    }
};


// Naive prefix-filter search over one W-byte window per step: the width-generic
// form of avx512_naive_search. Pattern byte j is broadcast and compared against
// text[i+j .. i+j+W-1]; the bitmap of surviving candidate positions is narrowed
// byte by byte and the lowest surviving bit is the answer.
//
// The first four bytes are compared unconditionally (no early-out test between
// them) because on ordinary text they are what kills essentially every
// candidate -- the branch would cost more than the compare. Needles shorter
// than four bytes take the generic loop from byte 0 so they stay on the vector
// path rather than dropping to the scalar tail for the whole haystack.
template <class Ops>
static inline std::pair<bool, size_t> x86_naive_search_t(const char* text, size_t n,
                                                         const char* pattern, size_t m) {
    constexpr size_t W = Ops::kWidth;
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    // A window reads bytes [i, i + W - 1 + (m - 1)], so require i + m + W - 1 <= n.
    if (m >= 4) {
        for (; i + m + W - 1 <= n; i += W) {
            uint32_t found = Ops::eq_mask(Ops::load(text + i + 0), Ops::broadcast(pattern[0]));
            found &= Ops::eq_mask(Ops::load(text + i + 1), Ops::broadcast(pattern[1]));
            found &= Ops::eq_mask(Ops::load(text + i + 2), Ops::broadcast(pattern[2]));
            found &= Ops::eq_mask(Ops::load(text + i + 3), Ops::broadcast(pattern[3]));
            for (size_t j = 4; j < m; ++j) {
                if (found == 0) break;
                found &= Ops::eq_mask(Ops::load(text + i + j), Ops::broadcast(pattern[j]));
            }
            if (found == 0) continue;
            return {true, i + (size_t)__builtin_ctz(found)};
        }
    } else {
        for (; i + m + W - 1 <= n; i += W) {
            uint32_t found = Ops::eq_mask(Ops::load(text + i), Ops::broadcast(pattern[0]));
            for (size_t j = 1; j < m; ++j) {
                if (found == 0) break;
                found &= Ops::eq_mask(Ops::load(text + i + j), Ops::broadcast(pattern[j]));
            }
            if (found == 0) continue;
            return {true, i + (size_t)__builtin_ctz(found)};
        }
    }

    // Scalar tail for the positions the vector loop could not safely cover.
    for (; i + m <= n; ++i) {
        if (std::memcmp(text + i, pattern, m) == 0) return {true, i};
    }
    return {false, 0};
}


// The same filter over a block of four W-byte chunks (4W bytes per step): the
// width-generic form of avx512_naive_search256. Each pattern-byte broadcast is
// built once and reused across all four chunks, which is where the win over the
// single-window kernel comes from; the price is that a match cannot be reported
// until the whole block is filtered, so the chunks are scanned in index order
// to keep the leftmost-match guarantee.
template <class Ops>
static inline std::pair<bool, size_t> x86_naive_search_wide_t(const char* text, size_t n,
                                                              const char* pattern, size_t m) {
    constexpr size_t W = Ops::kWidth;
    constexpr size_t B = 4 * W;  // bytes of candidate positions per block
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    // A block reads bytes [i, i + B - 1 + (m - 1)], so require i + m + B - 1 <= n.
    if (m >= 4) {
        for (; i + m + B - 1 <= n; i += B) {
            typename Ops::vec_t p = Ops::broadcast(pattern[0]);
            uint32_t fA = Ops::eq_mask(Ops::load(text + i + 0 * W), p);
            uint32_t fB = Ops::eq_mask(Ops::load(text + i + 1 * W), p);
            uint32_t fC = Ops::eq_mask(Ops::load(text + i + 2 * W), p);
            uint32_t fD = Ops::eq_mask(Ops::load(text + i + 3 * W), p);

            for (size_t j = 1; j < 4; ++j) {
                p = Ops::broadcast(pattern[j]);
                fA &= Ops::eq_mask(Ops::load(text + i + j + 0 * W), p);
                fB &= Ops::eq_mask(Ops::load(text + i + j + 1 * W), p);
                fC &= Ops::eq_mask(Ops::load(text + i + j + 2 * W), p);
                fD &= Ops::eq_mask(Ops::load(text + i + j + 3 * W), p);
            }

            for (size_t j = 4; j < m; ++j) {
                if ((fA | fB | fC | fD) == 0) break;
                p = Ops::broadcast(pattern[j]);
                fA &= Ops::eq_mask(Ops::load(text + i + j + 0 * W), p);
                fB &= Ops::eq_mask(Ops::load(text + i + j + 1 * W), p);
                fC &= Ops::eq_mask(Ops::load(text + i + j + 2 * W), p);
                fD &= Ops::eq_mask(Ops::load(text + i + j + 3 * W), p);
            }

            if ((fA | fB | fC | fD) == 0) continue;
            // Walk chunks in order so we return the lowest-index match.
            if (fA != 0) return {true, i + 0 * W + (size_t)__builtin_ctz(fA)};
            if (fB != 0) return {true, i + 1 * W + (size_t)__builtin_ctz(fB)};
            if (fC != 0) return {true, i + 2 * W + (size_t)__builtin_ctz(fC)};
            return {true, i + 3 * W + (size_t)__builtin_ctz(fD)};
        }
    } else {
        for (; i + m + B - 1 <= n; i += B) {
            typename Ops::vec_t p = Ops::broadcast(pattern[0]);
            uint32_t fA = Ops::eq_mask(Ops::load(text + i + 0 * W), p);
            uint32_t fB = Ops::eq_mask(Ops::load(text + i + 1 * W), p);
            uint32_t fC = Ops::eq_mask(Ops::load(text + i + 2 * W), p);
            uint32_t fD = Ops::eq_mask(Ops::load(text + i + 3 * W), p);

            for (size_t j = 1; j < m; ++j) {
                p = Ops::broadcast(pattern[j]);
                fA &= Ops::eq_mask(Ops::load(text + i + j + 0 * W), p);
                fB &= Ops::eq_mask(Ops::load(text + i + j + 1 * W), p);
                fC &= Ops::eq_mask(Ops::load(text + i + j + 2 * W), p);
                fD &= Ops::eq_mask(Ops::load(text + i + j + 3 * W), p);
            }

            if ((fA | fB | fC | fD) == 0) continue;
            if (fA != 0) return {true, i + 0 * W + (size_t)__builtin_ctz(fA)};
            if (fB != 0) return {true, i + 1 * W + (size_t)__builtin_ctz(fB)};
            if (fC != 0) return {true, i + 2 * W + (size_t)__builtin_ctz(fC)};
            return {true, i + 3 * W + (size_t)__builtin_ctz(fD)};
        }
    }

    for (; i + m <= n; ++i) {
        if (std::memcmp(text + i, pattern, m) == 0) return {true, i};
    }
    return {false, 0};
}


// StringZilla's three-anchor filter at width W: the width-generic form of
// avx512_stringzilla_find. Three needle offsets chosen by
// sz_locate_needle_anomalies are broadcast and compared against the haystack;
// a candidate must match at all three before a specialised compare verifies it
// in full. The
// filter is three compares per window regardless of the needle length, which is
// what makes it the better half of the scheme for long needles.
// Vectorised equality at width W, the narrow-ISA counterpart of
// sz_equal_avx512. Used when the needle is too long to sit in one register.
template <class Ops>
static inline bool x86_equal_t(const char* a, const char* b, size_t length) {
    constexpr size_t W = Ops::kWidth;
    // (1u << 32) is undefined, so spell the all-lanes mask per width.
    constexpr uint32_t kAll = (W >= 32) ? 0xFFFFFFFFu : ((1u << W) - 1);
    while (length >= W) {
        if (Ops::eq_mask(Ops::load(a), Ops::load(b)) != kAll) return false;
        a += W; b += W; length -= W;
    }
    while (length) {
        if (*a != *b) return false;
        ++a; ++b; --length;
    }
    return true;
}

template <class Ops>
static inline std::pair<bool, size_t> x86_stringzilla_find_t(const char* text, size_t n,
                                                             const char* pattern, size_t m) {
    constexpr size_t W = Ops::kWidth;
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    // A single-byte needle has no three distinct anchors: one broadcast compare
    // per window is already the whole search.
    if (m == 1) {
        const typename Ops::vec_t nv = Ops::broadcast(pattern[0]);
        size_t i = 0;
        for (; i + W <= n; i += W) {
            uint32_t eq = Ops::eq_mask(Ops::load(text + i), nv);
            if (eq != 0) return {true, i + (size_t)__builtin_ctz(eq)};
        }
        for (; i < n; ++i)
            if (text[i] == pattern[0]) return {true, i};
        return {false, 0};
    }

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies(pattern, m, off_first, off_mid, off_last);
    const typename Ops::vec_t first = Ops::broadcast(pattern[off_first]);
    const typename Ops::vec_t mid = Ops::broadcast(pattern[off_mid]);
    const typename Ops::vec_t last = Ops::broadcast(pattern[off_last]);

    // Verification follows the same three paths as the AVX-512 kernel, for the
    // same reason: a per-candidate std::memcmp call costs about ten cycles, and
    // on an input that defeats the filter every haystack position becomes a
    // candidate, so that constant is multiplied by n.
    //
    //   m <= 3   no verification -- the anchors 0, m/2, m-1 already cover every
    //            byte of the needle, so a surviving bit is a confirmed match.
    //   m <= W   the needle sits in one register, compared in a single step.
    //   m > W    x86_equal_t, W bytes per iteration.
    //
    // The narrow ISAs have no byte-granular masked load, so the register-sized
    // needle is staged through a zero-padded buffer once, here, rather than
    // over-reading the needle on every candidate.
    const bool anchors_cover_needle = (m <= 3);
    const bool needle_fits_register = (m <= W);
    alignas(32) char needle_buf[W] = {};
    typename Ops::vec_t needle_vec = first;  // overwritten when used
    uint32_t needle_bits = 0;
    if (needle_fits_register && !anchors_cover_needle) {
        std::memcpy(needle_buf, pattern, m);
        needle_vec = Ops::load(needle_buf);
        needle_bits = (m >= 32) ? 0xFFFFFFFFu : ((1u << m) - 1);
    }

    size_t i = 0;
    // Each step covers candidate positions [i, i + W - 1]. The last anchor of
    // the last candidate reads text[i + W - 1 + off_last] and off_last <= m - 1,
    // so i + m + W - 1 <= n keeps every load in bounds.
    for (; i + m + W - 1 <= n; i += W) {
        uint32_t mask = Ops::eq_mask(Ops::load(text + i + off_first), first);
        if (mask == 0) continue;
        mask &= Ops::eq_mask(Ops::load(text + i + off_mid), mid);
        if (mask == 0) continue;
        mask &= Ops::eq_mask(Ops::load(text + i + off_last), last);

        if (anchors_cover_needle) {
            if (mask != 0) return {true, i + (size_t)__builtin_ctz(mask)};
            continue;
        }

        // A full W-byte load at candidate i+b needs i + b + W <= n, and b can
        // reach W-1. When m < W the loop bound above does not imply that, so
        // check once per window rather than once per candidate; it only fails
        // within the last register-width of the haystack.
        const bool wide_load_safe = (i + 2 * W - 1 <= n);
        while (mask != 0) {
            size_t b = (size_t)__builtin_ctz(mask);
            bool equal;
            if (needle_fits_register && wide_load_safe) {
                equal = (Ops::eq_mask(Ops::load(text + i + b), needle_vec)
                         & needle_bits) == needle_bits;
            } else {
                equal = x86_equal_t<Ops>(text + i + b, pattern, m);
            }
            if (equal) return {true, i + b};
            mask &= mask - 1;  // clear the lowest set bit and continue
        }
    }

    for (; i + m <= n; ++i) {
        if (x86_equal_t<Ops>(text + i, pattern, m)) return {true, i};
    }
    return {false, 0};
}


// --- AVX2 (256-bit) exports ---
inline std::pair<bool, size_t> avx256_naive_search(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_t<x86_ops256>(t, n, p, m); }
inline std::pair<bool, size_t> avx256_naive_search128(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_wide_t<x86_ops256>(t, n, p, m); }
inline std::pair<bool, size_t> avx256_stringzilla_find(const char* t, size_t n, const char* p, size_t m)
    { return x86_stringzilla_find_t<x86_ops256>(t, n, p, m); }
// --- SSE2 (128-bit) exports ---
inline std::pair<bool, size_t> avx128_naive_search(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_t<x86_ops128>(t, n, p, m); }
inline std::pair<bool, size_t> avx128_naive_search64(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_wide_t<x86_ops128>(t, n, p, m); }
inline std::pair<bool, size_t> avx128_stringzilla_find(const char* t, size_t n, const char* p, size_t m)
    { return x86_stringzilla_find_t<x86_ops128>(t, n, p, m); }
