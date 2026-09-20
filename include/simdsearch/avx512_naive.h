#pragma once
// AVX-512 pieces shared by the component kernels (avx512search.h) and
// Needle-Hammer (needle_hammer.h): the masked alignment head, and the naive
// kernels that filter on the needle's first four bytes -- the loop
// Needle-Hammer's wide kernel is built on, and what it dispatches to for a
// four-byte needle. Not part of the public header's include closure.
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

// Scan the first `count` candidate positions with a single masked window.
//
// The strided loops below issue four loads per 64-byte chunk, at offsets 0 to
// 3; the offset-0 load is aligned exactly when the scan pointer is, and a
// misaligned buffer therefore splits a cache line on all four rather than
// three. Walking the scan pointer up to a 64-byte boundary first costs one
// masked window and removes a quarter of the split loads for the rest of the
// search.
//
// Requires count < 64 and count + m - 1 <= n, so every masked lane reads in
// bounds. Returns the first match below `count`, if any.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
avx512_masked_head(const char* text, const char* pattern, size_t m, size_t count) {
    __mmask64 f = (((__mmask64)1 << count) - 1);
    for (size_t k = 0; k < m && f != 0; ++k)
        f = _mm512_mask_cmpeq_epi8_mask(f, _mm512_maskz_loadu_epi8(f, text + k),
                                        _mm512_set1_epi8((char)pattern[k]));
    if (f != 0) return {true, (size_t)__builtin_ctzll(f)};
    return {false, 0};
}

// Candidate positions to skip so that text + head sits on a 64-byte boundary,
// clamped so the masked head never runs past the last candidate position.
static inline size_t avx512_align_head(const char* text, size_t n, size_t m) {
    const size_t off = ((uintptr_t)text) & 63;
    if (off == 0) return 0;
    const size_t head = 64 - off;
    const size_t positions = n - m + 1;          // callers guarantee n >= m
    return head < positions ? head : positions;
}

// Single-window kernel, 64 bytes per iteration.
//
//   independent compares  the four peeled compares do not chain through one
//                         mask register, so they issue in parallel. Chaining
//                         them spends a large share of profiled cycles on the
//                         serial dependency.
//   single-survivor guard when exactly one lane survives -- the usual case at the
//                         matching window -- verify it directly instead of
//                         narrowing. When MANY survive, which is what an
//                         adversary arranges, fall back to narrowing. Verifying
//                         every survivor instead costs one check per lane per
//                         window and is far worse on the adversarial shapes.
//   inline verification   the needle is preloaded once for m <= 64, so a
//                         candidate costs one masked compare rather than a call
//                         into __memcmp_evex_movbe, which profiles as a
//                         significant share of the total.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
avx512_naive_search_body(const char* text, size_t n,
                            const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    size_t i = 0;

    const bool fits = (m <= 64);
    const __mmask64 nmask = fits ? ((m == 64) ? ~(__mmask64)0
                                              : (((__mmask64)1 << m) - 1))
                                 : (__mmask64)0;
    const __m512i nvec = fits ? _mm512_maskz_loadu_epi8(nmask, pattern)
                              : _mm512_setzero_si512();

    if (m >= 4) {
        const size_t head = avx512_align_head(text, n, m);
        if (head) {
            auto r = avx512_masked_head(text, pattern, m, head);
            if (r.first) return r;
            i = head;
        }
        const __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
        const __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
        const __m512i p2 = _mm512_set1_epi8((char)pattern[2]);
        const __m512i p3 = _mm512_set1_epi8((char)pattern[3]);
        for (; i + m + 63 <= n; i += 64) {
            const __mmask64 c0 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+0)), p0);
            const __mmask64 c1 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+1)), p1);
            const __mmask64 c2 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+2)), p2);
            const __mmask64 c3 = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+3)), p3);
            __mmask64 f = (c0 & c1) & (c2 & c3);
            if (f == 0) continue;
            if ((f & (f - 1)) == 0) {
                const size_t b = (size_t)__builtin_ctzll(f);
                if (fits) {
                    if (_mm512_mask_cmpneq_epi8_mask(
                            nmask, _mm512_maskz_loadu_epi8(nmask, text + i + b), nvec) == 0)
                        return {true, i + b};
                } else if (std::memcmp(text + i + b + 4, pattern + 4, m - 4) == 0) {
                    return {true, i + b};
                }
                continue;
            }
            for (size_t k = 4; k < m && f != 0; ++k)
                f = _mm512_mask_cmpeq_epi8_mask(
                        f, _mm512_loadu_si512((const void*)(text + i + k)),
                        _mm512_set1_epi8((char)pattern[k]));
            if (f != 0) return {true, i + (size_t)__builtin_ctzll(f)};
        }
    }
    for (; i + m <= n; i += 64) {
        const size_t cand = n - m - i + 1;
        __mmask64 active = (cand >= 64) ? ~(__mmask64)0 : (((__mmask64)1 << cand) - 1);
        __mmask64 f = active;
        for (size_t k = 0; k < m && f != 0; ++k)
            f = _mm512_mask_cmpeq_epi8_mask(
                    f, _mm512_maskz_loadu_epi8(f, text + i + k),
                    _mm512_set1_epi8((char)pattern[k]));
        if (f != 0) return {true, i + (size_t)__builtin_ctzll(f)};
    }
    return {false, 0};
}

// Wide-stride kernel. The single-survivor guard is on the whole block, not
// per chunk: resolving chunks independently duplicates the pattern broadcasts
// that the shared narrowing loop exists to amortise, and measures worse on
// the adversarial shapes.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
avx512_naive_search256_body(const char* text, size_t n,
                               const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    size_t i = 0;

    const bool fits = (m <= 64);
    const __mmask64 nmask = fits ? ((m == 64) ? ~(__mmask64)0
                                              : (((__mmask64)1 << m) - 1))
                                 : (__mmask64)0;
    const __m512i nvec = fits ? _mm512_maskz_loadu_epi8(nmask, pattern)
                              : _mm512_setzero_si512();

    if (m >= 4) {
        const size_t head = avx512_align_head(text, n, m);
        if (head) {
            auto r = avx512_masked_head(text, pattern, m, head);
            if (r.first) return r;
            i = head;
        }
        const __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
        const __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
        const __m512i p2 = _mm512_set1_epi8((char)pattern[2]);
        const __m512i p3 = _mm512_set1_epi8((char)pattern[3]);
#define AVX512_CHUNK(OFF)                                                          \
        ((_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+0)), p0)   \
        & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+1)), p1))  \
        & (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+2)), p2)  \
        & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+3)), p3)))
        for (; i + m + 255 <= n; i += 256) {
            __mmask64 fA = AVX512_CHUNK(0), fB = AVX512_CHUNK(64);
            __mmask64 fC = AVX512_CHUNK(128), fD = AVX512_CHUNK(192);
            if ((fA | fB | fC | fD) == 0) continue;

            const int nz = (fA != 0) + (fB != 0) + (fC != 0) + (fD != 0);
            const __mmask64 one = fA | fB | fC | fD;
            if (nz == 1 && (one & (one - 1)) == 0) {
                const size_t off = (fA != 0) ? 0 : (fB != 0) ? 64 : (fC != 0) ? 128 : 192;
                const size_t b = off + (size_t)__builtin_ctzll(one);
                if (fits) {
                    if (_mm512_mask_cmpneq_epi8_mask(
                            nmask, _mm512_maskz_loadu_epi8(nmask, text + i + b), nvec) == 0)
                        return {true, i + b};
                } else if (std::memcmp(text + i + b + 4, pattern + 4, m - 4) == 0) {
                    return {true, i + b};
                }
                continue;
            }
            for (size_t k = 4; k < m; ++k) {                  // many: shared narrowing
                if ((fA | fB | fC | fD) == 0) break;
                const __m512i pk = _mm512_set1_epi8((char)pattern[k]);
                fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text+i+k+  0)), pk);
                fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text+i+k+ 64)), pk);
                fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text+i+k+128)), pk);
                fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text+i+k+192)), pk);
            }
            if (fA) return {true, i +   0 + (size_t)__builtin_ctzll(fA)};
            if (fB) return {true, i +  64 + (size_t)__builtin_ctzll(fB)};
            if (fC) return {true, i + 128 + (size_t)__builtin_ctzll(fC)};
            if (fD) return {true, i + 192 + (size_t)__builtin_ctzll(fD)};
        }
#undef AVX512_CHUNK
    }
    // Remainder: hand to the 64-byte kernel, which itself ends in a masked
    // window, so no byte is left to a scalar loop.
    if (i + m <= n) {
        auto r = avx512_naive_search_body(text + i, n - i, pattern, m);
        if (r.first) return {true, i + r.second};
    }
    return {false, 0};
}

