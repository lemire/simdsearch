#pragma once
#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

#include "kmp_twoway.h"
#include "sz_needle_anomalies.h"


// C library strstr. Requires NUL-terminated text and pattern (the benchmark
// inputs are std::string data, which is NUL-terminated and contains no NUL
// bytes). Reports byte offset of the first occurrence.
std::pair<bool, size_t> strstr_search(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const char* hit = std::strstr(text, pattern);
    if (hit == nullptr) return {false, 0};
    return {true, (size_t)(hit - text)};
}

// C library memmem (POSIX 2024; long-standing extension on glibc/BSD/macOS).
// Unlike strstr it is length-delimited, so it needs no NUL terminator and is
// safe on inputs containing NUL bytes.
std::pair<bool, size_t> memmem_search(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const char* hit = (const char*)::memmem(text, n, pattern, m);
    if (hit == nullptr) return {false, 0};
    return {true, (size_t)(hit - text)};
}

// std::search with std::default_searcher (C++17). The searcher is rebuilt per
// call to match the interface of the other functions here (which also do not
// amortize per-pattern preprocessing across calls).
std::pair<bool, size_t> std_default_searcher(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    auto it = std::search(text, text + n,
                          std::default_searcher(pattern, pattern + m));
    if (it == text + n) return {false, 0};
    return {true, (size_t)(it - text)};
}

// std::search with std::boyer_moore_searcher (C++17).
std::pair<bool, size_t> std_boyer_moore_searcher(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    auto it = std::search(text, text + n,
                          std::boyer_moore_searcher(pattern, pattern + m));
    if (it == text + n) return {false, 0};
    return {true, (size_t)(it - text)};
}

// std::search with std::boyer_moore_horspool_searcher (C++17).
std::pair<bool, size_t> std_boyer_moore_horspool_searcher(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    auto it = std::search(text, text + n,
                          std::boyer_moore_horspool_searcher(pattern, pattern + m));
    if (it == text + n) return {false, 0};
    return {true, (size_t)(it - text)};
}


// Boyer–Moore–Horspool. Reference scalar implementation: builds a bad-character
// shift table from the pattern, then scans the text shifting by the table entry
// for the last text byte of the current window. The table uses uint8_t so init
// is one cache-line-friendly memset; shifts are clamped to 255 (smaller shifts
// stay correct, just non-optimal for needles longer than 255 bytes).
std::pair<bool, size_t> bmh_search(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    uint8_t shift[256];
    uint8_t default_shift = (m > 255) ? 255 : (uint8_t)m;
    std::memset(shift, default_shift, sizeof(shift));
    size_t pre = (m > 256) ? m - 256 : 0;
    for (size_t i = pre; i + 1 < m; ++i) {
        shift[(uint8_t)pattern[i]] = (uint8_t)(m - 1 - i);
    }

    size_t i = 0;
    const size_t last = m - 1;
    while (i + m <= n) {
        uint8_t c = (uint8_t)text[i + last];
        if (c == (uint8_t)pattern[last] &&
            std::memcmp(text + i, pattern, last) == 0) {
            return {true, i};
        }
        i += shift[c];
    }
    return {false, 0};
}


// Boyer–Moore–Horspool with a 16-bit shift table. Identical to bmh_search but
// the bad-character shifts are stored as uint16_t, so they clamp at 65535
// instead of 255: needles up to 65535 bytes get their full skip distance, which
// matters for long patterns where the 8-bit version caps every skip at 255.
std::pair<bool, size_t> bmh_search16(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    uint16_t shift[256];
    uint16_t default_shift = (m > 65535) ? 65535 : (uint16_t)m;
    std::fill(shift, shift + 256, default_shift);
    size_t pre = (m > 65536) ? m - 65536 : 0;
    for (size_t i = pre; i + 1 < m; ++i) {
        shift[(uint8_t)pattern[i]] = (uint16_t)(m - 1 - i);
    }

    size_t i = 0;
    const size_t last = m - 1;
    while (i + m <= n) {
        uint8_t c = (uint8_t)text[i + last];
        if (c == (uint8_t)pattern[last] &&
            std::memcmp(text + i, pattern, last) == 0) {
            return {true, i};
        }
        i += shift[c];
    }
    return {false, 0};
}


// return bool and index of first occurrence
// based on https://onlinelibrary.wiley.com/doi/pdf/10.1002/spe.2511
//
// AVX-512 port of the NEON "naive" searcher. Each step consumes a 64-byte
// window (one ZMM register). For pattern byte j we broadcast it across all 64
// lanes and compare against text[i+j .. i+j+63]; the per-position match state
// lives in a 64-bit __mmask64 that is narrowed by each successive byte using
// _mm512_mask_cmpeq_epi8_mask (a fused "compare only where still alive, then
// AND"). When every candidate has died the inner loop bails early; a surviving
// mask gives the answer via the lowest set bit.
std::pair<bool, size_t> avx512_naive_search(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const size_t step = 64;

    size_t i = 0;
    // SIMD chunk reads bytes [i, i + 63 + (m - 1)], so require i + m + 63 <= n.
    if (n >= m + 63) {
        if (m >= 4) {
            for (; i + m + 63 <= n; i += step) {
                __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
                __mmask64 found = _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512((const void*)(text + i)), p0);
                __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
                found = _mm512_mask_cmpeq_epi8_mask(
                    found, _mm512_loadu_si512((const void*)(text + i + 1)), p1);
                __m512i p2 = _mm512_set1_epi8((char)pattern[2]);
                found = _mm512_mask_cmpeq_epi8_mask(
                    found, _mm512_loadu_si512((const void*)(text + i + 2)), p2);
                __m512i p3 = _mm512_set1_epi8((char)pattern[3]);
                found = _mm512_mask_cmpeq_epi8_mask(
                    found, _mm512_loadu_si512((const void*)(text + i + 3)), p3);
                for (size_t j = 4; j < m; ++j) {
                    if (found == 0) break;
                    __m512i pj = _mm512_set1_epi8((char)pattern[j]);
                    found = _mm512_mask_cmpeq_epi8_mask(
                        found, _mm512_loadu_si512((const void*)(text + i + j)), pj);
                }
                if (found == 0) continue;
                return {true, i + (size_t)__builtin_ctzll(found)};
            }
        } else {
            for (; i + m + 63 <= n; i += step) {
                __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
                __mmask64 found = _mm512_cmpeq_epi8_mask(
                    _mm512_loadu_si512((const void*)(text + i)), p0);
                for (size_t j = 1; j < m; ++j) {
                    if (found == 0) break;
                    __m512i pj = _mm512_set1_epi8((char)pattern[j]);
                    found = _mm512_mask_cmpeq_epi8_mask(
                        found, _mm512_loadu_si512((const void*)(text + i + j)), pj);
                }
                if (found == 0) continue;
                return {true, i + (size_t)__builtin_ctzll(found)};
            }
        }
    }

    // Scalar tail for positions the SIMD loop could not safely cover.
    for (; i + m <= n; ++i) {
        if (std::memcmp(text + i, pattern, m) == 0) {
            return {true, i};
        }
    }

    return {false, 0};
}

// The shipping form. Three ideas, each needed for a different reason:
//
//   independent compares  the four peeled compares do not chain through one
//                         mask register, so they issue in parallel. Chaining
//                         them costs ~50% of profiled cycles to the serial
//                         dependency.
//   single-survivor guard when exactly one lane survives -- the usual case at the
//                         matching window -- verify it directly instead of
//                         narrowing. When MANY survive, which is what an
//                         adversary arranges, fall back to narrowing. Verifying
//                         every survivor instead costs 64 checks per window and
//                         is up to 10x worse on the adversarial shapes.
//   inline verification   the needle is preloaded once for m <= 64, so a
//                         candidate costs one masked compare rather than a call
//                         into __memcmp_evex_movbe (12% of cycles).
static inline __attribute__((always_inline)) std::pair<bool, size_t>
avx512_naive_search_v3_body(const char* text, size_t n,
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
            if ((f & (f - 1)) == 0) {                       // one survivor
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
            for (size_t k = 4; k < m && f != 0; ++k)        // many: narrow
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

std::pair<bool, size_t> avx512_naive_search_v3(const char* text, size_t n,
                                               const char* pattern, size_t m) {
    return avx512_naive_search_v3_body(text, n, pattern, m);
}

// Wide-stride kernel, same three ideas as v3. The single-survivor guard is on the
// WHOLE block, not per chunk: resolving chunks independently duplicates the
// pattern broadcasts that the shared narrowing loop exists to amortise, which
// measured 2.2x worse on the adversarial shapes.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
avx512_naive_search256_v3_body(const char* text, size_t n,
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
            if (nz == 1 && (one & (one - 1)) == 0) {          // one survivor in the block
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
        auto r = avx512_naive_search_v3_body(text + i, n - i, pattern, m);
        if (r.first) return {true, i + r.second};
    }
    return {false, 0};
}

std::pair<bool, size_t> avx512_naive_search256_v3(const char* text, size_t n,
                                                  const char* pattern, size_t m) {
    return avx512_naive_search256_v3_body(text, n, pattern, m);
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
void avx512_naive_search_all(const char* text, size_t n, const char* pattern,
                             size_t m, F callback) {
    if (m == 0 || n < m) return;

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


// Same as avx512_naive_search but with a 256-byte stride: each iteration loads
// four 64-byte chunks (A/B/C/D), so each _mm512_set1_epi8 broadcast of a
// pattern byte is reused across four match accumulators instead of one. This
// mirrors the NEON neon_naive_search64 four-chunk unrolling, scaled from 16- to
// 64-byte lanes.
std::pair<bool, size_t> avx512_naive_search256(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    // SIMD reads bytes [i, i + 255 + (m - 1)], so require i + m + 255 <= n.
    if (n >= m + 255) {
        if (m >= 4) {
            for (; i + m + 255 <= n; i += 256) {
                __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
                __mmask64 fA = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i +   0)), p0);
                __mmask64 fB = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i +  64)), p0);
                __mmask64 fC = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i + 128)), p0);
                __mmask64 fD = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i + 192)), p0);

                __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
                fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text + i +   1)), p1);
                fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text + i +  65)), p1);
                fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text + i + 129)), p1);
                fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text + i + 193)), p1);

                __m512i p2 = _mm512_set1_epi8((char)pattern[2]);
                fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text + i +   2)), p2);
                fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text + i +  66)), p2);
                fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text + i + 130)), p2);
                fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text + i + 194)), p2);

                __m512i p3 = _mm512_set1_epi8((char)pattern[3]);
                fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text + i +   3)), p3);
                fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text + i +  67)), p3);
                fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text + i + 131)), p3);
                fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text + i + 195)), p3);

                for (size_t j = 4; j < m; ++j) {
                    if ((fA | fB | fC | fD) == 0) break;
                    __m512i pj = _mm512_set1_epi8((char)pattern[j]);
                    fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text + i + j +   0)), pj);
                    fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text + i + j +  64)), pj);
                    fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text + i + j + 128)), pj);
                    fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text + i + j + 192)), pj);
                }

                if ((fA | fB | fC | fD) == 0) continue;

                // Walk chunks in order so we return the lowest-index match.
                if (fA != 0) return {true, i +   0 + (size_t)__builtin_ctzll(fA)};
                if (fB != 0) return {true, i +  64 + (size_t)__builtin_ctzll(fB)};
                if (fC != 0) return {true, i + 128 + (size_t)__builtin_ctzll(fC)};
                return {true, i + 192 + (size_t)__builtin_ctzll(fD)};
            }
        } else {
            // Short needles (m = 1, 2, 3): keep the 256-byte stride but build
            // the four accumulators with a generic loop from byte 0, so short
            // patterns stay on the SIMD path instead of dropping to the scalar
            // tail over the whole haystack.
            for (; i + m + 255 <= n; i += 256) {
                __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
                __mmask64 fA = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i +   0)), p0);
                __mmask64 fB = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i +  64)), p0);
                __mmask64 fC = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i + 128)), p0);
                __mmask64 fD = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text + i + 192)), p0);

                for (size_t j = 1; j < m; ++j) {
                    __m512i pj = _mm512_set1_epi8((char)pattern[j]);
                    fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text + i + j +   0)), pj);
                    fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text + i + j +  64)), pj);
                    fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text + i + j + 128)), pj);
                    fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text + i + j + 192)), pj);
                }

                if ((fA | fB | fC | fD) == 0) continue;

                // Walk chunks in order so we return the lowest-index match.
                if (fA != 0) return {true, i +   0 + (size_t)__builtin_ctzll(fA)};
                if (fB != 0) return {true, i +  64 + (size_t)__builtin_ctzll(fB)};
                if (fC != 0) return {true, i + 128 + (size_t)__builtin_ctzll(fC)};
                return {true, i + 192 + (size_t)__builtin_ctzll(fD)};
            }
        }
    }

    // Scalar tail for positions the 256-byte SIMD loop could not safely cover.
    for (; i + m <= n; ++i) {
        if (std::memcmp(text + i, pattern, m) == 0) {
            return {true, i};
        }
    }

    return {false, 0};
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
// Result of a kernel that may abandon its scan. `resume` is the first position
// not yet ruled out, so the fallback searcher need not rescan the prefix.
struct avx512_guarded_result {
    bool found;
    size_t index;
    bool gave_up;    // budget exhausted; caller must fall back
    size_t resume;   // first position not yet ruled out, when gave_up
};

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
// so that neither can bail, the guarded instantiation measures 2251 ns against
// 3621 ns for identical work at m = 2048. One body removes that discrepancy
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
static inline avx512_guarded_result avx512_stringzilla_body(
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
            if (verified > budget_bytes) return {false, 0, true, i};
            mask &= mask - 1;  // clear the lowest set bit and continue
        }
    }

    return {false, 0, false, 0};
}

// The kernel as the algorithm table sees it: the shared body with a budget it
// can never exhaust, so this is the original algorithm with no guard behaviour.
std::pair<bool, size_t> avx512_stringzilla_find(const char* haystack, size_t h_len,
                                                const char* needle, size_t n_len)
{
    auto r = avx512_stringzilla_body<false>(haystack, h_len, needle, n_len,
                                            ~(size_t)0);  // unreachable budget
    return {r.found, r.index};
}

// The same kernel with upstream's UTF-8 lead-byte rule enabled, so the cost of
// that choice can be measured rather than assumed. Not the default: see
// sz_needle_anomalies.h.
std::pair<bool, size_t> avx512_stringzilla_find_hifilter(const char* haystack,
                                                         size_t h_len,
                                                         const char* needle,
                                                         size_t n_len)
{
    auto r = avx512_stringzilla_body<true>(haystack, h_len, needle, n_len,
                                           ~(size_t)0);
    return {r.found, r.index};
}


// Length-dispatched needle-hammer: the 256-byte-stride naive kernel for needles up to
// Threshold bytes, the StringZilla 3-anchor kernel above it.
//
// The two kernels fail in opposite directions. avx512_naive_search256 spends
// O(m) broadcast+compare rounds per 256-byte block whenever the needle's first
// four bytes keep candidates alive, so its cost grows with the needle length;
// but when the prefix filter bites (ordinary text) it is the fastest kernel
// here, because each pattern-byte broadcast is amortized over four 64-byte
// chunks. avx512_stringzilla_find is O(1) filter rounds per window no matter
// how long the needle is -- three anchors, chosen away from repeats -- but pays
// a per-window setup and a verifying compare per surviving candidate, which the naive
// kernel's mask narrowing avoids on short needles.
//
// So: cheap-and-length-sensitive below the threshold, length-insensitive above.
//
// Short-haystack guard. The 256-byte kernel cannot report a match until it has
// finished the entire block the match sits in -- it builds four chunk masks and
// only then scans them -- whereas the 64-byte kernel returns as soon as its one
// chunk has a hit. On a small haystack the match tends to arrive early relative
// to 256 bytes, so that block granularity costs up to 4x the compare work for
// nothing. Below kMinWide the 64-byte kernel is therefore the better lower half.
// Measured on a 1 KB haystack (synthetic mode): 133-159 ns without the guard
// versus 43-54 ns with it, matching find_avx512. A 8192-byte cutoff measures the
// same, so take the smaller.
//
// The second condition is the narrower structural one: the wide loop only runs
// while i + m + 255 <= n, and anything it cannot cover drops to a scalar memcmp
// loop. It matters only for needles too long for kMinWide to already catch.
template <size_t Threshold>
static inline std::pair<bool, size_t> avx512_needle_hammer_t(const char* text, size_t n,
                                                           const char* pattern, size_t m) {
    // 1024, not the 2048 the narrow widths use. The guard existed because the
    // wide kernel collapsed on small haystacks -- its 256-byte loop left up to
    // 255 positions to a scalar memcmp. That tail is gone here and the kernel
    // now falls through to the 64-byte path below m + 255, so it degenerates to
    // the single-window kernel rather than degrading. Measured across the fleet
    // the wide kernel is at least as good from 1024 bytes up; only at 512 do the
    // two AMD parts still prefer single. The narrow widths keep 2048 because
    // their kernels still have the scalar tail this value was chosen to avoid.
    constexpr size_t kMinWide = 1024;
    if (m > Threshold) return avx512_stringzilla_find(text, n, pattern, m);
    if (n < kMinWide || n < m + 255) return avx512_naive_search_v3_body(text, n, pattern, m);
    return avx512_naive_search256_v3_body(text, n, pattern, m);
}

// The scheme itself, switching at 512 bytes of needle.
//
// Chosen from all five benchmarked machines, not from one, because the two
// criteria that bear on it disagree and they disagree differently per vendor.
//
// Robustness. The switch point can only protect the adversarial crossover while
// it sits BELOW the wide kernel's own crossover with two-way, W*. Re-measured
// against the current kernels, W* is 128/113/119 bytes on Emerald/Granite/
// Sapphire Rapids and 87/104 on Zen 5/Zen 4, so the fleet floor is 128: any
// threshold at or above it is adversarially maximal everywhere, and none below
// it is. Above the floor the choice is purely benign.
//
// Benign. The anchored kernel is far weaker relative to the wide one on Zen, so
// Intel wants a small threshold and AMD a large one and no single constant is
// optimal for both. Geometric mean over the length sweep, as regret against the
// best switch point for that machine and length, worst over the fleet:
//
//   T          16     32     64    128    256    512
//   worst    1.86x  1.63x  1.44x  1.27x  1.13x  1.07x
//
// 512 clears the floor and minimises worst-case regret. The regret is
// almost entirely one vendor's: the Intel parts sit within 6.5% at every
// candidate, while Zen 5 pays 86% at T = 16 and nothing at 512.
//
// For a genuine worst-case bound use avx512_needle_hammer_guarded, which bounds
// both kernels rather than hoping the threshold avoids them -- and which, being
// bounded by construction, could take the benign optimum instead.
std::pair<bool, size_t> avx512_needle_hammer(const char* text, size_t n,
                                            const char* pattern, size_t m) {
    return avx512_needle_hammer_t<512>(text, n, pattern, m);
}

// Same scheme at other switch points, so the crossover can be located
// empirically rather than assumed.
std::pair<bool, size_t> avx512_needle_hammer16(const char* t, size_t n, const char* p, size_t m)
    { return avx512_needle_hammer_t<16>(t, n, p, m); }
std::pair<bool, size_t> avx512_needle_hammer32(const char* t, size_t n, const char* p, size_t m)
    { return avx512_needle_hammer_t<32>(t, n, p, m); }
std::pair<bool, size_t> avx512_needle_hammer64(const char* t, size_t n, const char* p, size_t m)
    { return avx512_needle_hammer_t<64>(t, n, p, m); }
std::pair<bool, size_t> avx512_needle_hammer128(const char* t, size_t n, const char* p, size_t m)
    { return avx512_needle_hammer_t<128>(t, n, p, m); }
std::pair<bool, size_t> avx512_needle_hammer256(const char* t, size_t n, const char* p, size_t m)
    { return avx512_needle_hammer_t<256>(t, n, p, m); }
std::pair<bool, size_t> avx512_needle_hammer512(const char* t, size_t n, const char* p, size_t m)
    { return avx512_needle_hammer_t<512>(t, n, p, m); }

// ===========================================================================
// Work-counter guarded needle-hammer
// ===========================================================================
//
// Needle-hammer is fast on real data and degrades to O(n*m) on inputs chosen to
// defeat its filter. The classical answer -- pair the filter with a linear-time
// searcher -- is usually stated as a needle-length rule, but needle length is
// only a proxy: it is the *selectivity of the filter on this haystack* that
// decides, and that is observable at run time. So observe it.
//
// Each kernel below counts the work its verification path actually does and
// abandons the scan for twoway_bc_search once that count exceeds a budget
// proportional to n. Because the budget is proportional to n and two-way is
// linear, the guarded searcher is O(n) on every input, including the ones that
// drive the unguarded kernels quadratic -- while on benign text the counter
// never approaches the budget and the fast path is untouched.
//
// Choosing the unit matters more than choosing the constant. We count only work
// that a *failing* filter causes, not the fixed cost of filtering:
//
//   wide kernel      one "narrowing round" = one broadcast plus four masked
//                    compares over a 256-byte block, counted only for rounds
//                    past the four unrolled ones. On ordinary text the mask is
//                    empty by round four and this loop does not execute at all,
//                    so the benign count is ~0 rather than merely small.
//
//   anchored kernel  bytes examined by its verifying compare (m per surviving
//                    candidate). On ordinary text three anchors admit almost no
//                    candidates, so again ~0.
//
// The budgets are set from an instruction model. Two-way retires
// about 14 instructions per haystack byte; a narrowing round is about 13
// instructions per 256 bytes covered. Allowing n rounds therefore spends roughly
// as many instructions as two-way would, so a guarded search that gives up costs
// about twice two-way and never less than two-way's guarantee.
//
// Where the guard can fire. A block admits at most m-4 rounds, so the whole
// haystack admits at most n(m-4)/256, and the budget of n/8 is out of reach
// exactly when m <= 36. That is the free path in the dispatcher, and it is the
// only length range where the guard provably costs nothing. Above it the guard
// can fire even where the unguarded kernel would have won: on the tail shape at
// m = 64 it gives up and pays 41 us against the unguarded kernel's 26 us. It is
// not free there. Note that 216 bytes, the instruction-model crossover with
// two-way quoted above, does not bound this counter and says nothing about when
// the guard fires. What the guard buys for
// that 57% is the bound: the same cell reaches 207 us unguarded at m = 512 and
// keeps climbing, while guarded it stops at 66 us, under two-way's 53 us doubled.
//
// The anchored budget is 16n bytes of verification, which costs well under one
// instruction per haystack byte (the compare is vectorised) and so is nearly free
// relative to the two-way run that follows it.

// Wide-stride kernel with a bounded narrowing loop. Body-for-body the same as
// avx512_naive_search256_v3_body with the counter compiled in, so any difference
// between guarded and unguarded is the guard and nothing else. Only the m >= 4
// path is instrumented: for m < 4 the narrowing loop cannot run more than twice
// per block, so its total work is bounded by 2n/256 rounds and no guard is
// needed.
static inline avx512_guarded_result avx512_naive_search256_guarded(
    const char* text, size_t n, const char* pattern, size_t m,
    size_t budget_rounds) {
    if (m == 0) return {true, 0, false, 0};
    if (n < m) return {false, 0, false, 0};
    if (m < 4 || n < m + 255) {
        auto [f, idx] = avx512_naive_search256_v3_body(text, n, pattern, m);
        return {f, idx, false, 0};
    }

    const bool fits = (m <= 64);
    const __mmask64 nmask = fits ? ((m == 64) ? ~(__mmask64)0
                                              : (((__mmask64)1 << m) - 1))
                                 : (__mmask64)0;
    const __m512i nvec = fits ? _mm512_maskz_loadu_epi8(nmask, pattern)
                              : _mm512_setzero_si512();

    const __m512i p0 = _mm512_set1_epi8((char)pattern[0]);
    const __m512i p1 = _mm512_set1_epi8((char)pattern[1]);
    const __m512i p2 = _mm512_set1_epi8((char)pattern[2]);
    const __m512i p3 = _mm512_set1_epi8((char)pattern[3]);
#define AVX512_CHUNK(OFF)                                                          \
        ((_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+0)), p0)   \
        & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+1)), p1))  \
        & (_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+2)), p2)  \
        & _mm512_cmpeq_epi8_mask(_mm512_loadu_si512((const void*)(text+i+(OFF)+3)), p3)))

    size_t rounds = 0;
    size_t i = 0;
    for (; i + m + 255 <= n; i += 256) {
        __mmask64 fA = AVX512_CHUNK(0), fB = AVX512_CHUNK(64);
        __mmask64 fC = AVX512_CHUNK(128), fD = AVX512_CHUNK(192);
        if ((fA | fB | fC | fD) == 0) continue;

        // One survivor in the block: verified in place, no narrowing, so nothing
        // is charged. That is not a hole an adversary can widen. The path costs
        // at most m bytes of vector compare per 256 bytes of haystack, so even
        // at the largest needle the wide kernel ever sees (m = tau = 512) it is
        // two compared bytes per haystack byte, a few hundredths of an
        // instruction each -- far below the ~14 instructions per byte that
        // two-way would spend on the same text. The expensive path is the one
        // with many survivors, and that is the one counted.
        const int nz = (fA != 0) + (fB != 0) + (fC != 0) + (fD != 0);
        const __mmask64 one = fA | fB | fC | fD;
        if (nz == 1 && (one & (one - 1)) == 0) {
            const size_t off = (fA != 0) ? 0 : (fB != 0) ? 64 : (fC != 0) ? 128 : 192;
            const size_t b = off + (size_t)__builtin_ctzll(one);
            if (fits) {
                if (_mm512_mask_cmpneq_epi8_mask(
                        nmask, _mm512_maskz_loadu_epi8(nmask, text + i + b), nvec) == 0)
                    return {true, i + b, false, 0};
            } else if (std::memcmp(text + i + b + 4, pattern + 4, m - 4) == 0) {
                return {true, i + b, false, 0};
            }
            continue;
        }

        // The only unbounded loop in this kernel, and so the only one counted.
        // The budget is tested once per 256-byte block rather than once per
        // round: keeping the inner loop byte-identical to the unguarded kernel
        // is what makes the benign overhead vanish (testing per round cost 8% on
        // the find-all workload). Overshoot is at most m-4 rounds, negligible
        // against a budget of n/8.
        size_t j = 4;
        for (; j < m; ++j) {
            if ((fA | fB | fC | fD) == 0) break;
            const __m512i pj = _mm512_set1_epi8((char)pattern[j]);
            fA = _mm512_mask_cmpeq_epi8_mask(fA, _mm512_loadu_si512((const void*)(text+i+j+  0)), pj);
            fB = _mm512_mask_cmpeq_epi8_mask(fB, _mm512_loadu_si512((const void*)(text+i+j+ 64)), pj);
            fC = _mm512_mask_cmpeq_epi8_mask(fC, _mm512_loadu_si512((const void*)(text+i+j+128)), pj);
            fD = _mm512_mask_cmpeq_epi8_mask(fD, _mm512_loadu_si512((const void*)(text+i+j+192)), pj);
        }
        rounds += j - 4;

        // Resolve before testing the budget. The narrowing above is spent
        // whether or not the budget survives it, so giving up on a block that
        // just produced the answer would discard a proven match and pay two-way
        // to find it again.
        if (fA) return {true, i +   0 + (size_t)__builtin_ctzll(fA), false, 0};
        if (fB) return {true, i +  64 + (size_t)__builtin_ctzll(fB), false, 0};
        if (fC) return {true, i + 128 + (size_t)__builtin_ctzll(fC), false, 0};
        if (fD) return {true, i + 192 + (size_t)__builtin_ctzll(fD), false, 0};
        if (rounds > budget_rounds) return {false, 0, true, i};
    }
#undef AVX512_CHUNK

    // Remainder: fewer than m + 255 bytes left, so the narrowing the 64-byte
    // kernel may do there is bounded by a constant independent of n. Uncounted
    // for the same reason the m < 4 case is.
    if (i + m <= n) {
        auto r = avx512_naive_search_v3_body(text + i, n - i, pattern, m);
        if (r.first) return {true, i + r.second, false, 0};
    }
    return {false, 0, false, 0};
}

// Anchored kernel with a bounded verification budget, counted in bytes examined
// by memcmp. Same body as the unguarded kernel above, with the counter compiled
// in, so any difference between the two is the guard and nothing else.
static inline avx512_guarded_result avx512_stringzilla_guarded(
    const char* text, size_t n, const char* pattern, size_t m,
    size_t budget_bytes) {
    return avx512_stringzilla_body<false>(text, n, pattern, m, budget_bytes);
}

// Hand the rest of the haystack to Crochemore-Perrin. The kernel has already
// proved there is no occurrence starting below `from`, so two-way only has to
// cover the remainder -- the abandoned prefix is not rescanned. Plain two-way
// rather than the bad-character variant: on the low-diversity haystacks that
// trip the guard the skip table buys nothing and measures ~4x slower.
static inline std::pair<bool, size_t> resume_twoway(const char* text, size_t n,
                                                    const char* pattern, size_t m,
                                                    size_t from) {
    if (from + m > n) return {false, 0};
    auto [f, idx] = twoway_search(text + from, n - from, pattern, m);
    if (!f) return {false, 0};
    return {true, from + idx};
}

// The guarded scheme. BudgetNum/BudgetDen scales both budgets against n so the
// constant can be measured rather than asserted:
//   wide kernel      budget = n * BudgetNum / BudgetDen narrowing rounds
//   anchored kernel  budget = 16 * that, in bytes of verification
// The shipped setting is n/8 rounds, measured so that a search that gives up
// costs about twice two-way rather than several times it.
template <size_t Threshold, size_t BudgetNum = 1, size_t BudgetDen = 1>
static inline std::pair<bool, size_t> avx512_needle_hammer_guarded_t(
    const char* text, size_t n, const char* pattern, size_t m) {
    constexpr size_t kMinWide = 1024;
    // Needles this short cannot exhaust the budget however hostile the input:
    // the narrowing loop runs at most m-4 times per 256-byte block, i.e. about
    // (n/256)(m-4) rounds, which stays under n/BudgetDen exactly when
    // m - 4 <= 256/BudgetDen. Below that bound the guard is not merely cheap,
    // it is absent -- the call goes straight to the unguarded kernel.
    constexpr size_t kFreeBelow = 4 + (256 * BudgetNum) / BudgetDen;

    if (m > Threshold) {
        // The anchored budget must scale with the needle as well as the
        // haystack. Benign verification work is (candidates surviving three
        // anchors) x m, and on a long needle drawn from ordinary text the
        // anchors are common letters, so a flat byte budget trips on perfectly
        // benign input.
        //
        // Budget the verification in bytes against the HAYSTACK, not the
        // needle. A max(2n, 128m) form, with the 128m floor there to stop
        // benign long-needle searches tripping, makes the worst case grow
        // linearly in m (38 us at m = 8192 against 13 us at m = 2048). Scaling
        // with n alone fixes both ends: 16n is far above the
        // benign verification load even for an 8 KB needle drawn from ordinary
        // text, and it caps adversarial verification independently of m. At
        // roughly one instruction per 32 bytes compared, 16n bytes is about n/2
        // instructions -- negligible beside the two-way run it precedes.
        const size_t anchored_budget = 16 * n;
        auto r = avx512_stringzilla_guarded(text, n, pattern, m, anchored_budget);
        if (!r.gave_up) return {r.found, r.index};
        return resume_twoway(text, n, pattern, m, r.resume);
    }
    // Small haystacks take the 64-byte kernel, which needs no guard: it is
    // entered only when n < kMinWide or n < m + 255, and in both cases the
    // number of window positions is O(m) rather than O(n), so its total work is
    // already bounded independently of the haystack length.
    if (n < kMinWide || n < m + 255) {
        return avx512_naive_search_v3_body(text, n, pattern, m);
    }
    // The free case: one comparison against a compile-time constant, then the
    // unguarded kernel. This is the branch the short needles of real workloads
    // take, which is why the guard costs them nothing measurable.
    if (m <= kFreeBelow) return avx512_naive_search256_v3_body(text, n, pattern, m);

    const size_t budget = (n / BudgetDen) * BudgetNum + 1;
    auto r = avx512_naive_search256_guarded(text, n, pattern, m, budget);
    if (!r.gave_up) return {r.found, r.index};
    return resume_twoway(text, n, pattern, m, r.resume);
}

// The recommended configuration: the same 256-byte switch point as
// avx512_needle_hammer, with a budget of n/8 narrowing rounds.
std::pair<bool, size_t> avx512_needle_hammer_guarded(const char* t, size_t n,
                                                    const char* p, size_t m)
    { return avx512_needle_hammer_guarded_t<512, 1, 8>(t, n, p, m); }

// Tighter and looser budgets, so the cost of the guard and the sharpness of the
// bound can both be measured instead of assumed.
std::pair<bool, size_t> avx512_needle_hammer_guarded_tight(const char* t, size_t n,
                                                           const char* p, size_t m)
    { return avx512_needle_hammer_guarded_t<512, 1, 32>(t, n, p, m); }
std::pair<bool, size_t> avx512_needle_hammer_guarded_loose(const char* t, size_t n,
                                                           const char* p, size_t m)
    { return avx512_needle_hammer_guarded_t<512, 1, 2>(t, n, p, m); }


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
std::pair<bool, size_t> avx512_stringzilla256_find(const char* text, size_t n,
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


// Needle-hammer at width W. Identical dispatch to avx512_needle_hammer_t, with
// both cutoffs scaled to the register width:
//
//   m > Threshold                  -> the anchored kernel (length-insensitive)
//   n < MinWide or too short for
//   one wide block                 -> the single-window naive kernel
//   otherwise                      -> the four-chunk naive kernel
//
// MinWide is the short-haystack guard: the wide kernel cannot report a match
// until it has filtered the entire 4W-byte block the match sits in, while the
// single-window kernel returns as soon as its one chunk has a hit, so on a
// small haystack the block granularity is paid for nothing.
//
// It is an absolute byte count, not a multiple of the block size. Scaling it
// per width (8 blocks: 2048 / 1024 / 512) was the obvious guess and it measured
// wrong -- on the 1 KB haystack of `synthetic` the wide kernel loses at every
// width (AVX2: 74 ns vs 49 ns for the single-window kernel; SSE2: 79 vs 65),
// so the cutoff has to sit above 1024 everywhere. The scaled rule put it at
// 1024 and 512, i.e. exactly below the haystack, and both narrow hammers took
// the wide path and lost. 2048 is the value measured for AVX-512 and it is the
// right side of that boundary for all three widths.
template <class Ops, size_t Threshold, size_t MinWide = 2048>
static inline std::pair<bool, size_t> x86_needle_hammer_t(const char* text, size_t n,
                                                          const char* pattern, size_t m) {
    constexpr size_t B = 4 * Ops::kWidth;
    if (m > Threshold) return x86_stringzilla_find_t<Ops>(text, n, pattern, m);
    if (n < MinWide || n < m + B - 1) return x86_naive_search_t<Ops>(text, n, pattern, m);
    return x86_naive_search_wide_t<Ops>(text, n, pattern, m);
}


// --- AVX2 (256-bit) exports ---
std::pair<bool, size_t> avx256_naive_search(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_t<x86_ops256>(t, n, p, m); }
std::pair<bool, size_t> avx256_naive_search128(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_wide_t<x86_ops256>(t, n, p, m); }
std::pair<bool, size_t> avx256_stringzilla_find(const char* t, size_t n, const char* p, size_t m)
    { return x86_stringzilla_find_t<x86_ops256>(t, n, p, m); }
// Switching at 3072, not at AVX-512's 128: the threshold belongs to the width,
// not to the scheme. See the note above the SSE2 export for the measurements
// and for why the narrow widths are chosen on benign data alone.
std::pair<bool, size_t> avx256_needle_hammer(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops256, 3072>(t, n, p, m); }
// Other switch points, so the crossover stays measurable at this width too.
std::pair<bool, size_t> avx256_needle_hammer64(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops256, 64>(t, n, p, m); }
std::pair<bool, size_t> avx256_needle_hammer512(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops256, 512>(t, n, p, m); }
std::pair<bool, size_t> avx256_needle_hammer8192(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops256, 8192>(t, n, p, m); }

// --- SSE2 (128-bit) exports ---
std::pair<bool, size_t> avx128_naive_search(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_t<x86_ops128>(t, n, p, m); }
std::pair<bool, size_t> avx128_naive_search64(const char* t, size_t n, const char* p, size_t m)
    { return x86_naive_search_wide_t<x86_ops128>(t, n, p, m); }
std::pair<bool, size_t> avx128_stringzilla_find(const char* t, size_t n, const char* p, size_t m)
    { return x86_stringzilla_find_t<x86_ops128>(t, n, p, m); }
// Switching at 8192. The crossover moves out sharply as the register narrows --
// 128 bytes at 512-bit, ~3K at 256-bit, ~8K at 128-bit -- because the anchored
// kernel filters only W candidate positions per window for a fixed three loads,
// three movemask round-trips through a GPR and two early-out branches, so its
// per-position overhead grows as the width shrinks while the naive wide kernel
// pays no such fixed cost. Re-measured against the fixed anchored kernel
// (horspool, ns per search, this machine):
//
//   width  kernel           1536   2048   3072   4096   8192  12288
//   256b   naive wide       6309   7015   8830  10294  16724      -
//   256b   anchored         8843   8574   8912   8754   8375      -  -> ~3.2K
//   128b   naive wide          -   8418      -  11007  16067  21024
//   128b   anchored            -  16482      -  16411  16057  15448  -> ~8.2K
//
// Unlike the AVX-512 kernel, these two thresholds are set by benign data alone,
// and that is a conclusion rather than an oversight. At 512-bit the threshold
// also decides the adversarial crossover, because the anchored kernel survives
// two of the three adversaries and the switch point is what hands work to it.
// Here it decides nothing: on the all-common-byte shape the narrow anchored
// kernels are worse than their own wide parents at every length measured
// (256-bit: 13149 vs 6696 ns at m = 32, 21004 vs 13141 at 64; 128-bit: 18344 vs
// 10347 and 27341 vs 20634), so switching earlier makes the worst case worse as
// well as the benign case. Nothing is traded away by putting these thresholds
// where benign data wants them.
std::pair<bool, size_t> avx128_needle_hammer(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops128, 8192>(t, n, p, m); }
std::pair<bool, size_t> avx128_needle_hammer64(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops128, 64>(t, n, p, m); }
std::pair<bool, size_t> avx128_needle_hammer512(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops128, 512>(t, n, p, m); }
std::pair<bool, size_t> avx128_needle_hammer8192(const char* t, size_t n, const char* p, size_t m)
    { return x86_needle_hammer_t<x86_ops128, 8192>(t, n, p, m); }
