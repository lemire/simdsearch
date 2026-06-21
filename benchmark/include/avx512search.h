#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

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
// only candidates that match at all three anchors, which a memcmp then
// verifies in full. Every load is a predicated _mm512_maskz_loadu_epi8, so a
// single masked loop covers the body, the tail, and haystacks shorter than one
// 64-byte window with no scalar fallback (masked-off lanes are never touched,
// so the loads stay in bounds at the end of the haystack).
std::pair<bool, size_t> avx512_stringzilla_find(const char* haystack, size_t h_len,
                                                const char* needle, size_t n_len)
{
    if (n_len == 0) return {true, 0};
    if (h_len < n_len) return {false, 0};

    // Single-byte needle: one masked broadcast compare per 64-byte window.
    if (n_len == 1) {
        __m512i n_vec = _mm512_set1_epi8((char)needle[0]);
        for (size_t i = 0; i < h_len; i += 64) {
            size_t cand = h_len - i;  // bytes (= candidate positions) remaining
            __mmask64 active = (cand >= 64) ? ~(__mmask64)0
                                            : (((__mmask64)1 << cand) - 1);
            __mmask64 eq = _mm512_mask_cmpeq_epi8_mask(
                active, _mm512_maskz_loadu_epi8(active, haystack + i), n_vec);
            if (eq != 0) return {true, i + (size_t)__builtin_ctzll(eq)};
        }
        return {false, 0};
    }

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies(needle, n_len, off_first, off_mid, off_last);
    __m512i first = _mm512_set1_epi8((char)needle[off_first]);
    __m512i mid = _mm512_set1_epi8((char)needle[off_mid]);
    __m512i last = _mm512_set1_epi8((char)needle[off_last]);

    // Each iteration handles up to 64 candidate start positions [i, i+63].
    // active masks to the candidates that actually exist; off_last <= n_len-1,
    // so the masked-off lanes are exactly the ones that would read past h_len.
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
        while (mask != 0) {
            size_t b = (size_t)__builtin_ctzll(mask);
            if (std::memcmp(haystack + i + b, needle, n_len) == 0)
                return {true, i + b};
            mask &= mask - 1;  // clear the lowest set bit and continue
        }
    }

    return {false, 0};
}
