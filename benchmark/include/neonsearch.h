#pragma once

#ifndef NEON_NH_TAU
#define NEON_NH_TAU 16
#endif
#ifndef NEON_NH_MU
#define NEON_NH_MU 256
#endif
#include <arm_neon.h>
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
std::pair<bool, size_t> neon_naive_search(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const size_t step = 16;

    size_t i = 0;
    // SIMD chunk reads bytes [i, i + 15 + (m - 1)], so require i + m + 15 <= n.
    if (n >= m + 15) {
        if(m >= 4) {
            for (; i + m + 15 <= n; i += step) {
                uint8x16_t t0 = vld1q_u8((const uint8_t*)(text + i));
                uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
                uint8x16_t found = vceqq_u8(t0, p0);
                uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
                found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 1)), p1));
                uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
                found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 2)), p2));
                uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
                found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 3)), p3));
                for (size_t j = 4; j < m; ++j) {
                    if (vmaxvq_u32(vreinterpretq_u32_u8(found)) == 0) break;
                    uint8x16_t tj = vld1q_u8((const uint8_t*)(text + i + j));
                    uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                    found = vandq_u8(found, vceqq_u8(tj, pj));
                }
                if (vmaxvq_u32(vreinterpretq_u32_u8(found)) == 0) continue;

                uint8x8_t nibble_mask = vshrn_n_u16(vreinterpretq_u16_u8(found), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nibble_mask), 0);
                size_t idx = __builtin_ctzll(mask) >> 2;
                return {true, i + idx};
            }

        } else {
            for (; i + m + 15 <= n; i += step) {
                uint8x16_t t0 = vld1q_u8((const uint8_t*)(text + i));
                uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
                uint8x16_t found = vceqq_u8(t0, p0);
                for (size_t j = 1; j < m; ++j) {
                    if (vmaxvq_u32(vreinterpretq_u32_u8(found)) == 0) break;
                    uint8x16_t tj = vld1q_u8((const uint8_t*)(text + i + j));
                    uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                    found = vandq_u8(found, vceqq_u8(tj, pj));
                }
                if (vmaxvq_u32(vreinterpretq_u32_u8(found)) == 0) continue;


                uint8x8_t nibble_mask = vshrn_n_u16(vreinterpretq_u16_u8(found), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nibble_mask), 0);
                size_t idx = __builtin_ctzll(mask) >> 2;
                return {true, i + idx};
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


// Find-all variant of neon_naive_search (NEON parity for avx512_naive_search_all).
// Instead of returning at the first match and discarding the rest of the mask,
// each 16-byte block builds the same per-position match vector and then
// enumerates EVERY match -- calling callback(index) for each occurrence -- via
// the shrn #4 trick (4 bits per lane) before advancing to the next block. This
// pays the pattern-byte broadcasts and the block scan once per block instead of
// re-entering a first-match search once per match. Occurrences are reported in
// increasing index order and include overlapping ones. callback must be a
// callable taking a single size_t index.
template <typename F>
void neon_naive_search_all(const char* text, size_t n, const char* pattern,
                           size_t m, F callback) {
    if (m == 0) { callback(0); return; }
    if (n < m) return;
    const size_t step = 16;

    // Visit every set lane in a match vector, lowest index first. The shrn #4
    // narrowing puts 4 bits per byte lane into a 64-bit value; ctz/4 is the lane.
    auto emit = [&](uint8x16_t found, size_t base) {
        uint8x8_t nibble = vshrn_n_u16(vreinterpretq_u16_u8(found), 4);
        uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nibble), 0);
        while (mask) {
            size_t b = (size_t)__builtin_ctzll(mask) >> 2;  // lane index
            callback(base + b);
            mask &= ~((uint64_t)0xF << (b * 4));  // clear this lane's nibble
        }
    };

    size_t i = 0;
    // SIMD chunk reads bytes [i, i + 15 + (m - 1)], so require i + m + 15 <= n.
    if (n >= m + 15) {
        if (m >= 4) {
            const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
            const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
            const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
            const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
            for (; i + m + 15 <= n; i += step) {
                uint8x16_t found = vceqq_u8(vld1q_u8((const uint8_t*)(text + i)), p0);
                found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 1)), p1));
                found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 2)), p2));
                found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 3)), p3));
                for (size_t j = 4; j < m; ++j) {
                    if (vmaxvq_u8(found) == 0) break;
                    uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                    found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j)), pj));
                }
                if (vmaxvq_u8(found) != 0) emit(found, i);
            }
        } else {
            const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
            for (; i + m + 15 <= n; i += step) {
                uint8x16_t found = vceqq_u8(vld1q_u8((const uint8_t*)(text + i)), p0);
                for (size_t j = 1; j < m; ++j) {
                    if (vmaxvq_u8(found) == 0) break;
                    uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                    found = vandq_u8(found, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j)), pj));
                }
                if (vmaxvq_u8(found) != 0) emit(found, i);
            }
        }
    }

    // Scalar tail for positions the SIMD loop could not safely cover.
    for (; i + m <= n; ++i)
        if (std::memcmp(text + i, pattern, m) == 0) callback(i);
}


// Same as neon_naive_search but with a 64-byte stride: each iteration loads
// four 16-byte chunks (A/B/C/D), so each vdupq_n_u8 broadcast of a pattern
// byte is reused across four match accumulators instead of one.
std::pair<bool, size_t> neon_naive_search64(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    // SIMD reads bytes [i, i + 63 + (m - 1)], so require i + m + 63 <= n.
    if (m >= 4 && n >= m + 63) {
        for (; i + m + 63 <= n; i += 64) {
            uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
            uint8x16_t fA = vceqq_u8(vld1q_u8((const uint8_t*)(text + i +  0)), p0);
            uint8x16_t fB = vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 16)), p0);
            uint8x16_t fC = vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 32)), p0);
            uint8x16_t fD = vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 48)), p0);

            uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
            fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text + i +  1)), p1));
            fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 17)), p1));
            fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 33)), p1));
            fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 49)), p1));

            uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
            fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text + i +  2)), p2));
            fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 18)), p2));
            fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 34)), p2));
            fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 50)), p2));

            uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
            fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text + i +  3)), p3));
            fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 19)), p3));
            fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 35)), p3));
            fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 51)), p3));

            for (size_t j = 4; j < m; ++j) {
                uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
                if (vmaxvq_u32(vreinterpretq_u32_u8(any)) == 0) break;
                uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j +  0)), pj));
                fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 16)), pj));
                fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 32)), pj));
                fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 48)), pj));
            }

            uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
            if (vmaxvq_u32(vreinterpretq_u32_u8(any)) == 0) continue;

            // Walk lanes in order so we return the lowest-index match.
            if (vmaxvq_u32(vreinterpretq_u32_u8(fA)) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fA), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + (__builtin_ctzll(mask) >> 2)};
            }
            if (vmaxvq_u32(vreinterpretq_u32_u8(fB)) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fB), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + 16 + (__builtin_ctzll(mask) >> 2)};
            }
            if (vmaxvq_u32(vreinterpretq_u32_u8(fC)) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fC), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + 32 + (__builtin_ctzll(mask) >> 2)};
            }
            uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fD), 4);
            uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
            return {true, i + 48 + (__builtin_ctzll(mask) >> 2)};
        }
    }

    // Scalar tail for positions the 64-byte SIMD loop could not safely cover.
    for (; i + m <= n; ++i) {
        if (std::memcmp(text + i, pattern, m) == 0) {
            return {true, i};
        }
    }

    return {false, 0};
}


// ---------------------------------------------------------------------------
// NEON Needle-Hammer.
//
// Same three ideas as the AVX-512 v3 kernels, retargeted to a machine with no
// mask registers. The parts that carry over unchanged are the ones about
// dependency structure, not about instruction set: filter on four needle bytes
// with INDEPENDENT compares so the four vceqq chains issue in parallel rather
// than serialising through one accumulator; take the overwhelmingly common
// "exactly one candidate survived the filter" case out of the narrowing loop and
// verify it in place; and leave no bytes to a scalar loop.
//
// What does not carry over is the mask. AVX-512 keeps candidates in a k register
// where "exactly one bit set" is the BLSR idiom on a value it already has. NEON
// keeps them in a uint8x16_t of 0x00/0xFF lanes, so the test needs a movemask
// first. shrn #4 narrows the 16 lanes to 4 bits each in a 64-bit GPR, and
// masking with 0x8888... leaves one bit per lane -- at which point BLSR works as
// it does on x86, and ctz/4 is the lane index. That movemask is not overhead
// charged to the guard: every path that reports a match needs it anyway.
static inline uint64_t neon_lane_mask(uint8x16_t v) {
    return vget_lane_u64(vreinterpret_u64_u8(
               vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0) &
           0x8888888888888888ull;
}

// Verify a candidate whose first four bytes already matched.
static inline bool neon_verify_tail(const char* p, const char* pattern, size_t m) {
    return std::memcmp(p + 4, pattern + 4, m - 4) == 0;
}

// Single-window kernel, 16-byte stride.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
neon_naive_search_v3_body(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    if (m >= 4 && n >= m + 15) {
        const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
        const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
        const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
        const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
        for (; i + m + 15 <= n; i += 16) {
            uint8x16_t f =
                vandq_u8(vandq_u8(vceqq_u8(vld1q_u8((const uint8_t*)(text+i+0)), p0),
                                  vceqq_u8(vld1q_u8((const uint8_t*)(text+i+1)), p1)),
                         vandq_u8(vceqq_u8(vld1q_u8((const uint8_t*)(text+i+2)), p2),
                                  vceqq_u8(vld1q_u8((const uint8_t*)(text+i+3)), p3)));
            uint64_t mask = neon_lane_mask(f);
            if (mask == 0) continue;

            if ((mask & (mask - 1)) == 0) {              // one survivor: verify here
                const size_t b = (size_t)__builtin_ctzll(mask) >> 2;
                if (neon_verify_tail(text + i + b, pattern, m)) return {true, i + b};
                continue;
            }
            for (size_t k = 4; k < m; ++k) {             // many: narrow (adversarial)
                if (vmaxvq_u32(vreinterpretq_u32_u8(f)) == 0) break;
                f = vandq_u8(f, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+k)),
                                         vdupq_n_u8((uint8_t)pattern[k])));
            }
            mask = neon_lane_mask(f);
            if (mask) return {true, i + ((size_t)__builtin_ctzll(mask) >> 2)};
        }

        // Overlap epilogue. NEON has no masked load, so instead of narrowing the
        // window we slide it back to the last legal position and re-scan bytes
        // already scanned. That is sound precisely because they were scanned and
        // held no match, so the lowest surviving lane here cannot precede i --
        // and it costs one window rather than up to 15 scalar memcmp calls.
        // Guarded on the main loop having run: if it did not, i is 0 and sliding
        // back would read out of bounds.
        if (i + m <= n) {
            const size_t last = n - m - 15;
            uint8x16_t f =
                vandq_u8(vandq_u8(vceqq_u8(vld1q_u8((const uint8_t*)(text+last+0)), p0),
                                  vceqq_u8(vld1q_u8((const uint8_t*)(text+last+1)), p1)),
                         vandq_u8(vceqq_u8(vld1q_u8((const uint8_t*)(text+last+2)), p2),
                                  vceqq_u8(vld1q_u8((const uint8_t*)(text+last+3)), p3)));
            for (size_t k = 4; k < m; ++k) {
                if (vmaxvq_u32(vreinterpretq_u32_u8(f)) == 0) break;
                f = vandq_u8(f, vceqq_u8(vld1q_u8((const uint8_t*)(text+last+k)),
                                         vdupq_n_u8((uint8_t)pattern[k])));
            }
            uint64_t mask = neon_lane_mask(f);
            while (mask) {
                const size_t b = last + ((size_t)__builtin_ctzll(mask) >> 2);
                if (b >= i) return {true, b};
                mask &= mask - 1;
            }
            return {false, 0};
        }
        return {false, 0};
    }

    // m < 4. The filter IS the whole needle here, so there is nothing left to
    // verify and no narrowing loop to enter; without this branch the kernel drops
    // to a scalar memcmp per position, which measured 76x slower than the
    // anchored kernel at m = 3 on a find-first workload.
    if (n >= m + 15) {
        const uint8x16_t q0 = vdupq_n_u8((uint8_t)pattern[0]);
        const uint8x16_t q1 = vdupq_n_u8((uint8_t)pattern[m > 1 ? 1 : 0]);
        const uint8x16_t q2 = vdupq_n_u8((uint8_t)pattern[m > 2 ? 2 : 0]);
        for (; i + m + 15 <= n; i += 16) {
            uint8x16_t f = vceqq_u8(vld1q_u8((const uint8_t*)(text + i)), q0);
            if (m > 1) f = vandq_u8(f, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+1)), q1));
            if (m > 2) f = vandq_u8(f, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+2)), q2));
            const uint64_t mask = neon_lane_mask(f);
            if (mask) return {true, i + ((size_t)__builtin_ctzll(mask) >> 2)};
        }
        if (i + m <= n) {
            const size_t last = n - m - 15;
            uint8x16_t f = vceqq_u8(vld1q_u8((const uint8_t*)(text + last)), q0);
            if (m > 1) f = vandq_u8(f, vceqq_u8(vld1q_u8((const uint8_t*)(text+last+1)), q1));
            if (m > 2) f = vandq_u8(f, vceqq_u8(vld1q_u8((const uint8_t*)(text+last+2)), q2));
            uint64_t mask = neon_lane_mask(f);
            while (mask) {
                const size_t b = last + ((size_t)__builtin_ctzll(mask) >> 2);
                if (b >= i) return {true, b};
                mask &= mask - 1;
            }
        }
        return {false, 0};
    }

    for (; i + m <= n; ++i)
        if (std::memcmp(text + i, pattern, m) == 0) return {true, i};
    return {false, 0};
}

// Wide-stride kernel, 64 bytes per iteration as four 16-byte chunks. As on
// AVX-512 the single-survivor guard is on the WHOLE block, not per chunk:
// resolving chunks independently duplicates the pattern broadcasts that the
// shared narrowing loop exists to amortise.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
neon_naive_search64_v3_body(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    if (m >= 4 && n >= m + 63) {
        const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
        const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
        const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
        const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
#define NEON_CHUNK(OFF)                                                            \
        vandq_u8(vandq_u8(vceqq_u8(vld1q_u8((const uint8_t*)(text+i+(OFF)+0)), p0), \
                          vceqq_u8(vld1q_u8((const uint8_t*)(text+i+(OFF)+1)), p1)),\
                 vandq_u8(vceqq_u8(vld1q_u8((const uint8_t*)(text+i+(OFF)+2)), p2), \
                          vceqq_u8(vld1q_u8((const uint8_t*)(text+i+(OFF)+3)), p3)))
        for (; i + m + 63 <= n; i += 64) {
            uint8x16_t fA = NEON_CHUNK(0),  fB = NEON_CHUNK(16);
            uint8x16_t fC = NEON_CHUNK(32), fD = NEON_CHUNK(48);
            uint64_t mA = neon_lane_mask(fA), mB = neon_lane_mask(fB);
            uint64_t mC = neon_lane_mask(fC), mD = neon_lane_mask(fD);
            if ((mA | mB | mC | mD) == 0) continue;

            const int nz = (mA != 0) + (mB != 0) + (mC != 0) + (mD != 0);
            const uint64_t one = mA | mB | mC | mD;
            if (nz == 1 && (one & (one - 1)) == 0) {     // one survivor in the block
                const size_t off = (mA != 0) ? 0 : (mB != 0) ? 16 : (mC != 0) ? 32 : 48;
                const size_t b = off + ((size_t)__builtin_ctzll(one) >> 2);
                if (neon_verify_tail(text + i + b, pattern, m)) return {true, i + b};
                continue;
            }
            for (size_t k = 4; k < m; ++k) {             // many: shared narrowing
                uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
                if (vmaxvq_u32(vreinterpretq_u32_u8(any)) == 0) break;
                const uint8x16_t pk = vdupq_n_u8((uint8_t)pattern[k]);
                fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+k+ 0)), pk));
                fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+k+16)), pk));
                fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+k+32)), pk));
                fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text+i+k+48)), pk));
            }
            if ((mA = neon_lane_mask(fA)) != 0) return {true, i +  0 + ((size_t)__builtin_ctzll(mA) >> 2)};
            if ((mB = neon_lane_mask(fB)) != 0) return {true, i + 16 + ((size_t)__builtin_ctzll(mB) >> 2)};
            if ((mC = neon_lane_mask(fC)) != 0) return {true, i + 32 + ((size_t)__builtin_ctzll(mC) >> 2)};
            if ((mD = neon_lane_mask(fD)) != 0) return {true, i + 48 + ((size_t)__builtin_ctzll(mD) >> 2)};
        }
#undef NEON_CHUNK
    }
    // Remainder: hand to the 16-byte kernel, which ends in an overlap window, so
    // no byte reaches a scalar loop.
    if (i + m <= n) {
        auto r = neon_naive_search_v3_body(text + i, n - i, pattern, m);
        if (r.first) return {true, i + r.second};
    }
    return {false, 0};
}

std::pair<bool, size_t> neon_naive_search_v3(const char* text, size_t n,
                                             const char* pattern, size_t m) {
    return neon_naive_search_v3_body(text, n, pattern, m);
}
std::pair<bool, size_t> neon_naive_search64_v3(const char* text, size_t n,
                                               const char* pattern, size_t m) {
    return neon_naive_search64_v3_body(text, n, pattern, m);
}

// Port of StringZilla's sz_find_vreinterpretq_u8_u4_: shrn #4 movemask, keeping
// one bit per lane so ctz/4 indexes candidates and "mask &= mask - 1" advances.
static inline uint64_t neon_match_mask_from_eq(uint8x16_t vec) {
    return vget_lane_u64(
               vreinterpret_u64_u8(
                   vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)),
               0) &
           0x8888888888888888ull;
}

static inline size_t neon_first_match_lane(uint64_t mask) {
    return (size_t)__builtin_ctzll(mask) / 4;
}

// Returns {found, index} of first occurrence (matches neon_naive_search
// interface). Faithful port of StringZilla's sz_find_neon. n_len == 2 and
// n_len == 3 use dedicated SIMD paths (as upstream does); longer needles pick
// three anchor bytes via sz_locate_needle_anomalies, AND three compare masks,
// verify survivors with memcmp. NEON has no masked loads, so a scalar tail
// covers positions the 16-byte window cannot reach.
std::pair<bool, size_t> neon_stringzilla_find(const char* haystack, size_t h_len,
                                              const char* needle, size_t n_len)
{
    if (n_len == 0) return {true, 0};
    if (h_len < n_len) return {false, 0};

    if (n_len == 1) {
        uint8x16_t n_vec = vdupq_n_u8((uint8_t)needle[0]);
        size_t i = 0;
        for (; i + 16 <= h_len; i += 16) {
            uint64_t matches = neon_match_mask_from_eq(
                vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i)), n_vec));
            if (matches) return {true, i + neon_first_match_lane(matches)};
        }
        for (; i < h_len; ++i)
            if (haystack[i] == needle[0]) return {true, i};
        return {false, 0};
    }

    if (n_len == 2) {
        uint8x16_t n0 = vdupq_n_u8((uint8_t)needle[0]);
        uint8x16_t n1 = vdupq_n_u8((uint8_t)needle[1]);
        size_t i = 0;
        for (; i + 17 <= h_len; i += 16) {
            uint8x16_t matches_vec = vandq_u8(
                vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i + 0)), n0),
                vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i + 1)), n1));
            uint64_t matches = neon_match_mask_from_eq(matches_vec);
            if (matches) return {true, i + neon_first_match_lane(matches)};
        }
        for (; i + n_len <= h_len; ++i) {
            if (std::memcmp(haystack + i, needle, n_len) == 0)
                return {true, i};
        }
        return {false, 0};
    }

    if (n_len == 3) {
        uint8x16_t n0 = vdupq_n_u8((uint8_t)needle[0]);
        uint8x16_t n1 = vdupq_n_u8((uint8_t)needle[1]);
        uint8x16_t n2 = vdupq_n_u8((uint8_t)needle[2]);
        size_t i = 0;
        for (; i + 18 <= h_len; i += 16) {
            uint8x16_t matches_vec = vandq_u8(
                vandq_u8(
                    vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i + 0)), n0),
                    vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i + 1)), n1)),
                vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i + 2)), n2));
            uint64_t matches = neon_match_mask_from_eq(matches_vec);
            if (matches) return {true, i + neon_first_match_lane(matches)};
        }
        for (; i + n_len <= h_len; ++i) {
            if (std::memcmp(haystack + i, needle, n_len) == 0)
                return {true, i};
        }
        return {false, 0};
    }

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies(needle, n_len, off_first, off_mid, off_last);
    uint8x16_t vfirst = vdupq_n_u8((uint8_t)needle[off_first]);
    uint8x16_t vmid = vdupq_n_u8((uint8_t)needle[off_mid]);
    uint8x16_t vlast = vdupq_n_u8((uint8_t)needle[off_last]);

    size_t i = 0;
    // Loads reach haystack[i + off_last + 15]; off_last <= n_len-1, so the
    // window is in bounds while i + n_len + 15 <= h_len.
    for (; i + n_len + 15 <= h_len; i += 16) {
        uint8x16_t hf = vld1q_u8((const uint8_t*)(haystack + i + off_first));
        uint8x16_t hm = vld1q_u8((const uint8_t*)(haystack + i + off_mid));
        uint8x16_t hl = vld1q_u8((const uint8_t*)(haystack + i + off_last));
        uint8x16_t matches_vec = vandq_u8(
            vandq_u8(vceqq_u8(hf, vfirst), vceqq_u8(hm, vmid)),
            vceqq_u8(hl, vlast));
        uint64_t matches = neon_match_mask_from_eq(matches_vec);
        while (matches) {
            size_t b = neon_first_match_lane(matches);
            if (std::memcmp(haystack + i + b, needle, n_len) == 0)
                return {true, i + b};
            matches &= matches - 1;
        }
    }

    // Scalar tail (StringZilla falls back to sz_find_serial here).
    for (; i + n_len <= h_len; ++i) {
        if (std::memcmp(haystack + i, needle, n_len) == 0)
            return {true, i};
    }

    return {false, 0};
}


// ---------------------------------------------------------------------------
// The scheme: dispatch on haystack size, then on needle length.
//
//   n < MinWide      the wide kernel's 64-byte stride cannot amortise on a small
//                    haystack -- it falls through to the 16-byte kernel below
//                    m + 63 anyway, so below MinWide we go there directly.
//   m <= Threshold   wide kernel. Filtering on four needle bytes rejects almost
//                    every position on real text, and the wide stride pays the
//                    four broadcasts once per 64 bytes instead of once per 16.
//   m >  Threshold   anchored kernel. Its three anchors are chosen for rarity,
//                    so its filter keeps working where a fixed first-four filter
//                    starts admitting candidates; and its cost per position does
//                    not grow with m, while the wide kernel's verification does.
template <size_t Threshold, size_t MinWide>
static inline std::pair<bool, size_t>
neon_needle_hammer_t(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    if (n < MinWide) return neon_naive_search_v3_body(text, n, pattern, m);
    if (m <= Threshold) return neon_naive_search64_v3_body(text, n, pattern, m);
    return neon_stringzilla_find(text, n, pattern, m);
}

std::pair<bool, size_t> neon_needle_hammer(const char* text, size_t n,
                                           const char* pattern, size_t m) {
    return neon_needle_hammer_t<NEON_NH_TAU, NEON_NH_MU>(text, n, pattern, m);
}

// Sweep instances, used only to fix the two constants above.
#define NEON_NH_SWEEP(T, U)                                                        \
    std::pair<bool, size_t> neon_needle_hammer_t##T##_u##U(                        \
        const char* text, size_t n, const char* pattern, size_t m) {               \
        return neon_needle_hammer_t<T, U>(text, n, pattern, m);                    \
    }
NEON_NH_SWEEP(16, 0)   NEON_NH_SWEEP(32, 0)   NEON_NH_SWEEP(64, 0)
NEON_NH_SWEEP(128, 0)  NEON_NH_SWEEP(256, 0)  NEON_NH_SWEEP(512, 0)
NEON_NH_SWEEP(1024, 0)

