#include <arm_neon.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>


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


// return bool and index of first occurrence
// based on https://onlinelibrary.wiley.com/doi/pdf/10.1002/spe.2511
std::pair<bool, size_t> neon_naive_search(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0 || n < m) return {false, 0};
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
                    if (vmaxvq_u32(found) == 0) break;
                    uint8x16_t tj = vld1q_u8((const uint8_t*)(text + i + j));
                    uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                    found = vandq_u8(found, vceqq_u8(tj, pj));
                }
                if (vmaxvq_u32(found) == 0) continue;

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
                    if (vmaxvq_u32(found) == 0) break;
                    uint8x16_t tj = vld1q_u8((const uint8_t*)(text + i + j));
                    uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                    found = vandq_u8(found, vceqq_u8(tj, pj));
                }
                if (vmaxvq_u32(found) == 0) continue;


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


// Same as neon_naive_search but with a 64-byte stride: each iteration loads
// four 16-byte chunks (A/B/C/D), so each vdupq_n_u8 broadcast of a pattern
// byte is reused across four match accumulators instead of one.
std::pair<bool, size_t> neon_naive_search64(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0 || n < m) return {false, 0};

    size_t i = 0;
    // SIMD reads bytes [i, i + 63 + (m - 1)], so require i + m + 63 <= n.
    if (n >= m + 63) {
      if (m >= 4) {
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
                if (vmaxvq_u32(any) == 0) break;
                uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j +  0)), pj));
                fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 16)), pj));
                fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 32)), pj));
                fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 48)), pj));
            }

            uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
            if (vmaxvq_u32(any) == 0) continue;

            // Walk lanes in order so we return the lowest-index match.
            if (vmaxvq_u32(fA) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fA), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + (__builtin_ctzll(mask) >> 2)};
            }
            if (vmaxvq_u32(fB) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fB), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + 16 + (__builtin_ctzll(mask) >> 2)};
            }
            if (vmaxvq_u32(fC) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fC), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + 32 + (__builtin_ctzll(mask) >> 2)};
            }
            uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fD), 4);
            uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
            return {true, i + 48 + (__builtin_ctzll(mask) >> 2)};
        }
      } else {
        // Short needles (1..3 bytes): keep the 64-byte stride, but build the
        // match mask one byte at a time instead of with the 4-byte prefix, so
        // tiny patterns get SIMD too rather than falling to the scalar loop.
        for (; i + m + 63 <= n; i += 64) {
            uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
            uint8x16_t fA = vceqq_u8(vld1q_u8((const uint8_t*)(text + i +  0)), p0);
            uint8x16_t fB = vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 16)), p0);
            uint8x16_t fC = vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 32)), p0);
            uint8x16_t fD = vceqq_u8(vld1q_u8((const uint8_t*)(text + i + 48)), p0);

            for (size_t j = 1; j < m; ++j) {
                uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
                fA = vandq_u8(fA, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j +  0)), pj));
                fB = vandq_u8(fB, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 16)), pj));
                fC = vandq_u8(fC, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 32)), pj));
                fD = vandq_u8(fD, vceqq_u8(vld1q_u8((const uint8_t*)(text + i + j + 48)), pj));
            }

            uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
            if (vmaxvq_u32(any) == 0) continue;

            if (vmaxvq_u32(fA) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fA), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + (__builtin_ctzll(mask) >> 2)};
            }
            if (vmaxvq_u32(fB) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fB), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + 16 + (__builtin_ctzll(mask) >> 2)};
            }
            if (vmaxvq_u32(fC) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fC), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + 32 + (__builtin_ctzll(mask) >> 2)};
            }
            uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(fD), 4);
            uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
            return {true, i + 48 + (__builtin_ctzll(mask) >> 2)};
        }
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


// movemask-style reduction for NEON: the vshrn-by-4 trick turns each 0xFF lane
// into a nibble, and the 0x8888... mask keeps one set bit per matching lane, so
// __builtin_ctzll(mask) >> 2 yields the lowest matching lane index.
static inline uint64_t sz_neon_mask(uint8x16_t v) {
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0)
           & 0x8888888888888888ull;
}

// Whole-needle equality used to verify candidates, NEON for >= 16 bytes (with an
// overlapping final block), scalar otherwise. Mirrors sz_equal_neon.
static inline bool sz_neon_equal(const char* a, const char* b, size_t len) {
    if (len < 16) return std::memcmp(a, b, len) == 0;
    size_t off = 0;
    for (; off + 16 <= len; off += 16) {
        uint8x16_t av = vld1q_u8((const uint8_t*)(a + off));
        uint8x16_t bv = vld1q_u8((const uint8_t*)(b + off));
        if (vminvq_u8(vceqq_u8(av, bv)) != 255) return false;
    }
    uint8x16_t av = vld1q_u8((const uint8_t*)(a + len - 16));
    uint8x16_t bv = vld1q_u8((const uint8_t*)(b + len - 16));
    return vminvq_u8(vceqq_u8(av, bv)) == 255;
}

// Pick three "anchor" offsets (first / middle / last) used as SIMD pre-filters.
// Defaults are 0, len/2, len-1; on collisions the middle pivots right and the
// last pivots left to find distinct bytes; for len > 8 the first and middle are
// shifted toward high-entropy bytes (value <= 191, i.e. not a UTF-8 multi-byte
// rune prefix). Ported verbatim from StringZilla's sz_locate_needle_anomalies_.
static inline void sz_locate_anchors(const char* start, size_t length,
                                     size_t* first, size_t* second, size_t* third) {
    *first = 0;
    *second = length / 2;
    *third = length - 1;

    int has_duplicates = start[*first] == start[*second] ||
                         start[*first] == start[*third] ||
                         start[*second] == start[*third];

    if (length > 3 && has_duplicates) {
        while (start[*second] == start[*first] && *second + 1 < *third) ++(*second);
        while ((start[*third] == start[*second] || start[*third] == start[*first]) &&
               *third > (*second + 1))
            --(*third);
    }

    if (length > 8) {
        const uint8_t* u = (const uint8_t*)start;
        size_t vf = *first, vs = *second, vt = *third;
        while ((u[vs] > 191 || u[vs] == u[vt]) && (vs + 1 < vt)) ++vs;
        if (u[vs] < 191) { *second = vs; } else { vs = *second; }
        while ((u[vf] > 191 || u[vf] == u[vs] || u[vf] == u[vt]) && (vf + 1 < vs)) ++vf;
        if (u[vf] < 191) { *first = vf; }
    }
}

// Faithful reproduction of StringZilla's NEON substring search (sz_find_neon):
// a 16-position-per-iteration scan that ANDs comparisons of three spread anchor
// bytes into a movemask, then fully verifies each surviving candidate. The
// n_len == 2 and n_len == 3 paths need no verification because their anchors
// already cover every needle byte.
// Returns {found, index} of first occurrence (matches neon_naive_search interface).
std::pair<bool, size_t> neon_stringzilla_find(const char* haystack, size_t h_len,
                                              const char* needle, size_t n_len)
{
    if (n_len == 0) return {true, 0};
    if (h_len < n_len) return {false, 0};

    size_t i = 0;

    // Single-byte needle: one anchor.
    if (n_len == 1) {
        uint8x16_t nv = vdupq_n_u8((uint8_t)needle[0]);
        for (; i + 16 <= h_len; i += 16) {
            uint64_t mask = sz_neon_mask(vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i)), nv));
            if (mask) return {true, i + (__builtin_ctzll(mask) >> 2)};
        }
    } else if (n_len == 2) {
        // First + last anchor cover the whole needle: no verification needed.
        uint8x16_t nf = vdupq_n_u8((uint8_t)needle[0]);
        uint8x16_t nl = vdupq_n_u8((uint8_t)needle[1]);
        for (; i + 17 <= h_len; i += 16) {
            uint8x16_t hf = vld1q_u8((const uint8_t*)(haystack + i + 0));
            uint8x16_t hl = vld1q_u8((const uint8_t*)(haystack + i + 1));
            uint64_t mask = sz_neon_mask(vandq_u8(vceqq_u8(hf, nf), vceqq_u8(hl, nl)));
            if (mask) return {true, i + (__builtin_ctzll(mask) >> 2)};
        }
    } else if (n_len == 3) {
        // The three anchors are the literal three needle bytes: no verification.
        uint8x16_t nf = vdupq_n_u8((uint8_t)needle[0]);
        uint8x16_t nm = vdupq_n_u8((uint8_t)needle[1]);
        uint8x16_t nl = vdupq_n_u8((uint8_t)needle[2]);
        for (; i + 18 <= h_len; i += 16) {
            uint8x16_t hf = vld1q_u8((const uint8_t*)(haystack + i + 0));
            uint8x16_t hm = vld1q_u8((const uint8_t*)(haystack + i + 1));
            uint8x16_t hl = vld1q_u8((const uint8_t*)(haystack + i + 2));
            uint8x16_t mv = vandq_u8(vandq_u8(vceqq_u8(hf, nf), vceqq_u8(hm, nm)),
                                     vceqq_u8(hl, nl));
            uint64_t mask = sz_neon_mask(mv);
            if (mask) return {true, i + (__builtin_ctzll(mask) >> 2)};
        }
    } else {
        // n_len >= 4: three anomaly-picked anchors plus full-needle verification.
        size_t of, om, ol;
        sz_locate_anchors(needle, n_len, &of, &om, &ol);
        uint8x16_t nf = vdupq_n_u8((uint8_t)needle[of]);
        uint8x16_t nm = vdupq_n_u8((uint8_t)needle[om]);
        uint8x16_t nl = vdupq_n_u8((uint8_t)needle[ol]);
        // Anchor load at haystack + i + ol spans 16 lanes; ol <= n_len - 1, so
        // i + n_len + 16 <= h_len keeps every load and verification in bounds.
        for (; i + n_len + 16 <= h_len; i += 16) {
            uint8x16_t hf = vld1q_u8((const uint8_t*)(haystack + i + of));
            uint8x16_t hm = vld1q_u8((const uint8_t*)(haystack + i + om));
            uint8x16_t hl = vld1q_u8((const uint8_t*)(haystack + i + ol));
            uint8x16_t mv = vandq_u8(vandq_u8(vceqq_u8(hf, nf), vceqq_u8(hm, nm)),
                                     vceqq_u8(hl, nl));
            uint64_t mask = sz_neon_mask(mv);
            while (mask) {
                size_t off = __builtin_ctzll(mask) >> 2;
                if (sz_neon_equal(haystack + i + off, needle, n_len))
                    return {true, i + off};
                mask &= mask - 1;  // drop this candidate, try the next lane
            }
        }
    }

    // Scalar tail for positions the SIMD loop could not safely cover.
    for (; i + n_len <= h_len; ++i) {
        if (std::memcmp(haystack + i, needle, n_len) == 0)
            return {true, i};
    }

    return {false, 0};
}