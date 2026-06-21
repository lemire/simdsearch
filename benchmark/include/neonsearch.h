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
    }

    // Scalar tail for positions the 64-byte SIMD loop could not safely cover.
    for (; i + m <= n; ++i) {
        if (std::memcmp(text + i, pattern, m) == 0) {
            return {true, i};
        }
    }

    return {false, 0};
}


// Pick three needle offsets to anchor the SIMD pre-filter on, faithfully
// porting StringZilla's sz_locate_needle_anomalies_. Start with first / middle
// / last; if any of the three bytes collide, walk the middle and last offsets
// inward so the trio stays distinct (a stronger filter). For needles longer
// than 8 bytes, prefer "vibrant" bytes < 191 — values >= 192 are UTF-8
// continuation bytes that recur in text, so anchoring on them is weak.
static inline void sz_locate_needle_anomalies(const char* start, size_t length,
                                              size_t& first, size_t& second,
                                              size_t& third) {
    const unsigned char* s = (const unsigned char*)start;
    first = 0;
    second = length / 2;
    third = length - 1;

    bool has_duplicates = s[first] == s[second] || s[first] == s[third] ||
                          s[second] == s[third];
    if (length > 3 && has_duplicates) {
        while (s[second] == s[first] && second + 1 < third) ++second;
        while ((s[third] == s[second] || s[third] == s[first]) &&
               third > second + 1)
            --third;
    }

    if (length > 8) {
        size_t vfirst = first, vsecond = second, vthird = third;
        while ((s[vsecond] > 191 || s[vsecond] == s[vthird]) &&
               (vsecond + 1 < vthird))
            ++vsecond;
        if (s[vsecond] < 191) second = vsecond;
        else vsecond = second;
        while ((s[vfirst] > 191 || s[vfirst] == s[vsecond] ||
                s[vfirst] == s[vthird]) &&
               (vfirst + 1 < vsecond))
            ++vfirst;
        if (s[vfirst] < 191) first = vfirst;
    }
}

// Returns {found, index} of first occurrence (matches neon_naive_search
// interface). Faithful port of StringZilla's sz_find_neon. Three needle bytes
// (first/middle/last, chosen by sz_locate_needle_anomalies) are broadcast and
// compared against 16-byte haystack windows; ANDing the three masks leaves
// only candidates matching at all three anchors, which a memcmp then verifies
// in full. The 16-lane match vector is narrowed to a 64-bit value (4 bits per
// lane) via shrn #4 so the matches can be iterated with a trailing-zero count,
// exactly as StringZilla does. NEON has no masked loads, so a serial scalar
// tail covers positions the 16-byte window cannot reach.
std::pair<bool, size_t> neon_stringzilla_find(const char* haystack, size_t h_len,
                                              const char* needle, size_t n_len)
{
    if (n_len == 0) return {true, 0};
    if (h_len < n_len) return {false, 0};

    // Single-byte needle: one broadcast compare per 16-byte window.
    if (n_len == 1) {
        uint8x16_t n_vec = vdupq_n_u8((uint8_t)needle[0]);
        size_t i = 0;
        for (; i + 16 <= h_len; i += 16) {
            uint8x16_t eq = vceqq_u8(vld1q_u8((const uint8_t*)(haystack + i)), n_vec);
            if (vmaxvq_u8(eq) != 0) {
                uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(eq), 4);
                uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
                return {true, i + ((size_t)__builtin_ctzll(mask) >> 2)};
            }
        }
        for (; i < h_len; ++i)
            if (haystack[i] == needle[0]) return {true, i};
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
        uint8x16_t matches = vandq_u8(
            vandq_u8(vceqq_u8(hf, vfirst), vceqq_u8(hm, vmid)),
            vceqq_u8(hl, vlast));
        uint8x8_t nm = vshrn_n_u16(vreinterpretq_u16_u8(matches), 4);
        uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(nm), 0);
        while (mask != 0) {
            size_t b = (size_t)__builtin_ctzll(mask) >> 2;  // lane index
            if (std::memcmp(haystack + i + b, needle, n_len) == 0)
                return {true, i + b};
            mask &= ~((uint64_t)0xF << (b * 4));  // clear this lane's nibble
        }
    }

    // Scalar tail (StringZilla falls back to sz_find_serial here).
    for (; i + n_len <= h_len; ++i) {
        if (std::memcmp(haystack + i, needle, n_len) == 0)
            return {true, i};
    }

    return {false, 0};
}