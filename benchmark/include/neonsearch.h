#pragma once
// Needle-Hammer on AArch64 NEON.
//
// Function for function this mirrors avx512search.h: a single-window naive
// kernel, a wide-stride naive kernel built from four of those windows, the
// three-anchor (StringZilla) kernel, the length-dispatched scheme that combines
// them, and a work-counting guarded variant of the same scheme. Reading the two
// headers side by side is meant to be possible.
//
// Three things do not carry over from AVX-512, and all three bear on where the
// scheme's two constants land:
//
//   register width   NEON is 128-bit, so one window covers 16 candidate
//                    positions rather than 64 and a wide block covers 64 rather
//                    than 256. Every per-window and per-block fixed cost is
//                    therefore spread over a quarter as many positions.
//
//   no mask registers  AVX-512 keeps the candidate set in a k register, where
//                    "compare only where a candidate is still alive, then AND"
//                    is one instruction and "exactly one bit set" is the BLSR
//                    idiom on a value it already has. NEON keeps the set in a
//                    uint8x16_t of 0x00/0xFF lanes: the AND is a separate
//                    instruction, and any test that needs a bit index first
//                    needs a movemask (see neon_lane_mask) whose transfer to a
//                    general register has double-digit latency. Every loop here
//                    therefore keeps the candidate set in a vector register and
//                    converts only when it must.
//
//   no masked loads  AVX-512's anchored kernel covers its tail with predicated
//                    loads and needs no scalar fallback. NEON has no
//                    byte-granular masked load, so instead of narrowing the
//                    last window we slide it back to the last position where
//                    every load is in bounds and re-scan bytes already scanned
//                    -- sound precisely because they were scanned and held no
//                    match, so any lane below the resume point can be skipped.
//                    That costs one window rather than up to fifteen scalar
//                    memcmp calls.
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

#include "common_search.h"

// Window width and wide-block width, named so the dispatch below reads the same
// as the AVX-512 one (where they are 64 and 256).
static constexpr size_t kNeonW = 16;
static constexpr size_t kNeonB = 4 * kNeonW;   // 64

// Movemask. shrn #4 narrows the eight 16-bit halves of the compare result to
// four bits each in one 64-bit general register; ANDing with 0x8888... leaves
// exactly one bit per byte lane, at bit 4k+3 for lane k. At that point
// __builtin_ctzll(mask) >> 2 is the lane index and mask &= mask - 1 clears it,
// exactly as on a k register.
static inline uint64_t neon_lane_mask(uint8x16_t v) {
    return vget_lane_u64(vreinterpret_u64_u8(
               vshrn_n_u16(vreinterpretq_u16_u8(v), 4)), 0) &
           0x8888888888888888ull;
}

// The lane bits that a mask of `count` live candidates would set, so a partial
// window can be tested without a second movemask.
static inline uint64_t neon_lane_bits(size_t count) {
    return 0x8888888888888888ull &
           ((count >= 16) ? ~(uint64_t)0 : (((uint64_t)1 << (4 * count)) - 1));
}

// "Is any lane still alive", without the general-register round trip above.
// Used by the narrowing loops, which need the answer but not the index.
static inline bool neon_any(uint8x16_t v) {
    return vmaxvq_u32(vreinterpretq_u32_u8(v)) != 0;
}

static inline uint8x16_t neon_load(const char* p) {
    return vld1q_u8((const uint8_t*)p);
}

// Vectorised equality, the NEON counterpart of sz_equal_avx512. Used to confirm
// a surviving candidate when the needle is too long to sit in one register.
// Deliberately not std::memcmp: the library call costs about ten cycles of call
// and setup per candidate, and on an input where the filter stops discriminating
// that fixed cost is paid once per haystack position.
static inline bool neon_equal(const char* a, const char* b, size_t length) {
    while (length >= 16) {
        if (vminvq_u8(vceqq_u8(neon_load(a), neon_load(b))) != 0xFF) return false;
        a += 16; b += 16; length -= 16;
    }
    while (length) {
        if (*a != *b) return false;
        ++a; ++b; --length;
    }
    return true;
}

// A needle of at most 16 bytes staged once into a register, so verifying a
// candidate is one load, one compare and one mask test rather than a call into
// memcmp. NEON has no byte-granular masked load, so the needle is zero-padded
// through a stack buffer here instead of being loaded under a mask on every
// candidate.
struct neon_needle_reg {
    uint8x16_t vec;
    uint64_t bits;   // the lane bits that must all survive
};

static inline neon_needle_reg neon_load_needle(const char* p, size_t m) {
    alignas(16) uint8_t buf[16] = {};
    std::memcpy(buf, p, m);          // caller guarantees m <= 16
    return {vld1q_u8(buf), neon_lane_bits(m)};
}

// Reads 16 bytes at p, so the caller must have checked p + 16 against the end
// of the haystack.
static inline bool neon_needle_eq(const char* p, const neon_needle_reg& nr) {
    return (neon_lane_mask(vceqq_u8(neon_load(p), nr.vec)) & nr.bits) == nr.bits;
}


// ===========================================================================
// Naive prefix-filter kernels
// ===========================================================================

// Single-window kernel, 16 candidate positions per iteration. The same three
// ideas as the AVX-512 single-window kernel:
//
//   independent compares  the four peeled compares do not chain through one
//                         register, so they issue in parallel.
//   single-survivor guard when exactly one lane survives -- the usual case at
//                         the matching window -- verify it directly instead of
//                         narrowing. When MANY survive, which is what an
//                         adversary arranges, fall back to narrowing.
//   inline verification   the needle is preloaded once for m <= 16, so a
//                         candidate costs one load and one compare rather than
//                         a call into memcmp.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
neon_naive_search_body(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    // Fewer than sixteen candidate positions: no window fits, and the scalar
    // loop runs at most fifteen times, a per-search constant rather than
    // anything that grows with n.
    if (n < m + kNeonW - 1) {
        for (size_t i = 0; i + m <= n; ++i)
            if (std::memcmp(text + i, pattern, m) == 0) return {true, i};
        return {false, 0};
    }

    // The last window start at which every load is in bounds.
    const size_t last = n - m - (kNeonW - 1);
    size_t i = 0;

    if (m >= 4) {
        const bool fits = (m <= kNeonW);
        const neon_needle_reg nr =
            fits ? neon_load_needle(pattern, m) : neon_needle_reg{vdupq_n_u8(0), 0};
        // Verify a candidate whose first four bytes already matched. The
        // register path needs sixteen readable bytes; within the last register
        // width of the haystack it falls back to comparing the tail.
        auto verify = [&](size_t pos) -> bool {
            if (fits && pos + kNeonW <= n) return neon_needle_eq(text + pos, nr);
            return std::memcmp(text + pos + 4, pattern + 4, m - 4) == 0;
        };

        const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
        const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
        const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
        const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
#define NEON_FILTER4(BASE)                                                     \
        vandq_u8(vandq_u8(vceqq_u8(neon_load((BASE) + 0), p0),                 \
                          vceqq_u8(neon_load((BASE) + 1), p1)),                \
                 vandq_u8(vceqq_u8(neon_load((BASE) + 2), p2),                 \
                          vceqq_u8(neon_load((BASE) + 3), p3)))

        for (; i <= last; i += kNeonW) {
            uint8x16_t f = NEON_FILTER4(text + i);
            uint64_t mask = neon_lane_mask(f);
            if (mask == 0) continue;

            if ((mask & (mask - 1)) == 0) {              // one survivor
                const size_t b = i + ((size_t)__builtin_ctzll(mask) >> 2);
                if (verify(b)) return {true, b};
                continue;
            }
            for (size_t k = 4; k < m; ++k) {             // many: narrow
                if (!neon_any(f)) break;
                f = vandq_u8(f, vceqq_u8(neon_load(text + i + k),
                                         vdupq_n_u8((uint8_t)pattern[k])));
            }
            mask = neon_lane_mask(f);
            if (mask) return {true, i + ((size_t)__builtin_ctzll(mask) >> 2)};
        }

        // Overlap epilogue, in place of the masked window AVX-512 has. Narrow
        // on every needle byte so a survivor is a confirmed match, then report
        // the lowest one that the main loop had not already ruled out.
        if (i + m <= n) {
            uint8x16_t f = NEON_FILTER4(text + last);
            for (size_t k = 4; k < m; ++k) {
                if (!neon_any(f)) break;
                f = vandq_u8(f, vceqq_u8(neon_load(text + last + k),
                                         vdupq_n_u8((uint8_t)pattern[k])));
            }
            uint64_t mask = neon_lane_mask(f);
            while (mask) {
                const size_t b = last + ((size_t)__builtin_ctzll(mask) >> 2);
                if (b >= i) return {true, b};
                mask &= mask - 1;
            }
        }
#undef NEON_FILTER4
        return {false, 0};
    }

    // m < 4. The filter IS the whole needle here, so there is nothing left to
    // verify and no narrowing loop to enter. Without this branch a sub-4-byte
    // needle drops to one memcmp call per haystack position, which measured
    // dozens of times slower than the anchored kernel at m = 3.
    {
        const uint8x16_t q0 = vdupq_n_u8((uint8_t)pattern[0]);
        const uint8x16_t q1 = vdupq_n_u8((uint8_t)pattern[m > 1 ? 1 : 0]);
        const uint8x16_t q2 = vdupq_n_u8((uint8_t)pattern[m > 2 ? 2 : 0]);
        auto filter3 = [&](const char* base) {
            uint8x16_t f = vceqq_u8(neon_load(base + 0), q0);
            if (m > 1) f = vandq_u8(f, vceqq_u8(neon_load(base + 1), q1));
            if (m > 2) f = vandq_u8(f, vceqq_u8(neon_load(base + 2), q2));
            return f;
        };
        for (; i <= last; i += kNeonW) {
            const uint64_t mask = neon_lane_mask(filter3(text + i));
            if (mask) return {true, i + ((size_t)__builtin_ctzll(mask) >> 2)};
        }
        if (i + m <= n) {
            uint64_t mask = neon_lane_mask(filter3(text + last));
            while (mask) {
                const size_t b = last + ((size_t)__builtin_ctzll(mask) >> 2);
                if (b >= i) return {true, b};
                mask &= mask - 1;
            }
        }
    }
    return {false, 0};
}

std::pair<bool, size_t> neon_naive_search(const char* text, size_t n,
                                          const char* pattern, size_t m) {
    return neon_naive_search_body(text, n, pattern, m);
}


// Wide-stride kernel, 64 candidate positions per iteration as four 16-byte
// chunks. Same three ideas as the single-window kernel; the single-survivor
// guard is on the WHOLE block, not per chunk, because resolving chunks
// independently duplicates the pattern broadcasts that the shared narrowing
// loop exists to amortise.
static inline __attribute__((always_inline)) std::pair<bool, size_t>
neon_naive_search64_body(const char* text, size_t n, const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};

    size_t i = 0;
    if (m >= 4 && n >= m + kNeonB - 1) {
        const bool fits = (m <= kNeonW);
        const neon_needle_reg nr =
            fits ? neon_load_needle(pattern, m) : neon_needle_reg{vdupq_n_u8(0), 0};
        auto verify = [&](size_t pos) -> bool {
            if (fits && pos + kNeonW <= n) return neon_needle_eq(text + pos, nr);
            return std::memcmp(text + pos + 4, pattern + 4, m - 4) == 0;
        };

        const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
        const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
        const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
        const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
#define NEON_CHUNK(OFF)                                                        \
        vandq_u8(vandq_u8(vceqq_u8(neon_load(text + i + (OFF) + 0), p0),       \
                          vceqq_u8(neon_load(text + i + (OFF) + 1), p1)),      \
                 vandq_u8(vceqq_u8(neon_load(text + i + (OFF) + 2), p2),       \
                          vceqq_u8(neon_load(text + i + (OFF) + 3), p3)))
        for (; i + m + kNeonB - 1 <= n; i += kNeonB) {
            uint8x16_t fA = NEON_CHUNK(0),  fB = NEON_CHUNK(16);
            uint8x16_t fC = NEON_CHUNK(32), fD = NEON_CHUNK(48);
            uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
            if (!neon_any(any)) continue;

            const uint64_t mA = neon_lane_mask(fA), mB = neon_lane_mask(fB);
            const uint64_t mC = neon_lane_mask(fC), mD = neon_lane_mask(fD);
            const int nz = (mA != 0) + (mB != 0) + (mC != 0) + (mD != 0);
            const uint64_t one = mA | mB | mC | mD;
            if (nz == 1 && (one & (one - 1)) == 0) {     // one survivor in the block
                const size_t off = (mA != 0) ? 0 : (mB != 0) ? 16 : (mC != 0) ? 32 : 48;
                const size_t b = i + off + ((size_t)__builtin_ctzll(one) >> 2);
                if (verify(b)) return {true, b};
                continue;
            }
            for (size_t k = 4; k < m; ++k) {             // many: shared narrowing
                any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
                if (!neon_any(any)) break;
                const uint8x16_t pk = vdupq_n_u8((uint8_t)pattern[k]);
                fA = vandq_u8(fA, vceqq_u8(neon_load(text + i + k +  0), pk));
                fB = vandq_u8(fB, vceqq_u8(neon_load(text + i + k + 16), pk));
                fC = vandq_u8(fC, vceqq_u8(neon_load(text + i + k + 32), pk));
                fD = vandq_u8(fD, vceqq_u8(neon_load(text + i + k + 48), pk));
            }
            uint64_t r;
            if ((r = neon_lane_mask(fA))) return {true, i +  0 + ((size_t)__builtin_ctzll(r) >> 2)};
            if ((r = neon_lane_mask(fB))) return {true, i + 16 + ((size_t)__builtin_ctzll(r) >> 2)};
            if ((r = neon_lane_mask(fC))) return {true, i + 32 + ((size_t)__builtin_ctzll(r) >> 2)};
            if ((r = neon_lane_mask(fD))) return {true, i + 48 + ((size_t)__builtin_ctzll(r) >> 2)};
        }
#undef NEON_CHUNK
    }
    // Remainder: hand to the 16-byte kernel, which itself ends in an overlap
    // window, so no byte is left to a scalar loop unless the haystack is too
    // short for even one window.
    if (i + m <= n) {
        auto r = neon_naive_search_body(text + i, n - i, pattern, m);
        if (r.first) return {true, i + r.second};
    }
    return {false, 0};
}

std::pair<bool, size_t> neon_naive_search64(const char* text, size_t n,
                                            const char* pattern, size_t m) {
    return neon_naive_search64_body(text, n, pattern, m);
}


// Find-all variant of neon_naive_search: instead of returning at the first
// match, each 16-byte window enumerates EVERY surviving lane before advancing.
// Occurrences are reported in increasing index order and include overlapping
// ones. callback must be a callable taking a single size_t index.
template <typename F>
void neon_naive_search_all(const char* text, size_t n, const char* pattern,
                           size_t m, F callback) {
    // Empty needle: match at every index in [0, n], matching the first-match
    // loop baseline (returns {true, 0} then advances one byte).
    if (m == 0) {
        for (size_t i = 0; i <= n; ++i) callback(i);
        return;
    }
    if (n < m) return;

    size_t i = 0;
    if (n >= m + kNeonW - 1) {
        if (m >= 4) {
            const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
            const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
            const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
            const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
            for (; i + m + kNeonW - 1 <= n; i += kNeonW) {
                uint8x16_t f =
                    vandq_u8(vandq_u8(vceqq_u8(neon_load(text + i + 0), p0),
                                      vceqq_u8(neon_load(text + i + 1), p1)),
                             vandq_u8(vceqq_u8(neon_load(text + i + 2), p2),
                                      vceqq_u8(neon_load(text + i + 3), p3)));
                for (size_t j = 4; j < m; ++j) {
                    if (!neon_any(f)) break;
                    f = vandq_u8(f, vceqq_u8(neon_load(text + i + j),
                                             vdupq_n_u8((uint8_t)pattern[j])));
                }
                uint64_t mask = neon_lane_mask(f);
                while (mask) {
                    callback(i + ((size_t)__builtin_ctzll(mask) >> 2));
                    mask &= mask - 1;
                }
            }
        } else {
            const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
            for (; i + m + kNeonW - 1 <= n; i += kNeonW) {
                uint8x16_t f = vceqq_u8(neon_load(text + i), p0);
                for (size_t j = 1; j < m; ++j) {
                    if (!neon_any(f)) break;
                    f = vandq_u8(f, vceqq_u8(neon_load(text + i + j),
                                             vdupq_n_u8((uint8_t)pattern[j])));
                }
                uint64_t mask = neon_lane_mask(f);
                while (mask) {
                    callback(i + ((size_t)__builtin_ctzll(mask) >> 2));
                    mask &= mask - 1;
                }
            }
        }
    }

    // Scalar tail for the positions the vector loop could not safely cover.
    for (; i + m <= n; ++i)
        if (std::memcmp(text + i, pattern, m) == 0) callback(i);
}


// ===========================================================================
// The three-anchor (StringZilla) kernel
// ===========================================================================
//
// Port of sz_find_neon. Three needle bytes (first/middle/last, chosen by
// sz_locate_needle_anomalies away from repeats) are broadcast and compared
// against the haystack; ANDing the three results leaves only candidates that
// match at all three anchors, which a specialised compare then verifies in
// full. The filter is three compares per window regardless of the needle
// length, which is what makes it the better half of the scheme for long
// needles.
//
// The kernel is written once, with a verification budget in bytes, and both
// entry points below call it -- so they run the same instructions and differ
// only in the budget they pass, which is what makes the guarded and unguarded
// searchers comparable. Unguarded callers pass SIZE_MAX; the counter cannot
// reach it.
//
// Verification follows the same three length-specialised paths as the AVX-512
// kernel, for the same reason: a per-candidate std::memcmp call costs about ten
// cycles, and on an input that defeats the filter every haystack position
// becomes a candidate, so that constant is multiplied by n.
//
//   m <= 3    no verification at all -- the anchors 0, m/2 and m-1 already
//             cover every byte of the needle.
//   m <= 16   the whole needle sits in one register, staged once outside the
//             loop; a candidate costs one load, one compare and a mask test.
//   m > 16    neon_equal, 16 bytes per iteration.
template <bool FilterHighBytes = false>
static inline simd_guarded_result neon_stringzilla_body(
    const char* haystack, size_t h_len, const char* needle, size_t n_len,
    size_t budget_bytes) {
    if (n_len == 0) return {true, 0, false, 0};
    if (h_len < n_len) return {false, 0, false, 0};

    // Single-byte needle: one broadcast compare per window. It verifies
    // nothing, so it cannot exhaust a budget and needs no guard.
    if (n_len == 1) {
        const uint8x16_t nv = vdupq_n_u8((uint8_t)needle[0]);
        size_t i = 0;
        for (; i + kNeonW <= h_len; i += kNeonW) {
            const uint64_t eq = neon_lane_mask(vceqq_u8(neon_load(haystack + i), nv));
            if (eq) return {true, i + ((size_t)__builtin_ctzll(eq) >> 2), false, 0};
        }
        for (; i < h_len; ++i)
            if (haystack[i] == needle[0]) return {true, i, false, 0};
        return {false, 0, false, 0};
    }

    // Too short for one window: at most fifteen positions, a per-search
    // constant.
    if (h_len < n_len + kNeonW - 1) {
        for (size_t i = 0; i + n_len <= h_len; ++i)
            if (neon_equal(haystack + i, needle, n_len)) return {true, i, false, 0};
        return {false, 0, false, 0};
    }

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies_t<FilterHighBytes>(needle, n_len, off_first,
                                                  off_mid, off_last);
    const uint8x16_t vfirst = vdupq_n_u8((uint8_t)needle[off_first]);
    const uint8x16_t vmid = vdupq_n_u8((uint8_t)needle[off_mid]);
    const uint8x16_t vlast = vdupq_n_u8((uint8_t)needle[off_last]);

    const bool anchors_cover_needle = (n_len <= 3);
    const bool needle_fits_register = (n_len <= kNeonW);
    const neon_needle_reg nr = (needle_fits_register && !anchors_cover_needle)
                                   ? neon_load_needle(needle, n_len)
                                   : neon_needle_reg{vdupq_n_u8(0), 0};
    auto equal_at = [&](size_t pos) -> bool {
        if (needle_fits_register && pos + kNeonW <= h_len)
            return neon_needle_eq(haystack + pos, nr);
        return neon_equal(haystack + pos, needle, n_len);
    };

    // The last window start at which the off_last load is in bounds:
    // off_last <= n_len - 1, so the highest byte read is haystack[i + n_len + 14].
    const size_t last = h_len - n_len - (kNeonW - 1);
    size_t verified = 0;
    size_t i = 0;

#define NEON_ANCHORS(BASE)                                                     \
        vandq_u8(vandq_u8(vceqq_u8(neon_load((BASE) + off_first), vfirst),     \
                          vceqq_u8(neon_load((BASE) + off_mid), vmid)),        \
                 vceqq_u8(neon_load((BASE) + off_last), vlast))

    for (; i <= last; i += kNeonW) {
        uint64_t mask = neon_lane_mask(NEON_ANCHORS(haystack + i));
        if (mask == 0) continue;

        // The anchors already tested every byte; no verification to do.
        if (anchors_cover_needle)
            return {true, i + ((size_t)__builtin_ctzll(mask) >> 2), false, 0};

        while (mask) {
            const size_t b = i + ((size_t)__builtin_ctzll(mask) >> 2);
            verified += n_len;
            if (equal_at(b)) return {true, b, false, 0};
            // Budget tested after the compare: the candidate is already paid
            // for, so a match is reported rather than thrown away. Overshoot is
            // one verification.
            if (verified > budget_bytes) return {false, 0, true, i};
            mask &= mask - 1;
        }
    }

    // Overlap epilogue, in place of AVX-512's masked tail window. Lanes below
    // the resume point were already ruled out by the loop above, so they are
    // skipped rather than re-verified.
    if (i + n_len <= h_len) {
        uint64_t mask = neon_lane_mask(NEON_ANCHORS(haystack + last));
        while (mask) {
            const size_t b = last + ((size_t)__builtin_ctzll(mask) >> 2);
            mask &= mask - 1;
            if (b < i) continue;
            if (anchors_cover_needle) return {true, b, false, 0};
            verified += n_len;
            if (equal_at(b)) return {true, b, false, 0};
            if (verified > budget_bytes) return {false, 0, true, b};
        }
    }
#undef NEON_ANCHORS
    return {false, 0, false, 0};
}

// The kernel as the algorithm table sees it: the shared body with a budget it
// can never exhaust, so this is the original algorithm with no guard behaviour.
std::pair<bool, size_t> neon_stringzilla_find(const char* haystack, size_t h_len,
                                              const char* needle, size_t n_len) {
    auto r = neon_stringzilla_body<false>(haystack, h_len, needle, n_len,
                                          ~(size_t)0);
    return {r.found, r.index};
}

// The same kernel with upstream's UTF-8 lead-byte anchor rule enabled, so the
// cost of that choice can be measured rather than assumed.
std::pair<bool, size_t> neon_stringzilla_find_hifilter(const char* haystack,
                                                       size_t h_len,
                                                       const char* needle,
                                                       size_t n_len) {
    auto r = neon_stringzilla_body<true>(haystack, h_len, needle, n_len,
                                         ~(size_t)0);
    return {r.found, r.index};
}

// Anchored kernel with a bounded verification budget. Same body as above with
// the counter compiled in, so any difference between the two is the guard and
// nothing else.
static inline simd_guarded_result neon_stringzilla_guarded(
    const char* text, size_t n, const char* pattern, size_t m,
    size_t budget_bytes) {
    return neon_stringzilla_body<false>(text, n, pattern, m, budget_bytes);
}


// A second way to combine the two ideas, for reference: keep the three-anchor
// filter but run it with the 64-byte stride of the wide naive kernel, so each
// anchor broadcast is reused across four windows and the per-window bookkeeping
// is paid once per 64 candidate positions instead of once per 16. Unlike the
// length-dispatched scheme there is no threshold -- the filter cost is already
// independent of m -- so this is length-insensitive everywhere.
//
// This matters more on NEON than on AVX-512, because a 128-bit window amortises
// the per-window fixed cost over a quarter as many positions.
std::pair<bool, size_t> neon_stringzilla64_find(const char* text, size_t n,
                                                const char* pattern, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    // One-byte needles have no three distinct anchors, and a short haystack has
    // no room for a block; the single-window kernel already handles both.
    if (m == 1 || n < m + kNeonB - 1) return neon_stringzilla_find(text, n, pattern, m);

    size_t off_first, off_mid, off_last;
    sz_locate_needle_anomalies(pattern, m, off_first, off_mid, off_last);
    const uint8x16_t vfirst = vdupq_n_u8((uint8_t)pattern[off_first]);
    const uint8x16_t vmid = vdupq_n_u8((uint8_t)pattern[off_mid]);
    const uint8x16_t vlast = vdupq_n_u8((uint8_t)pattern[off_last]);

    const bool anchors_cover_needle = (m <= 3);
    const bool fits = (m <= kNeonW);
    const neon_needle_reg nr = (fits && !anchors_cover_needle)
                                   ? neon_load_needle(pattern, m)
                                   : neon_needle_reg{vdupq_n_u8(0), 0};

    size_t i = 0;
    for (; i + m + kNeonB - 1 <= n; i += kNeonB) {
        const char* a = text + i + off_first;
        uint8x16_t fA = vceqq_u8(neon_load(a +  0), vfirst);
        uint8x16_t fB = vceqq_u8(neon_load(a + 16), vfirst);
        uint8x16_t fC = vceqq_u8(neon_load(a + 32), vfirst);
        uint8x16_t fD = vceqq_u8(neon_load(a + 48), vfirst);
        if (!neon_any(vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD)))) continue;

        const char* b = text + i + off_mid;
        fA = vandq_u8(fA, vceqq_u8(neon_load(b +  0), vmid));
        fB = vandq_u8(fB, vceqq_u8(neon_load(b + 16), vmid));
        fC = vandq_u8(fC, vceqq_u8(neon_load(b + 32), vmid));
        fD = vandq_u8(fD, vceqq_u8(neon_load(b + 48), vmid));
        if (!neon_any(vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD)))) continue;

        const char* c = text + i + off_last;
        fA = vandq_u8(fA, vceqq_u8(neon_load(c +  0), vlast));
        fB = vandq_u8(fB, vceqq_u8(neon_load(c + 16), vlast));
        fC = vandq_u8(fC, vceqq_u8(neon_load(c + 32), vlast));
        fD = vandq_u8(fD, vceqq_u8(neon_load(c + 48), vlast));

        // Verify survivors in increasing index order: chunk by chunk, lowest
        // lane first, so the first compare that succeeds is the leftmost
        // occurrence.
        const uint64_t masks[4] = {neon_lane_mask(fA), neon_lane_mask(fB),
                                   neon_lane_mask(fC), neon_lane_mask(fD)};
        for (size_t k = 0; k < 4; ++k) {
            uint64_t mk = masks[k];
            while (mk) {
                const size_t off = i + 16 * k + ((size_t)__builtin_ctzll(mk) >> 2);
                bool eq;
                if (anchors_cover_needle) eq = true;
                else if (fits && off + kNeonW <= n) eq = neon_needle_eq(text + off, nr);
                else eq = neon_equal(text + off, pattern, m);
                if (eq) return {true, off};
                mk &= mk - 1;
            }
        }
    }

    // Remainder: hand the uncovered suffix to the single-window kernel.
    auto [f, idx] = neon_stringzilla_find(text + i, n - i, pattern, m);
    if (!f) return {false, 0};
    return {true, i + idx};
}


// ===========================================================================
// Needle-Hammer
// ===========================================================================
//
// Length-dispatched: the wide naive kernel for needles up to Threshold bytes,
// the anchored kernel above it, and the single-window naive kernel whenever the
// haystack is too small for the wide one to amortise its block.
//
// The two kernels fail in opposite directions. The wide naive kernel spends
// O(m) broadcast-and-compare rounds per block whenever the needle's first four
// bytes keep candidates alive, so its cost grows with the needle length; but
// when the prefix filter bites (ordinary text) each pattern-byte broadcast is
// amortised over four windows. The anchored kernel is O(1) filter rounds per
// window no matter how long the needle is, but pays a per-window setup and a
// verifying compare per surviving candidate.
//
// So: cheap-and-length-sensitive below the threshold, length-insensitive above.
//
// Both constants are deliberately template parameters rather than the literals
// the AVX-512 header uses, because on ARM they have to be fitted across parts
// that disagree, not on one machine. The sweep instances at the bottom of this
// file exist for exactly that.
//
//   Threshold (tau)  the needle length at which the anchored kernel takes over.
//   MinWide (mu)     the haystack length below which the wide kernel's block
//                    granularity is not worth paying for: it cannot report a
//                    match until it has filtered the whole 64-byte block the
//                    match sits in, while the single-window kernel returns as
//                    soon as its one window has a hit.
template <size_t Threshold, size_t MinWide>
static inline std::pair<bool, size_t> neon_needle_hammer_t(const char* text, size_t n,
                                                           const char* pattern, size_t m) {
    if (m > Threshold) return neon_stringzilla_find(text, n, pattern, m);
    if (n < MinWide || n < m + kNeonB - 1)
        return neon_naive_search_body(text, n, pattern, m);
    return neon_naive_search64_body(text, n, pattern, m);
}

// The shipped constants. Overridable at build time so the sweep harness can
// re-fit them without editing this file.
//
// tau = 16 is set by the adversarial side alone, which is a different argument
// from the one that sets it at 512-bit width. On benign text at this width the
// two kernels' curves are close to parallel in m -- which of them wins is
// decided by the corpus alphabet, not by the needle length -- so the whole
// candidate range from 4 to 4096 lies within a few percent and there is no
// benign crossing to sit at. That frees the constant to satisfy the robustness
// bound instead: below tau the wide kernel runs, and on a needle a^(m-1)b over
// an all-'a' haystack its cost is linear in m while two-way's is flat, so the
// two cross at W* ~ 20 bytes. Any tau at or below 16 is adversarially maximal
// and, at this width, costs essentially nothing benign to be so.
//
// mu = 256 is a straight rescaling of the 1024 used at 512-bit width: the guard
// exists because the wide kernel cannot resolve a match before finishing its
// B-byte block, and B is a quarter the size here.
#ifndef NEON_NH_TAU
#define NEON_NH_TAU 16
#endif
#ifndef NEON_NH_MU
#define NEON_NH_MU 256
#endif

std::pair<bool, size_t> neon_needle_hammer(const char* text, size_t n,
                                           const char* pattern, size_t m) {
    return neon_needle_hammer_t<NEON_NH_TAU, NEON_NH_MU>(text, n, pattern, m);
}

// Sweep instances, so both constants can be located empirically rather than
// assumed. The tau row holds mu at its shipped value and vice versa: the two
// are close to independent, because mu only decides which naive kernel runs
// below tau and tau only decides whether a naive kernel runs at all.
#define NEON_NH_TAU_SWEEP(T)                                                   \
    std::pair<bool, size_t> neon_needle_hammer_t##T(                           \
        const char* t, size_t n, const char* p, size_t m) {                    \
        return neon_needle_hammer_t<T, NEON_NH_MU>(t, n, p, m);                \
    }
NEON_NH_TAU_SWEEP(4)    NEON_NH_TAU_SWEEP(8)    NEON_NH_TAU_SWEEP(16)
NEON_NH_TAU_SWEEP(32)   NEON_NH_TAU_SWEEP(64)   NEON_NH_TAU_SWEEP(128)
NEON_NH_TAU_SWEEP(256)  NEON_NH_TAU_SWEEP(512)  NEON_NH_TAU_SWEEP(1024)
NEON_NH_TAU_SWEEP(2048) NEON_NH_TAU_SWEEP(4096)
#undef NEON_NH_TAU_SWEEP

#define NEON_NH_MU_SWEEP(U)                                                    \
    std::pair<bool, size_t> neon_needle_hammer_m##U(                           \
        const char* t, size_t n, const char* p, size_t m) {                    \
        return neon_needle_hammer_t<NEON_NH_TAU, U>(t, n, p, m);               \
    }
NEON_NH_MU_SWEEP(0)    NEON_NH_MU_SWEEP(64)   NEON_NH_MU_SWEEP(128)
NEON_NH_MU_SWEEP(256)  NEON_NH_MU_SWEEP(512)  NEON_NH_MU_SWEEP(1024)
NEON_NH_MU_SWEEP(2048) NEON_NH_MU_SWEEP(4096)
#undef NEON_NH_MU_SWEEP


// ===========================================================================
// Work-counter guarded Needle-Hammer
// ===========================================================================
//
// Needle-Hammer is fast on real data and degrades to O(n*m) on inputs chosen to
// defeat its filter. Each kernel below counts the work its verification path
// actually does and abandons the scan for two-way once that count exceeds a
// budget proportional to n. Because the budget is proportional to n and two-way
// is linear, the guarded searcher is O(n) on every input, while on ordinary
// text the counter never approaches the budget and the fast path is untouched.
//
// The unit is the same as on AVX-512 -- narrowing rounds for the wide kernel,
// bytes of verification for the anchored one -- but the accounting moves,
// because a NEON block is 64 bytes rather than 256:
//
//   * A round still costs about the same handful of instructions (one broadcast
//     and four load-compare-and triples), so a budget of n * Num / Den rounds
//     still buys the same number of instructions as it does at 512-bit. The
//     instruction model that sets the constant therefore carries over unchanged.
//
//   * What does not carry over is the length at which the guard becomes
//     reachable. A block admits at most m - 4 rounds, so the whole haystack
//     admits at most n(m-4)/B rounds, and the budget is out of reach exactly
//     when m - 4 <= B * Num / Den. With B = 256 and a budget of n/8 that free
//     range runs to m = 36; with B = 64 it stops at m = 12. On ARM the same
//     budget therefore starts guarding needles three times shorter, which is
//     why the budget is a template parameter here and why the shipped value is
//     re-fitted rather than copied.
static inline simd_guarded_result neon_naive_search64_guarded(
    const char* text, size_t n, const char* pattern, size_t m,
    size_t budget_rounds) {
    if (m == 0) return {true, 0, false, 0};
    if (n < m) return {false, 0, false, 0};
    // m < 4 has no narrowing loop at all, and a haystack too short for one
    // block does bounded work by construction.
    if (m < 4 || n < m + kNeonB - 1) {
        auto [f, idx] = neon_naive_search64_body(text, n, pattern, m);
        return {f, idx, false, 0};
    }

    const bool fits = (m <= kNeonW);
    const neon_needle_reg nr =
        fits ? neon_load_needle(pattern, m) : neon_needle_reg{vdupq_n_u8(0), 0};
    auto verify = [&](size_t pos) -> bool {
        if (fits && pos + kNeonW <= n) return neon_needle_eq(text + pos, nr);
        return std::memcmp(text + pos + 4, pattern + 4, m - 4) == 0;
    };

    const uint8x16_t p0 = vdupq_n_u8((uint8_t)pattern[0]);
    const uint8x16_t p1 = vdupq_n_u8((uint8_t)pattern[1]);
    const uint8x16_t p2 = vdupq_n_u8((uint8_t)pattern[2]);
    const uint8x16_t p3 = vdupq_n_u8((uint8_t)pattern[3]);
#define NEON_CHUNK(OFF)                                                        \
        vandq_u8(vandq_u8(vceqq_u8(neon_load(text + i + (OFF) + 0), p0),       \
                          vceqq_u8(neon_load(text + i + (OFF) + 1), p1)),      \
                 vandq_u8(vceqq_u8(neon_load(text + i + (OFF) + 2), p2),       \
                          vceqq_u8(neon_load(text + i + (OFF) + 3), p3)))

    size_t rounds = 0;
    size_t i = 0;
    for (; i + m + kNeonB - 1 <= n; i += kNeonB) {
        uint8x16_t fA = NEON_CHUNK(0),  fB = NEON_CHUNK(16);
        uint8x16_t fC = NEON_CHUNK(32), fD = NEON_CHUNK(48);
        uint8x16_t any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
        if (!neon_any(any)) continue;

        // One survivor in the block: verified in place, no narrowing, so
        // nothing is charged. That is not a hole an adversary can widen -- the
        // path costs at most m bytes of vector compare per 64 bytes of
        // haystack, far below what two-way would spend on the same text. The
        // expensive path is the one with many survivors, and that is the one
        // counted.
        const uint64_t mA = neon_lane_mask(fA), mB = neon_lane_mask(fB);
        const uint64_t mC = neon_lane_mask(fC), mD = neon_lane_mask(fD);
        const int nz = (mA != 0) + (mB != 0) + (mC != 0) + (mD != 0);
        const uint64_t one = mA | mB | mC | mD;
        if (nz == 1 && (one & (one - 1)) == 0) {
            const size_t off = (mA != 0) ? 0 : (mB != 0) ? 16 : (mC != 0) ? 32 : 48;
            const size_t b = i + off + ((size_t)__builtin_ctzll(one) >> 2);
            if (verify(b)) return {true, b, false, 0};
            continue;
        }

        // The only unbounded loop in this kernel, and so the only one counted.
        // The budget is tested once per block rather than once per round:
        // keeping the inner loop identical to the unguarded kernel is what
        // makes the benign overhead vanish. Overshoot is at most m-4 rounds.
        size_t j = 4;
        for (; j < m; ++j) {
            any = vorrq_u8(vorrq_u8(fA, fB), vorrq_u8(fC, fD));
            if (!neon_any(any)) break;
            const uint8x16_t pj = vdupq_n_u8((uint8_t)pattern[j]);
            fA = vandq_u8(fA, vceqq_u8(neon_load(text + i + j +  0), pj));
            fB = vandq_u8(fB, vceqq_u8(neon_load(text + i + j + 16), pj));
            fC = vandq_u8(fC, vceqq_u8(neon_load(text + i + j + 32), pj));
            fD = vandq_u8(fD, vceqq_u8(neon_load(text + i + j + 48), pj));
        }
        rounds += j - 4;

        // Resolve before testing the budget. The narrowing above is spent
        // whether or not the budget survives it, so giving up on a block that
        // just produced the answer would discard a proven match and pay two-way
        // to find it again.
        uint64_t r;
        if ((r = neon_lane_mask(fA))) return {true, i +  0 + ((size_t)__builtin_ctzll(r) >> 2), false, 0};
        if ((r = neon_lane_mask(fB))) return {true, i + 16 + ((size_t)__builtin_ctzll(r) >> 2), false, 0};
        if ((r = neon_lane_mask(fC))) return {true, i + 32 + ((size_t)__builtin_ctzll(r) >> 2), false, 0};
        if ((r = neon_lane_mask(fD))) return {true, i + 48 + ((size_t)__builtin_ctzll(r) >> 2), false, 0};
        if (rounds > budget_rounds) return {false, 0, true, i};
    }
#undef NEON_CHUNK

    // Remainder: fewer than m + 63 bytes left, so the narrowing the 16-byte
    // kernel may do there is bounded by a constant independent of n. Uncounted
    // for the same reason the m < 4 case is.
    if (i + m <= n) {
        auto r = neon_naive_search_body(text + i, n - i, pattern, m);
        if (r.first) return {true, i + r.second, false, 0};
    }
    return {false, 0, false, 0};
}

// The guarded scheme. BudgetNum/BudgetDen scales both budgets against n so the
// constant can be measured rather than asserted:
//   wide kernel      budget = n * BudgetNum / BudgetDen narrowing rounds
//   anchored kernel  budget = AnchorBudget * n bytes of verification
template <size_t Threshold, size_t BudgetNum = 1, size_t BudgetDen = 8,
          size_t MinWide = NEON_NH_MU, size_t AnchorBudget = 16>
static inline std::pair<bool, size_t> neon_needle_hammer_guarded_t(
    const char* text, size_t n, const char* pattern, size_t m) {
    // Needles this short cannot exhaust the budget however hostile the input:
    // the narrowing loop runs at most m-4 times per 64-byte block, i.e. about
    // (n/64)(m-4) rounds, which stays under n*Num/Den exactly when
    // m - 4 <= 64*Num/Den. Below that bound the guard is not merely cheap, it
    // is absent -- the call goes straight to the unguarded kernel.
    constexpr size_t kFreeBelow = 4 + (kNeonB * BudgetNum) / BudgetDen;

    if (m > Threshold) {
        // Budget the verification in bytes against the HAYSTACK, not the
        // needle: benign verification work is (candidates surviving three
        // anchors) x m, and on a long needle drawn from ordinary text the
        // anchors are common letters, so a flat byte budget trips on perfectly
        // benign input. Scaling with n caps adversarial verification
        // independently of m and still sits far above the benign load.
        auto r = neon_stringzilla_guarded(text, n, pattern, m, AnchorBudget * n);
        if (!r.gave_up) return {r.found, r.index};
        return resume_twoway(text, n, pattern, m, r.resume);
    }
    // Small haystacks take the single-window kernel, which needs no guard: it
    // is entered only when n < MinWide or n < m + 63, and in both cases the
    // number of window positions is O(m) rather than O(n).
    if (n < MinWide || n < m + kNeonB - 1)
        return neon_naive_search_body(text, n, pattern, m);
    // The free case: one comparison against a compile-time constant, then the
    // unguarded kernel.
    if (m <= kFreeBelow) return neon_naive_search64_body(text, n, pattern, m);

    const size_t budget = (n / BudgetDen) * BudgetNum + 1;
    auto r = neon_naive_search64_guarded(text, n, pattern, m, budget);
    if (!r.gave_up) return {r.found, r.index};
    return resume_twoway(text, n, pattern, m, r.resume);
}

// The recommended configuration: the shipped switch point with the shipped
// budget.
#ifndef NEON_NH_BUDGET_NUM
#define NEON_NH_BUDGET_NUM 1
#endif
#ifndef NEON_NH_BUDGET_DEN
#define NEON_NH_BUDGET_DEN 2
#endif

std::pair<bool, size_t> neon_needle_hammer_guarded(const char* t, size_t n,
                                                   const char* p, size_t m) {
    return neon_needle_hammer_guarded_t<NEON_NH_TAU, NEON_NH_BUDGET_NUM,
                                        NEON_NH_BUDGET_DEN>(t, n, p, m);
}

// Tighter and looser budgets, so the cost of the guard and the sharpness of the
// bound can both be measured instead of assumed. n/32 keeps the free range at
// m <= 6 (so nearly every needle is counted); n/2 keeps it at m <= 36, matching
// the free range AVX-512 gets from n/8.
//
// Note that with the shipped ARM budget at n/2 the "loose" instance coincides
// with the default, so the three distinct settings actually measured are n/32
// (tight), n/8 (the AVX-512 budget, below) and n/2. That coincidence is the
// point rather than an oversight: the fit moved the shipped value to what would
// have been the loose end at 512-bit width, because a NEON block is a quarter
// the size and the same rounds-per-byte budget starts guarding needles three
// times shorter. Widening "loose" further would measure a budget nothing
// recommends.
std::pair<bool, size_t> neon_needle_hammer_guarded_tight(const char* t, size_t n,
                                                         const char* p, size_t m) {
    return neon_needle_hammer_guarded_t<NEON_NH_TAU, 1, 32>(t, n, p, m);
}
std::pair<bool, size_t> neon_needle_hammer_guarded_loose(const char* t, size_t n,
                                                         const char* p, size_t m) {
    return neon_needle_hammer_guarded_t<NEON_NH_TAU, 1, 2>(t, n, p, m);
}
// The AVX-512 budget transplanted unchanged, for the comparison the appendix
// makes: same rounds-per-byte, but a free range that stops at m = 12 instead of
// m = 36 because the block is a quarter of the size.
std::pair<bool, size_t> neon_needle_hammer_guarded_avx512budget(const char* t, size_t n,
                                                                const char* p, size_t m) {
    return neon_needle_hammer_guarded_t<NEON_NH_TAU, 1, 8>(t, n, p, m);
}
