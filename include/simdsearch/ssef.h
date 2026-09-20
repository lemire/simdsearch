#pragma once
// SSEF: Külekci's SIMD filter for long patterns (Prague Stringology
// Conference 2009), after the implementation in SMART (Faro and Lecroq).
//
// The text is read in aligned 16-byte blocks. A block is reduced to a 16-bit
// fingerprint, one bit per byte, taken from one bit position of each byte;
// a table built from the pattern maps each fingerprint value to the pattern
// alignments (offsets of the pattern's own 16-byte windows) consistent with
// it, and only those alignments are verified with memcmp. With L = m/16 - 1
// the search probes one block in L: a pattern of length m covers at least L
// whole aligned blocks wherever it sits, so one of them is probed. That is
// what makes the algorithm sublinear on average -- it does not look at most
// of the text -- and it is why it needs m >= 32 (L >= 1).
//
// SMART's code takes the sign bit of each byte, which on ASCII text is always
// zero and leaves the filter with nothing to work with; Külekci's paper
// shifts the bytes so that an informative bit lands in the sign position and
// suggests the shift per alphabet. Here the bit is chosen from the pattern:
// the one whose values over the pattern's bytes are closest to half and half,
// which is the most selective single bit the fingerprint can use.
//
// The table is built once per pattern (ssef_prep) and reused across searches,
// the same amortized setting the benchmark gives Horspool, KMP and two-way;
// the per-call cost of clearing a 65536-entry table would otherwise dominate
// a short search. Like SMART's version this searcher reports the first
// occurrence; candidates from one probed block are verified in ascending
// order and successive probed blocks cover disjoint, increasing ranges of
// start positions, so the first verified match is the first occurrence.
#include <emmintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

struct ssef_prep {
    static constexpr uint32_t kNone = 0xFFFFFFFFu;
    size_t m = 0;
    size_t last = 0;              // blocks skipped per probe, m/16 - 1
    int shift = 0;                // left shift that brings the chosen bit to the sign
    std::vector<uint32_t> head;   // fingerprint -> first entry, or kNone
    std::vector<uint32_t> next;   // entry -> next entry with the same fingerprint
    std::vector<uint32_t> pos;    // entry -> pattern alignment offset

    bool applicable() const { return m >= 32; }

    void build(const char* pattern, size_t m_) {
        m = m_;
        head.clear(); next.clear(); pos.clear();
        if (m < 32) return;
        const unsigned char* x = (const unsigned char*)pattern;
        // The bit whose split over the pattern's bytes is closest to even.
        int best_bit = 7; size_t best_gap = m + 1;
        for (int b = 0; b < 8; ++b) {
            size_t ones = 0;
            for (size_t i = 0; i < m; ++i) ones += (x[i] >> b) & 1;
            const size_t gap = ones > m - ones ? ones - (m - ones) : (m - ones) - ones;
            if (gap < best_gap) { best_gap = gap; best_bit = b; }
        }
        shift = 7 - best_bit;
        last = m / 16 - 1;
        head.assign(65536, kNone);
        const size_t entries = last * 16;
        next.resize(entries); pos.resize(entries);
        std::vector<uint32_t> tail(65536, kNone);
        // Alignment i pairs the probed text block with the pattern window that
        // starts at j = last*16 - i; the block's fingerprint must equal that
        // window's. Entries are appended in increasing i, so each list is in
        // ascending order of start position.
        for (size_t i = 0; i < entries; ++i) {
            const size_t j = entries - i;
            unsigned f = 0;
            for (int k = 0; k < 16; ++k) f |= (unsigned)((x[j + k] >> best_bit) & 1) << k;
            pos[i] = (uint32_t)i; next[i] = kNone;
            if (head[f] == kNone) head[f] = (uint32_t)i; else next[tail[f]] = (uint32_t)i;
            tail[f] = (uint32_t)i;
        }
    }

    std::pair<bool, size_t> search(const char* text, size_t n, const char* pattern) const {
        if (n < m) return {false, 0};
        const unsigned char* y = (const unsigned char*)text;
        const size_t nblocks = n / 16;
        const __m128i zero = _mm_setzero_si128();
        (void)zero;
        for (size_t blk = last; blk < nblocks; blk += last) {
            __m128i v = _mm_loadu_si128((const __m128i*)(y + blk * 16));
            // Shift within 16-bit lanes: the high byte of each lane receives
            // only its own bits in bit 7, so the sign of every byte is the
            // chosen bit of that byte.
            if (shift) v = _mm_slli_epi16(v, shift);
            const unsigned f = (unsigned)_mm_movemask_epi8(v);
            uint32_t e = head[f];
            if (e == kNone) continue;
            const size_t base = (blk - last) * 16;
            for (; e != kNone; e = next[e]) {
                const size_t start = base + pos[e];
                if (start + m <= n && std::memcmp(pattern, y + start, m) == 0) return {true, start};
            }
        }
        return {false, 0};
    }
};

// Stateless entry point: preprocess, then search. Patterns shorter than 32
// bytes are outside the algorithm's domain and reported as not applicable
// ({false, SIZE_MAX}, the benchmark's sentinel).
static inline std::pair<bool, size_t> ssef_search(const char* text, size_t n,
                                                  const char* pattern, size_t m) {
    if (m < 32) return {false, SIZE_MAX};
    ssef_prep p; p.build(pattern, m);
    return p.search(text, n, pattern);
}
