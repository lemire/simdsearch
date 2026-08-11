#pragma once
// Everything the SIMD backends share: the anchor selector that both anchored
// kernels filter on, and the scalar and library searchers the benchmark
// compares against. Nothing here is architecture-specific, so avx512search.h
// and neonsearch.h both include it and neither has to carry a copy.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <utility>

#include "kmp_twoway.h"

// Pick three needle offsets to anchor the SIMD pre-filter on: first, middle and
// last, walked inward so the three bytes stay distinct where possible. This is
// StringZilla's sz_locate_needle_anomalies_ (find.h), with its one
// encoding-specific refinement made optional -- see below.
//
// FilterHighBytes (default false)
// ------------------------------
// Upstream additionally prefers, for needles longer than 8 bytes, byte values
// at or below 191. The argument is about information content under UTF-8: a byte
// above 191 is the LEAD byte of a multi-byte rune (0b110xxxxx, 0b1110xxxx or
// 0b11110xxx), whose high bits encode the sequence length, leaving only 3 to 5
// bits that distinguish one character from another. ASCII carries 7 such bits and
// a continuation byte (128..191) carries 6, so continuation bytes are good
// anchors and are kept; only lead bytes are skipped.
//
// We make this opt-in and leave it OFF by default, so the kernel is generic
// rather than tuned for one encoding. The rule helps only where lead bytes are
// both frequent and drawn from a tiny set of values -- measurably on Cyrillic
// UTF-8, where nearly every lead byte is 0xD0 or 0xD1, and not detectably
// anywhere else tested. Both settings are benchmarked, so the difference can be
// measured on whatever corpus you care about.
template <bool FilterHighBytes = false>
static inline void sz_locate_needle_anomalies_t(const char* start, size_t length,
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

    if constexpr (FilterHighBytes) {
        if (length > 8) {
            size_t vfirst = first, vsecond = second, vthird = third;
            while ((s[vsecond] > 191 || s[vsecond] == s[vthird]) &&
                   (vsecond + 1 < vthird))
                ++vsecond;
            // Asymmetric on purpose: the skip loop stops at 191 but the
            // accept test rejects it, so a needle whose only non-lead candidate
            // is exactly 0xBF keeps the default anchor. 0xBF is a continuation
            // byte and the heuristic above says it is a good one, so this is an
            // off-by-one -- but it is upstream's, and this is a port. Left as
            // is so the anchors match sz_find byte for byte; it costs at most
            // one missed anchor improvement and never correctness.
            if (s[vsecond] < 191) second = vsecond;
            else vsecond = second;
            while ((s[vfirst] > 191 || s[vfirst] == s[vsecond] ||
                    s[vfirst] == s[vthird]) &&
                   (vfirst + 1 < vsecond))
                ++vfirst;
            if (s[vfirst] < 191) first = vfirst;
        }
    }
}

// The default selector: generic, no encoding-specific rule.
static inline void sz_locate_needle_anomalies(const char* start, size_t length,
                                              size_t& first, size_t& second,
                                              size_t& third) {
    sz_locate_needle_anomalies_t<false>(start, length, first, second, third);
}


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


// Result of a kernel that may abandon its scan. `resume` is the first position
// not yet ruled out, so the fallback searcher need not rescan the prefix. Both
// backends' guarded kernels report through this.
struct simd_guarded_result {
    bool found;
    size_t index;
    bool gave_up;    // budget exhausted; caller must fall back
    size_t resume;   // first position not yet ruled out, when gave_up
};

// Hand the rest of the haystack to Crochemore-Perrin. The kernel has already
// proved there is no occurrence starting below `from`, so two-way only has to
// cover the remainder -- the abandoned prefix is not rescanned. Plain two-way
// rather than the bad-character variant: on the low-diversity haystacks that
// trip the guard the skip table buys nothing and measures slower.
static inline std::pair<bool, size_t> resume_twoway(const char* text, size_t n,
                                                    const char* pattern, size_t m,
                                                    size_t from) {
    if (from + m > n) return {false, 0};
    auto [f, idx] = twoway_search(text + from, n - from, pattern, m);
    if (!f) return {false, 0};
    return {true, from + idx};
}
