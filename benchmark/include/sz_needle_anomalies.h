#pragma once

#include <cstddef>

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
// anywhere else we tested. Both settings are benchmarked; the paper reports the
// difference.
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
