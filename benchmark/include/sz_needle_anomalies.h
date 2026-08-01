#pragma once

#include <cstddef>

// Pick three needle offsets to anchor the SIMD pre-filter on, faithfully
// porting StringZilla's sz_locate_needle_anomalies_ (find.h). Start with first /
// middle / last; if any of the three bytes collide, walk the middle and last
// offsets inward so the trio stays distinct. For needles longer than 8 bytes,
// prefer "vibrant" bytes <= 191.
//
// The 191 bound is about information content, not about non-ASCII text. A byte
// above 191 is the LEAD byte of a multi-byte UTF-8 rune (0b110xxxxx, 0b1110xxxx
// or 0b11110xxx), whose top bits encode the sequence length, leaving only 5, 4 or
// 3 bits that distinguish one character from another. ASCII carries 7 such bits
// and a continuation byte (128..191) carries 6. So continuation bytes are good
// anchors and are deliberately KEPT; it is the lead bytes that are skipped. In
// Cyrillic text, for instance, almost every lead byte is 0xD0 or 0xD1, which
// would anchor on a large fraction of positions, while the continuation byte
// beside it identifies the letter.
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