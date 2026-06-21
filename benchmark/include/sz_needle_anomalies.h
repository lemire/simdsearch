#pragma once

#include <cstddef>

// Pick three needle offsets to anchor the SIMD pre-filter on, faithfully
// porting StringZilla's sz_locate_needle_anomalies_ (find.h). Start with first /
// middle / last; if any of the three bytes collide, walk the middle and last
// offsets inward so the trio stays distinct. For needles longer than 8 bytes,
// prefer "vibrant" bytes < 191 — values >= 192 are UTF-8 continuation bytes.
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