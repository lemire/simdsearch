#pragma once
// simdsearch: substring search with AVX-512.
//
//   #include <simdsearch/simdsearch.h>
//   auto [found, index] = simdsearch::find(text, n, pattern, m);
//   size_t pos = simdsearch::find(haystack, needle);   // string_views; npos if absent
//
// The searcher is Guardzilla (guardzilla.h). Needs AVX-512F and AVX-512BW.
#include <string_view>

#include "guardzilla.h"

namespace simdsearch {

using guardzilla::find;

inline size_t find(std::string_view haystack, std::string_view needle) {
    const auto [found, index] = guardzilla::find(haystack.data(), haystack.size(), needle.data(), needle.size());
    return found ? index : std::string_view::npos;
}

}  // namespace simdsearch
