#pragma once
// simdsearch: the public header.
//
//   #include <simdsearch/simdsearch.h>
//   auto [found, index] = simdsearch::find(text, n, pattern, m);
//   size_t pos = simdsearch::find(haystack_view, needle_view);   // or npos
//
// Needle-Hammer (needle_hammer.h): SIMD filter plus a linear-time two-way
// fallback, so every input is searched in O(n + m). AVX-512 (F + BW) on
// x86-64 or NEON on AArch64, chosen at compile time; see the README.
#include "needle_hammer.h"

namespace simdsearch {

// First occurrence of pattern[0..m) in text[0..n): {true, index} or
// {false, 0}. The empty needle matches at 0. Neither buffer needs a NUL
// terminator, both may contain any byte value, and no byte outside them is
// read.
using needle_hammer::find;

// The same kernel without the work counter: a few percent faster on ordinary
// text, no longer linear in the worst case. For measurement, not for input
// you do not control.
using needle_hammer::find_unguarded;

}  // namespace simdsearch
