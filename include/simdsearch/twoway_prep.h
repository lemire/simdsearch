#pragma once
// The needle preprocessing of the Crochemore-Perrin two-way algorithm: the
// critical factorization and the `twoway_prep` record built from it. Both the
// scalar two-way (kmp_twoway.h) and the vectorized one (twoway_simd.h) run on
// it; it is split out so the library's public header depends on nothing else
// from the scalar searchers.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Critical factorization: split the needle into needle[0..crit) . needle[crit..m)
// at a point where the local period equals the global period. Crochemore and
// Perrin show that with an ordered alphabet, one of the two maximal suffixes --
// under the normal ordering and under its reverse -- starts at such a point, so
// compute both and keep the one that starts later. Returns crit and sets *period
// to the period of the right half; the factorization guarantees crit < *period
// and crit + *period <= m.
//
// max_suffix is the index of the last byte of the left half, or SIZE_MAX when
// the left half is empty; needle[max_suffix + k] then wraps to needle[k - 1],
// which is exactly the byte the algorithm wants. That wraparound is deliberate
// (and well defined for unsigned arithmetic).
static inline size_t twoway_critical_factorization(const unsigned char *needle,
                                                   size_t m, size_t *period) {
  if (m < 3) {  // 1 and 2 have nothing to factor
    *period = 1;
    return m - 1;
  }

  size_t max_suffix = SIZE_MAX, j = 0, k = 1, p = 1;
  while (j + k < m) {
    unsigned char a = needle[j + k], b = needle[max_suffix + k];
    if (a < b) {  // suffix is smaller: the period is the whole prefix so far
      j += k;
      k = 1;
      p = j - max_suffix;
    } else if (a == b) {  // advance through a repetition of the current period
      if (k != p) {
        ++k;
      } else {
        j += p;
        k = 1;
      }
    } else {  // suffix is larger: restart here
      max_suffix = j++;
      k = p = 1;
    }
  }
  *period = p;

  // Same walk again with the comparison reversed.
  size_t max_suffix_rev = SIZE_MAX;
  j = 0;
  k = p = 1;
  while (j + k < m) {
    unsigned char a = needle[j + k], b = needle[max_suffix_rev + k];
    if (b < a) {
      j += k;
      k = 1;
      p = j - max_suffix_rev;
    } else if (a == b) {
      if (k != p) {
        ++k;
      } else {
        j += p;
        k = 1;
      }
    } else {
      max_suffix_rev = j++;
      k = p = 1;
    }
  }

  // Keep the factorization with the shorter left half (+1 turns "last byte of
  // the left half" into "first byte of the right half", and maps SIZE_MAX to 0).
  if (max_suffix_rev + 1 < max_suffix + 1) return max_suffix + 1;
  *period = p;
  return max_suffix_rev + 1;
}

// Everything two-way needs to know about a needle.
struct twoway_prep {
  size_t crit = 0;      // start of the right half
  size_t period = 1;    // global period when `periodic`, else the mismatch shift
  bool periodic = false;  // does the left half repeat the needle's period?

  void build(const char *needle, size_t m) {
    if (m == 0) {
      crit = 0;
      period = 1;
      periodic = false;
      return;
    }
    crit = twoway_critical_factorization((const unsigned char *)needle, m,
                                         &period);
    // The needle is periodic iff its left half is a repetition of the period;
    // crit + period <= m, so this compare stays in bounds.
    periodic = std::memcmp(needle, needle + period, crit) == 0;
    // Aperiodic needles cannot overlap themselves within the window, so any
    // mismatch lets us shift past the whole factorization.
    if (!periodic) period = std::max(crit, m - crit) + 1;
  }
};

