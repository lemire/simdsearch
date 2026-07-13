#pragma once

// Two classical comparison-based searchers with worst-case linear time, in the
// same {found, index} interface as the rest of the benchmark:
//
//   kmp_search        Knuth-Morris-Pratt: O(m) failure table, then a scan that
//                     never re-reads a text byte (at most 2n comparisons).
//   twoway_search     Crochemore-Perrin two-way: O(m) critical factorization,
//                     no tables, O(1) extra space, at most 2n comparisons.
//   twoway_search_bc  two-way plus a 256-entry bad-character skip table -- the
//                     variant glibc's memmem/strstr actually run for needles
//                     longer than 32 bytes. Sublinear in practice, still linear
//                     in the worst case.
//
// Each algorithm comes in two forms: a stateless function that rebuilds its
// preprocessing on every call, and a "prepared" object that holds the
// preprocessing so it can be reused across searches with the same needle. The
// benchmark reports both, since for short needles the O(m + sigma) preprocessing
// can easily dominate the scan.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Knuth-Morris-Pratt
// ---------------------------------------------------------------------------

// Build the KMP failure links for pattern[0..m). next[i] is where to resume in
// the pattern after a mismatch against pattern[i], with next[0] == -1 as the
// sentinel meaning "no prefix left; consume the text byte and restart".
//
// This is the "strong" form from the original KMP paper: when the border we
// would fall back to is followed by the very byte that just mismatched, the
// link jumps past it (next[i] = next[j] instead of j), so a text byte is never
// compared against a pattern byte that is already known to differ.
//
// next needs m + 1 entries. int32_t keeps the table cache-friendly and caps
// needles at 2 GB, which is far beyond anything this benchmark searches for.
static inline void kmp_build(const char *pattern, size_t m, int32_t *next) {
  next[0] = -1;
  int32_t j = -1;
  for (size_t i = 0; i < m;) {
    while (j >= 0 && pattern[i] != pattern[(size_t)j]) j = next[j];
    ++i;
    ++j;
    next[i] = (i < m && pattern[i] == pattern[(size_t)j]) ? next[j] : j;
  }
}

// KMP scan against a pre-built failure table. The text index only ever moves
// forward: on a mismatch the matched-prefix length k falls back through the
// failure links instead of rewinding the text.
static inline std::pair<bool, size_t> kmp_search_with(const int32_t *next,
                                                      const char *text, size_t n,
                                                      const char *pattern,
                                                      size_t m) {
  if (m == 0) return {true, 0};
  if (n < m) return {false, 0};
  int32_t k = 0;  // bytes of the pattern matched so far
  for (size_t i = 0; i < n; ++i) {
    while (k >= 0 && pattern[(size_t)k] != text[i]) k = next[k];
    ++k;
    if ((size_t)k == m) return {true, i + 1 - m};
  }
  return {false, 0};
}

// Failure table owned across calls, for the amortized variant.
struct kmp_prep {
  std::vector<int32_t> next;

  void build(const char *pattern, size_t m) {
    next.resize(m + 1);
    kmp_build(pattern, m, next.data());
  }

  std::pair<bool, size_t> find(const char *text, size_t n, const char *pattern,
                               size_t m) const {
    return kmp_search_with(next.data(), text, n, pattern, m);
  }
};

// Needles up to this length build their table on the stack, so the common case
// pays no allocation.
static constexpr size_t kmp_stack_needle = 1024;

// Stateless KMP: builds the failure table, then scans.
static inline std::pair<bool, size_t> kmp_search(const char *text, size_t n,
                                                 const char *pattern, size_t m) {
  if (m == 0) return {true, 0};
  if (n < m) return {false, 0};
  int32_t stack_next[kmp_stack_needle + 1];
  std::vector<int32_t> heap_next;
  int32_t *next = stack_next;
  if (m > kmp_stack_needle) {
    heap_next.resize(m + 1);
    next = heap_next.data();
  }
  kmp_build(pattern, m, next);
  return kmp_search_with(next, text, n, pattern, m);
}

// ---------------------------------------------------------------------------
// Two-way (Crochemore-Perrin), following glibc's string/str-two-way.h
// ---------------------------------------------------------------------------

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

// Two-way scan with a pre-built factorization. Each window is checked right half
// first (left to right from crit), then left half (right to left). Because the
// factorization is critical, a mismatch in the right half at offset i lets the
// window jump to i - crit + 1 without missing an occurrence.
static inline std::pair<bool, size_t> twoway_search_with(const twoway_prep &tw,
                                                         const char *text,
                                                         size_t n,
                                                         const char *pattern,
                                                         size_t m) {
  if (m == 0) return {true, 0};
  if (n < m) return {false, 0};
  const unsigned char *hay = (const unsigned char *)text;
  const unsigned char *nd = (const unsigned char *)pattern;
  const size_t crit = tw.crit;
  const size_t period = tw.period;
  size_t j = 0;

  if (tw.periodic) {
    // A periodic needle shifts by only one period after a full right-half
    // match, so remember how much of the right half is already known to match
    // and skip re-comparing it (this is what keeps the scan linear).
    size_t memory = 0;
    while (j + m <= n) {
      size_t i = std::max(crit, memory);
      while (i < m && nd[i] == hay[i + j]) ++i;
      if (i < m) {  // right half failed: shift past the mismatch
        j += i - crit + 1;
        memory = 0;
        continue;
      }
      size_t d = crit;  // left half, backwards, down to what memory covers
      while (d > memory && nd[d - 1] == hay[d - 1 + j]) --d;
      if (d <= memory) return {true, j};
      j += period;
      memory = m - period;
    }
  } else {
    while (j + m <= n) {
      size_t i = crit;
      while (i < m && nd[i] == hay[i + j]) ++i;
      if (i < m) {
        j += i - crit + 1;
        continue;
      }
      size_t d = crit;
      while (d > 0 && nd[d - 1] == hay[d - 1 + j]) --d;
      if (d == 0) return {true, j};
      j += period;  // = max(crit, m - crit) + 1
    }
  }
  return {false, 0};
}

// Stateless two-way: factor the needle, then scan.
static inline std::pair<bool, size_t> twoway_search(const char *text, size_t n,
                                                    const char *pattern,
                                                    size_t m) {
  if (m == 0) return {true, 0};
  if (n < m) return {false, 0};
  twoway_prep tw;
  tw.build(pattern, m);
  return twoway_search_with(tw, text, n, pattern, m);
}

// Two-way plus a bad-character table, as in glibc's two_way_long_needle: before
// touching the window at all, look up the shift for the haystack byte aligned
// with the needle's last byte. Non-zero shift means no occurrence can start
// here, so the window jumps without a single needle comparison -- the sublinear
// behaviour of Boyer-Moore-Horspool, on top of two-way's linear worst case.
struct twoway_bc_prep {
  twoway_prep tw;
  size_t bad[256];  // distance from a byte's last occurrence to the needle end

  void build(const char *needle, size_t m) {
    tw.build(needle, m);
    std::fill(bad, bad + 256, m);
    for (size_t i = 0; i < m; ++i)
      bad[(unsigned char)needle[i]] = m - i - 1;
  }
};

static inline std::pair<bool, size_t> twoway_bc_search_with(
    const twoway_bc_prep &tb, const char *text, size_t n, const char *pattern,
    size_t m) {
  if (m == 0) return {true, 0};
  if (n < m) return {false, 0};
  const unsigned char *hay = (const unsigned char *)text;
  const unsigned char *nd = (const unsigned char *)pattern;
  const size_t crit = tb.tw.crit;
  const size_t period = tb.tw.period;
  size_t j = 0;

  if (tb.tw.periodic) {
    size_t memory = 0;
    while (j + m <= n) {
      size_t shift = tb.bad[hay[j + m - 1]];
      if (shift > 0) {
        // A short shift on a periodic needle would land inside the region the
        // memory covers, where we already know a byte is out of place, so jump
        // past that region instead.
        if (memory && shift < period) shift = m - period;
        memory = 0;
        j += shift;
        continue;
      }
      // The last byte matched (that is what shift == 0 means), so the right
      // half only has to be checked up to m - 1.
      size_t i = std::max(crit, memory);
      while (i + 1 < m && nd[i] == hay[i + j]) ++i;
      if (i + 1 < m) {
        j += i - crit + 1;
        memory = 0;
        continue;
      }
      size_t d = crit;
      while (d > memory && nd[d - 1] == hay[d - 1 + j]) --d;
      if (d <= memory) return {true, j};
      j += period;
      memory = m - period;
    }
  } else {
    while (j + m <= n) {
      size_t shift = tb.bad[hay[j + m - 1]];
      if (shift > 0) {
        j += shift;
        continue;
      }
      size_t i = crit;
      while (i + 1 < m && nd[i] == hay[i + j]) ++i;
      if (i + 1 < m) {
        j += i - crit + 1;
        continue;
      }
      size_t d = crit;
      while (d > 0 && nd[d - 1] == hay[d - 1 + j]) --d;
      if (d == 0) return {true, j};
      j += period;
    }
  }
  return {false, 0};
}

// Stateless two-way with bad-character skipping.
static inline std::pair<bool, size_t> twoway_bc_search(const char *text,
                                                       size_t n,
                                                       const char *pattern,
                                                       size_t m) {
  if (m == 0) return {true, 0};
  if (n < m) return {false, 0};
  twoway_bc_prep tb;
  tb.build(pattern, m);
  return twoway_bc_search_with(tb, text, n, pattern, m);
}
