// Correctness tests for the SIMD and scalar string searchers.
//
// Every searcher is checked against a std::string::find reference across a
// battery of deterministic edge cases plus a large randomized fuzz sweep. The
// randomized inputs use a small alphabet so partial matches (shared prefixes,
// repeated anchor bytes) occur constantly, exercising the SIMD verify paths and
// early-out logic. The test is self-contained: it returns non-zero on the first
// mismatch so CTest reports a clean pass/fail.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
  #include "avx512search.h"
  #define SIMDSEARCH_AVX512 1
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #include "neonsearch.h"
  #define SIMDSEARCH_NEON 1
#else
  #error "No supported SIMD backend (need AVX-512 BW or ARM NEON)"
#endif

using search_fn = std::pair<bool, size_t> (*)(const char *, size_t,
                                              const char *, size_t);

struct NamedFn {
  const char *name;
  search_fn fn;
};

// Reference: first occurrence of pattern in text, via std::string::find.
static std::pair<bool, size_t> reference(const char *text, size_t n,
                                         const char *pattern, size_t m) {
  std::string_view sv(text, n);
  size_t r = sv.find(std::string_view(pattern, m));
  if (r == std::string_view::npos) return {false, 0};
  return {true, r};
}

static size_t g_failures = 0;
static size_t g_checks = 0;

// Compare one searcher against the reference for a single (text, pattern) pair.
// m == 0 is skipped: the searchers disagree on the empty-needle convention and
// the benchmark never passes an empty needle.
static void check(const NamedFn &nf, const std::string &text,
                  const std::string &pat) {
  if (pat.empty()) return;
  auto [rf, ri] = reference(text.data(), text.size(), pat.data(), pat.size());
  auto [gf, gi] = nf.fn(text.data(), text.size(), pat.data(), pat.size());
  ++g_checks;
  if (gf != rf || (rf && gi != ri)) {
    if (g_failures < 20) {
      std::printf(
          "MISMATCH %-26s text_len=%zu pat_len=%zu got={%d,%zu} ref={%d,%zu}\n",
          nf.name, text.size(), pat.size(), (int)gf, gi, (int)rf, ri);
    }
    ++g_failures;
  }
}

int main() {
  std::vector<NamedFn> fns = {
      {"bmh_search", bmh_search},
      {"bmh_search16", bmh_search16},
#if defined(SIMDSEARCH_AVX512)
      {"avx512_naive_search", avx512_naive_search},
      {"avx512_naive_search256", avx512_naive_search256},
      {"avx512_stringzilla_find", avx512_stringzilla_find},
#elif defined(SIMDSEARCH_NEON)
      {"neon_naive_search", neon_naive_search},
      {"neon_naive_search64", neon_naive_search64},
      {"neon_stringzilla_find", neon_stringzilla_find},
#endif
  };

  // ---- Deterministic edge cases ----
  {
    std::string base = "the quick brown fox jumps over the lazy dog";
    std::vector<std::string> pats = {
        "t", "g", "z", "the", "dog", "fox", "the quick", "lazy dog",
        base, base + "!", "cat", "quickx", " ", "  ", "oo", "the the"};
    for (auto &p : pats)
      for (auto &fn : fns) check(fn, base, p);

    // Pattern at every alignment boundary around 64 bytes (SIMD stride).
    std::string blob(200, 'a');
    for (size_t pos : {0u, 1u, 31u, 32u, 60u, 63u, 64u, 65u, 127u, 196u}) {
      std::string t = blob;
      std::string needle = "XYZW";
      if (pos + needle.size() <= t.size()) {
        t.replace(pos, needle.size(), needle);
        for (auto &fn : fns) check(fn, t, needle);
      }
    }

    // Long single-character runs with a needle of varying length, found and
    // not-found, to exercise both the wide stride and the scalar tail.
    for (size_t tlen : {1u, 7u, 63u, 64u, 65u, 256u, 257u, 1000u}) {
      std::string t(tlen, 'b');
      for (size_t mlen : {1u, 2u, 4u, 5u, 16u, 33u, 64u, 100u}) {
        if (mlen > tlen) continue;
        std::string found(mlen, 'b');
        std::string missing(mlen, 'b');
        missing.back() = 'q';  // identical except last byte
        for (auto &fn : fns) {
          check(fn, t, found);
          check(fn, t, missing);
        }
      }
    }

    // UTF-8 Cyrillic text: exercises vibrant-byte anchor selection (bytes > 191).
    // Source file is UTF-8; char literals carry the multibyte sequences as bytes.
    {
      std::string utf8 = "Привет мир и солнце";
      std::vector<std::string> pats = {"При", "вет", "мир", "солнце", "нет"};
      for (auto &p : pats)
        for (auto &fn : fns) check(fn, utf8, p);
    }

    // Needle longer than 8 bytes whose default anchors are UTF-8 continuation
    // bytes (>= 192); vibrant pivot must shift anchors before SIMD filtering.
    {
      std::string cont(12, '\x80');
      cont[0] = 'X';
      cont[6] = 'Z';
      cont[11] = 'Y';
      std::string hay(200, '\x80');
      hay.replace(50, cont.size(), cont);
      for (auto &fn : fns) check(fn, hay, cont);
      hay.replace(50, cont.size(), std::string(cont.size(), '\x80'));
      for (auto &fn : fns) check(fn, hay, cont);
    }

    // Short needles at SIMD boundaries (dedicated n_len == 2/3 paths).
    {
      std::string t(64, 'q');
      t[31] = 'a';
      t[32] = 'b';
      for (auto &fn : fns) {
        check(fn, t, "ab");
        check(fn, t, "abc");
        check(fn, t, "qb");
      }
    }
  }

  // ---- Randomized fuzz sweep over a small alphabet ----
  std::mt19937_64 gen(0xC0FFEE123456789ull);  // fixed seed: reproducible
  const char alphabet[] = "abc";               // tiny -> many partial matches
  const size_t A = sizeof(alphabet) - 1;

  for (int iter = 0; iter < 3000; ++iter) {
    std::uniform_int_distribution<size_t> tlen_dist(1, 400);
    size_t tlen = tlen_dist(gen);
    std::string text(tlen, '?');
    for (auto &c : text) c = alphabet[gen() % A];

    // Mix two needle sources: substrings cut from the text (guaranteed found)
    // and freshly random needles (usually absent for longer lengths).
    std::uniform_int_distribution<size_t> plen_dist(1, 40);
    size_t plen = std::min(plen_dist(gen), tlen);

    std::string pat;
    if (gen() & 1) {
      std::uniform_int_distribution<size_t> start_dist(0, tlen - plen);
      pat = text.substr(start_dist(gen), plen);
    } else {
      pat.resize(plen);
      for (auto &c : pat) c = alphabet[gen() % A];
    }

    for (auto &fn : fns) check(fn, text, pat);
  }

  // ---- Wider-byte fuzz: bytes in [0, 255] stress vibrant-byte pivoting ----
  for (int iter = 0; iter < 1000; ++iter) {
    std::uniform_int_distribution<size_t> tlen_dist(1, 400);
    size_t tlen = tlen_dist(gen);
    std::string text(tlen, '\0');
    for (auto &c : text) c = (char)(gen() & 0xFF);

    std::uniform_int_distribution<size_t> plen_dist(1, 40);
    size_t plen = std::min(plen_dist(gen), tlen);

    std::string pat;
    if (gen() & 1) {
      std::uniform_int_distribution<size_t> start_dist(0, tlen - plen);
      pat = text.substr(start_dist(gen), plen);
    } else {
      pat.resize(plen);
      for (auto &c : pat) c = (char)(gen() & 0xFF);
    }

    for (auto &fn : fns) check(fn, text, pat);
  }

  std::printf("ran %zu checks, %zu failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
