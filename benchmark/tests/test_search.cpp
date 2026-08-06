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
// The empty needle is included: every searcher must agree with
// string_view::find(""), which is {true, 0} for any haystack including an empty
// one. It is the one convention every kernel has to special-case by hand, so it
// is exactly the kind of thing that drifts.
static void check(const NamedFn &nf, const std::string &text,
                  const std::string &pat) {
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
      {"memmem_search", memmem_search},
      {"bmh_search", bmh_search},
      {"bmh_search16", bmh_search16},
      {"kmp_search", kmp_search},
      {"twoway_search", twoway_search},
      {"twoway_bc_search", twoway_bc_search},
#if defined(SIMDSEARCH_AVX512)
      {"avx512_naive_search", avx512_naive_search},
      {"avx512_naive_search_v3", avx512_naive_search_v3},
      {"avx512_naive_search256_v3", avx512_naive_search256_v3},
      {"avx512_naive_search256", avx512_naive_search256},
      {"avx512_stringzilla_find", avx512_stringzilla_find},
      // The same kernel with the optional UTF-8 lead-byte anchor rule on. It
      // picks different anchors, so it exercises a different path through the
      // selector and must be validated separately.
      {"avx512_stringzilla_find_hifilter", avx512_stringzilla_find_hifilter},
      {"avx512_needle_hammer", avx512_needle_hammer},
      {"avx512_needle_hammer16", avx512_needle_hammer16},
      {"avx512_needle_hammer32", avx512_needle_hammer32},
      {"avx512_needle_hammer64", avx512_needle_hammer64},
      {"avx512_needle_hammer128", avx512_needle_hammer128},
      {"avx512_needle_hammer256", avx512_needle_hammer256},
      {"avx512_needle_hammer512", avx512_needle_hammer512},
      // The guarded variants must agree with everyone else on every input,
      // including the ones that make them abandon the filter for two-way --
      // the fallback path is only correct if it returns the same index.
      {"avx512_needle_hammer_guarded", avx512_needle_hammer_guarded},
      {"avx512_needle_hammer_guarded_tight", avx512_needle_hammer_guarded_tight},
      {"avx512_needle_hammer_guarded_loose", avx512_needle_hammer_guarded_loose},
      {"avx512_stringzilla256_find", avx512_stringzilla256_find},
      {"avx256_naive_search", avx256_naive_search},
      {"avx256_naive_search128", avx256_naive_search128},
      {"avx256_stringzilla_find", avx256_stringzilla_find},
      {"avx256_needle_hammer", avx256_needle_hammer},
      {"avx256_needle_hammer64", avx256_needle_hammer64},
      {"avx256_needle_hammer512", avx256_needle_hammer512},
      {"avx128_naive_search", avx128_naive_search},
      {"avx128_naive_search64", avx128_naive_search64},
      {"avx128_stringzilla_find", avx128_stringzilla_find},
      {"avx128_needle_hammer", avx128_needle_hammer},
      {"avx128_needle_hammer64", avx128_needle_hammer64},
      {"avx128_needle_hammer512", avx128_needle_hammer512},
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
      // 3 exercises the anchored kernel's no-verification path (anchors cover
      // the whole needle only for m <= 3); 63/64/65 straddle the register-width
      // boundary where it switches from one masked compare to sz_equal_avx512.
      // 15/16/17 and 31/32/33 straddle the m <= W register-compare boundary at
      // SSE2 and AVX2 width; 63/64/65 do the same for AVX-512.
      for (size_t mlen : {1u, 2u, 3u, 4u, 5u, 15u, 16u, 17u, 31u, 32u, 33u,
                          63u, 64u, 65u, 100u}) {
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

  // ---- Long-needle fuzz: needle lengths straddle needle-hammer's 512-byte switch
  // point, and the haystack is long enough (>= m + 255) that the 256-byte-
  // stride kernels run their main loop rather than falling straight through to
  // the remainder path. Without this needle-hammer's above-threshold branch and
  // avx512_stringzilla256_find's block loop are never taken.
  for (size_t plen : {size_t{200}, size_t{255}, size_t{256}, size_t{257},
                      size_t{511}, size_t{512}, size_t{513}, size_t{600},
                      size_t{1024}, size_t{1500}}) {
    for (int iter = 0; iter < 30; ++iter) {
      // Haystack sizes around each width's wide-block gate: the 512-bit
      // kernels need n >= m + 255, the 256-bit ones n >= m + 127 and the
      // 128-bit ones n >= m + 63. Straddling all three exercises both the
      // main block loop and the scalar remainder at every register width.
      for (size_t tlen : {plen, plen + 1, plen + 62, plen + 63, plen + 64,
                          plen + 126, plen + 127, plen + 128, plen + 254,
                          plen + 255, plen + 256, plen + 700, 4 * plen + 300}) {
        std::string text(tlen, '?');
        for (auto &c : text) c = alphabet[gen() % A];

        std::string pat;
        int kind = (int)(gen() % 3);
        if (kind == 0) {  // cut from the text: guaranteed present
          std::uniform_int_distribution<size_t> start_dist(0, tlen - plen);
          pat = text.substr(start_dist(gen), plen);
        } else if (kind == 1) {  // random: over "abc" it is effectively absent
          pat.resize(plen);
          for (auto &c : pat) c = alphabet[gen() % A];
        } else {  // present-but-for-one-byte: forces deep verification
          std::uniform_int_distribution<size_t> start_dist(0, tlen - plen);
          pat = text.substr(start_dist(gen), plen);
          pat[gen() % plen] = 'z';  // 'z' is outside the alphabet
        }

        for (auto &fn : fns) check(fn, text, pat);
      }
    }
  }

  // ---- Empty needle, explicitly ------------------------------------------
  for (const char *t : {"", "a", "abcabc"})
    for (auto &fn : fns) check(fn, std::string(t), std::string());

#if defined(SIMDSEARCH_AVX512)
  // ---- Budget exhausted, then two-way finds the match --------------------
  //
  // The fuzz sweep above can only reach the give-up path by accident, and on
  // inputs where the answer is "absent" -- so a guard that gave up and returned
  // {false, 0} without resuming would still pass. This case is deterministic and
  // the answer is a real match, so it fails if the resume is dropped, if it
  // resumes at the wrong offset, or if the budget stops firing at all.
  //
  // Haystack is all 'a' with the needle appended, needle is a^(m-1) then 'b'.
  // Every position survives the four-byte filter, so the wide kernel narrows
  // m-4 rounds per 256-byte block: n(m-4)/256 rounds against a budget of n/8,
  // which at m = 64 is 4x over. The single match sits at the very end, past the
  // give-up point, and only the two-way resume can find it.
  {
    const size_t m = 64, prefix = 20000;
    std::string needle(m - 1, 'a');
    needle += 'b';
    std::string hay(prefix, 'a');
    hay += needle;

    auto r = avx512_naive_search256_guarded(hay.data(), hay.size(),
                                            needle.data(), m, hay.size() / 8 + 1);
    ++g_checks;
    if (!r.gave_up) {
      std::printf("MISMATCH budget test: wide kernel did not exhaust its "
                  "budget (found=%d index=%zu)\n", (int)r.found, r.index);
      ++g_failures;
    }
    ++g_checks;
    if (r.gave_up && r.resume > prefix) {
      std::printf("MISMATCH budget test: resumed at %zu, past the match at "
                  "%zu\n", r.resume, prefix);
      ++g_failures;
    }
    // The scheme as a whole must still return the match.
    for (const NamedFn &nf : {NamedFn{"avx512_needle_hammer_guarded",
                                      avx512_needle_hammer_guarded},
                              NamedFn{"avx512_needle_hammer",
                                      avx512_needle_hammer}})
      check(nf, hay, needle);

    // Same shape with no match at all: the guard must report absence, not a
    // spurious hit, after giving up.
    std::string absent(prefix + m, 'a');
    for (const NamedFn &nf : {NamedFn{"avx512_needle_hammer_guarded",
                                      avx512_needle_hammer_guarded}})
      check(nf, absent, needle);
  }
#endif

  std::printf("ran %zu checks, %zu failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
