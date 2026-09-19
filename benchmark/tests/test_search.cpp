// Correctness tests for the SIMD and scalar string searchers.
//
// Every searcher is checked against a std::string::find reference across a
// battery of deterministic edge cases plus a large randomized fuzz sweep. The
// randomized inputs use a small alphabet so partial matches (shared prefixes,
// repeated anchor bytes) occur constantly, exercising the SIMD verify paths and
// early-out logic. The test is self-contained: it returns non-zero on the first
// mismatch so CTest reports a clean pass/fail.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
  #include "avx512search.h"
  #include "needle_hammer.h"
  #define SIMDSEARCH_AVX512 1
#elif defined(__aarch64__) || defined(_M_ARM64)
  #include "neonsearch.h"
  #include "needle_hammer.h"
  #define SIMDSEARCH_NEON 1
#else
  #error "No SIMD backend: this project targets AVX-512 (F + BW) or AArch64 NEON."
#endif

// The backend-specific names the two structural tests at the bottom of this file
// need: the find-all enumerator, the guarded wide kernel, and the scheme with
// and without its work counter. Naming them once here keeps those tests written
// against the scheme rather than against one architecture.
#if defined(SIMDSEARCH_AVX512)
  #define SIMD_NAIVE_SEARCH_ALL avx512_naive_search_all
  #define SIMD_NEEDLE_HAMMER avx512_needle_hammer_unguarded
  #define SIMD_NEEDLE_HAMMER_GUARDED avx512_needle_hammer
#else
  #define SIMD_NAIVE_SEARCH_ALL neon_naive_search_all
  #define SIMD_NEEDLE_HAMMER neon_needle_hammer_unguarded
  #define SIMD_NEEDLE_HAMMER_GUARDED neon_needle_hammer
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
//
// Overloads take either std::string or raw (ptr, len) so alignment tests can
// pass a crafted misaligned pointer without copying into a fresh allocation.
static void check(const NamedFn &nf, const char *text, size_t n,
                  const char *pat, size_t m) {
  auto [rf, ri] = reference(text, n, pat, m);
  auto [gf, gi] = nf.fn(text, n, pat, m);
  ++g_checks;
  if (gf != rf || (rf && gi != ri)) {
    if (g_failures < 20) {
      std::printf(
          "MISMATCH %-26s text_len=%zu pat_len=%zu got={%d,%zu} ref={%d,%zu}\n",
          nf.name, n, m, (int)gf, gi, (int)rf, ri);
    }
    ++g_failures;
  }
}

static void check(const NamedFn &nf, const std::string &text,
                  const std::string &pat) {
  check(nf, text.data(), text.size(), pat.data(), pat.size());
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
      {"avx512_naive_search256", avx512_naive_search256},
      {"avx512_stringzilla_find", avx512_stringzilla_find},
      // The same kernel with the optional UTF-8 lead-byte anchor rule on. It
      // picks different anchors, so it exercises a different path through the
      // selector and must be validated separately.
      {"avx512_stringzilla_find_hifilter", avx512_stringzilla_find_hifilter},
      // Needle-Hammer must agree with everyone else on every input, including
      // the ones that make it abandon the filter for two-way -- the fallback
      // path is only correct if it returns the same index -- and so must the
      // same kernel with the counter compiled out.
      {"avx512_needle_hammer", avx512_needle_hammer},
      {"avx512_needle_hammer_unguarded", avx512_needle_hammer_unguarded},
      {"twoway_simd_search", twoway_simd_search},
      {"avx512_stringzilla256_find", avx512_stringzilla256_find},
      {"avx256_naive_search", avx256_naive_search},
      {"avx256_naive_search128", avx256_naive_search128},
      {"avx256_stringzilla_find", avx256_stringzilla_find},
      {"avx128_naive_search", avx128_naive_search},
      {"avx128_naive_search64", avx128_naive_search64},
      {"avx128_stringzilla_find", avx128_stringzilla_find},
#endif
#if defined(SIMDSEARCH_NEON)
      {"neon_naive_search", neon_naive_search},
      {"neon_naive_search64", neon_naive_search64},
      {"neon_stringzilla_find", neon_stringzilla_find},
      {"neon_needle_hammer", neon_needle_hammer},
      {"neon_needle_hammer_unguarded", neon_needle_hammer_unguarded},
      {"twoway_simd_search", twoway_simd_search},
      // The same kernel with the optional UTF-8 lead-byte anchor rule on. It
      // picks different anchors, so it exercises a different path through the
      // selector and must be validated separately.
      {"neon_stringzilla_find_hifilter", neon_stringzilla_find_hifilter},
      {"neon_stringzilla64_find", neon_stringzilla64_find},
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

  // ---- Haystack alignment -------------------------------------------------
  //
  // The kernels walk the scan pointer to a 64-byte boundary and cover the
  // positions below it with one masked window. That head is a distinct code
  // path taken only when the buffer is misaligned, and a std::string happens to
  // land wherever the allocator puts it, so the fuzz sweep above exercises one
  // arbitrary alignment rather than all of them. Here we place the same text at
  // every offset in a 64-byte window and call the kernels on that pointer
  // directly -- copying into std::string would re-allocate and lose the
  // crafted misalignment.
  {
    std::vector<char> raw(4096 + 128);
    char *page = raw.data() + (64 - (reinterpret_cast<uintptr_t>(raw.data()) & 63));
    std::mt19937 g(99);
    const char alpha[] = "abc";
    for (size_t off = 0; off < 64; ++off) {
      char *hay = page + off;
      const size_t hlen = 2048;
      for (size_t i = 0; i < hlen; ++i) hay[i] = alpha[g() % 3];
      for (size_t plen : {4u, 5u, 8u, 17u, 64u, 100u}) {
        // present, at a position that straddles the alignment head
        for (size_t at : {size_t(0), size_t(1), size_t(37), size_t(63), size_t(64),
                          size_t(300)}) {
          if (at + plen > hlen) continue;
          for (auto &fn : fns) check(fn, hay, hlen, hay + at, plen);
        }
        // absent
        std::string absent(plen, 'z');
        for (auto &fn : fns) check(fn, hay, hlen, absent.data(), absent.size());
      }
    }
  }

  // ---- Empty needle, explicitly ------------------------------------------
  for (const char *t : {"", "a", "abcabc"})
    for (auto &fn : fns) check(fn, std::string(t), std::string());

  // ---- Find-all enumerator ------------------------------------------------
  //
  // Overlapping occurrences, short needles, and empty-needle convention must
  // agree with a pos+1 walk of the first-match kernel. ctest only runs this
  // file; findall benchmark mode is optional and easy to skip.
  {
    auto ref_findall = [](const char *t, size_t n, const char *p, size_t m) {
      std::vector<size_t> idx;
      if (m == 0) {
        for (size_t i = 0; i <= n; ++i) idx.push_back(i);
        return idx;
      }
      size_t pos = 0;
      while (pos + m <= n) {
        auto [f, i] = reference(t + pos, n - pos, p, m);
        if (!f) break;
        idx.push_back(pos + i);
        pos += i + 1;
      }
      return idx;
    };
    auto check_all = [&](const char *t, size_t n, const char *p, size_t m) {
      std::vector<size_t> got;
      SIMD_NAIVE_SEARCH_ALL(t, n, p, m, [&](size_t i) { got.push_back(i); });
      auto exp = ref_findall(t, n, p, m);
      ++g_checks;
      if (got != exp) {
        if (g_failures < 20) {
          std::printf("MISMATCH naive_search_all text_len=%zu pat_len=%zu "
                      "got %zu hits ref %zu hits\n",
                      n, m, got.size(), exp.size());
        }
        ++g_failures;
      }
    };

    // Dense small-alphabet: many overlapping matches.
    {
      std::string t(200, 'a');
      for (size_t m : {1u, 2u, 3u, 4u, 5u, 8u, 17u, 64u}) {
        std::string p(m, 'a');
        check_all(t.data(), t.size(), p.data(), p.size());
      }
      std::string miss(4, 'b');
      check_all(t.data(), t.size(), miss.data(), miss.size());
    }
    // Pattern placed near SIMD block tails and remainder.
    {
      std::string t(130, 'x');
      t.replace(60, 4, "abcd");
      t.replace(120, 4, "abcd");
      check_all(t.data(), t.size(), "abcd", 4);
      check_all(t.data(), t.size(), "ab", 2);
      check_all(t.data(), t.size(), "x", 1);
    }
    // Empty needle: indices 0..n (n+1 matches), matching the first-match loop
    // baseline (advance one byte after each {true, 0}).
    for (const char *s : {"", "a", "abc"})
      check_all(s, std::strlen(s), "", 0);
  }

  // ---- Budget exhausted, then two-way finds the match --------------------
  //
  // The fuzz sweep above can only reach the give-up path by accident, and on
  // inputs where the answer is "absent" -- so a guard that gave up and returned
  // {false, 0} without resuming would still pass. This case is deterministic and
  // the answer is a real match, so it fails if the resume is dropped, if it
  // resumes at the wrong offset, or if the budget stops firing at all.
  //
  // The input has to defeat the kernel's filter, which anchors on any byte
  // rare in the needle: the block shape, an all-'a' needle over a haystack of
  // (a^(m-1) b) repeated, offers no distinctive byte at all. A run of m 'a's
  // appended is the only match. The narrowing exceeds the budget long before
  // it, and only the two-way resume can find it.
  {
    const size_t m = 64, prefix = 20000;
    std::string needle(m, 'a');
    std::string hay(prefix, 'a');
    for (size_t i = 0; i < prefix; ++i) if (i % m == m - 1) hay[i] = 'b';
    hay += needle;
    const size_t match = hay.find(needle);
    const size_t budget = hay.size() / needle_hammer::kBudgetDen + 1;
    needle_hammer::anchors a = needle_hammer::select(needle.data(), m);
    needle_hammer::result r = a.k == 3
        ? needle_hammer::wide<true, 3>(hay.data(), hay.size(), needle.data(), m, a, budget, 0, 0)
        : needle_hammer::wide<true, 4>(hay.data(), hay.size(), needle.data(), m, a, budget, 0, 0);
    if (r.state == 2) {
      // Three anchors found the survivors too frequent, as the dispatcher
      // would: continue with four from where it stopped.
      a.k = 4;
      std::sort(a.o, a.o + 4);
      r = needle_hammer::wide<true, 4>(hay.data(), hay.size(), needle.data(), m, a,
                                       budget, r.rounds, r.resume);
    }
    const bool gave_up = (r.state == 1);
    ++g_checks;
    if (!gave_up) {
      std::printf("MISMATCH budget test: wide kernel did not exhaust its "
                  "budget (found=%d index=%zu)\n", (int)r.found, r.index);
      ++g_failures;
    }
    ++g_checks;
    if (gave_up && r.resume > match) {
      std::printf("MISMATCH budget test: resumed at %zu, past the match at "
                  "%zu\n", r.resume, match);
      ++g_failures;
    }
    // The scheme as a whole must still return the match.
    for (const NamedFn &nf : {NamedFn{"needle_hammer_guarded",
                                      SIMD_NEEDLE_HAMMER_GUARDED},
                              NamedFn{"needle_hammer", SIMD_NEEDLE_HAMMER}})
      check(nf, hay, needle);

    // Same shape with no match at all: the guard must report absence, not a
    // spurious hit, after giving up.
    std::string absent = hay.substr(0, prefix);
    for (const NamedFn &nf : {NamedFn{"needle_hammer_guarded",
                                      SIMD_NEEDLE_HAMMER_GUARDED}})
      check(nf, absent, needle);
  }

  // ---- One near-match per block: the verification is charged -------------
  //
  // Needle (a^255 b)^k, haystack the needle with its second-to-last byte
  // changed, tiled. Exactly one position per 256-byte block passes the
  // anchors, and its in-place verification matches the first 64 bytes and
  // runs deep before failing; no narrowing round ever runs. The verification
  // must be charged to the budget, so that the guarded kernel gives up on
  // this input instead of comparing m bytes per block for the whole
  // haystack. Driven as the dispatcher drives it: escalation, then a fresh
  // budget with four anchors, then the give-up.
  {
    const size_t m = 2048, n = 1 << 20;
    std::string needle(m, 'a');
    for (size_t i = 0; i < m; ++i) needle[i] = (i % 256 == 255) ? 'b' : 'a';
    std::string q = needle; q[m - 2] = 'c';
    std::string hay(n, 'a');
    for (size_t i = 0; i < n; ++i) hay[i] = q[i % m];
    const size_t budget = n / needle_hammer::kBudgetDen + 1;
    needle_hammer::anchors a = needle_hammer::select(needle.data(), m);
    auto run = [&](const needle_hammer::anchors &an, size_t carried, size_t from) {
      switch (an.k) {
        case 2: return needle_hammer::wide<true, 2>(hay.data(), n, needle.data(), m, an, budget, carried, from);
        case 3: return needle_hammer::wide<true, 3>(hay.data(), n, needle.data(), m, an, budget, carried, from);
        default: return needle_hammer::wide<true, 4>(hay.data(), n, needle.data(), m, an, budget, carried, from);
      }
    };
    needle_hammer::result r = run(a, 0, 0);
    while (r.state != 0 && a.k < 4) {
      if (r.state == 1) { a.k = 4; std::sort(a.o, a.o + 4); r = run(a, 0, r.resume); }
      else { a = needle_hammer::escalate(a); r = run(a, r.rounds, r.resume); }
    }
    ++g_checks;
    if (r.state != 1) {
      std::printf("MISMATCH near-match test: the guarded kernel did not give up "
                  "(state=%d found=%d rounds=%zu)\n", r.state, (int)r.found, r.rounds);
      ++g_failures;
    }
    ++g_checks;
    if (r.state == 1 && r.rounds <= budget) {
      std::printf("MISMATCH near-match test: gave up with %zu rounds under a budget of %zu\n",
                  r.rounds, budget);
      ++g_failures;
    }
    for (const NamedFn &nf : {NamedFn{"needle_hammer_guarded", SIMD_NEEDLE_HAMMER_GUARDED},
                              NamedFn{"needle_hammer", SIMD_NEEDLE_HAMMER}})
      check(nf, hay, needle);
  }

  std::printf("ran %zu checks, %zu failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
