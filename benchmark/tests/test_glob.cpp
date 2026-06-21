// Correctness tests for the glob matchers in globmatch.h.
//
// Strategy:
//   * An independent, memoized oracle (oracle_match) defines the expected result
//     for star-only patterns ('*', '?', '\' escape and literals) using standard
//     wildcard semantics. The Redis baseline is validated against it on
//     non-empty strings (see the empty-string note below).
//   * Every barch matcher is cross-checked against the Redis baseline (the spec)
//     over a large randomized fuzz sweep plus hand-written edge cases.
//
// Two domains are fuzzed separately:
//   * Full alphabet (incl. '?'): validates the Redis baseline and barch's
//     general matcher (barch_general / barch_stringmatchlen general path).
//   * Fast-path domain ('*' + literals, no '?'): additionally validates barch's
//     asterisk fast path and the star-only dispatcher. This is the domain the
//     optimization targets and the only shape the glob_benchmark generates.
//
// Known upstream limitations of barch's asterisk fast path (deliberately NOT
// exercised against the baseline):
//   * it mishandles a '?' that immediately follows a '*' (its source carries
//     "TODO ?BUG?" markers); e.g. "*?aa" can match a string one char too short.
//     Such "*?" patterns are excluded from the fast-path fuzz.
//   * its memchr/memmem skip compares raw bytes, so it is case-sensitive only;
//     the fast path is therefore only validated with nocase == 0.
//
// Empty-string note: Redis's stringmatchlen only matches an empty string with an
// empty pattern (its outer loop requires a non-empty string), so e.g. "*" does
// NOT match "" under Redis even though standard glob would match. The oracle uses
// standard semantics, so oracle-vs-Redis comparison is skipped for empty strings;
// the empty-string behaviour is instead pinned by explicit edge cases below.
//
// Returns 0 on success, 1 on the first batch of mismatches.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "globmatch.h"

// Independent reference for star-only patterns: the classic wildcard DP, made
// O(plen*slen) by memoizing on (pattern index, string index). '*' matches zero
// or more characters, '?' exactly one, '\' escapes the next character.
static bool oracle_at(const std::string &p, const std::string &s, int nocase,
                      size_t pi, size_t si, std::vector<int8_t> &memo) {
  const size_t W = s.size() + 1;
  int8_t &m = memo[pi * W + si];
  if (m != -1) return m != 0;
  bool res;
  if (pi == p.size()) {
    res = (si == s.size());
  } else if (p[pi] == '*') {
    // consume zero pattern '*' chars, or one string char
    res = oracle_at(p, s, nocase, pi + 1, si, memo) ||
          (si < s.size() && oracle_at(p, s, nocase, pi, si + 1, memo));
  } else if (p[pi] == '?') {
    res = si < s.size() && oracle_at(p, s, nocase, pi + 1, si + 1, memo);
  } else {
    char pc = p[pi];
    size_t npi = pi + 1;
    if (p[pi] == '\\' && pi + 1 < p.size()) {  // escaped literal
      pc = p[pi + 1];
      npi = pi + 2;
    }
    bool eq = si < s.size() &&
              (nocase ? tolower((unsigned char)pc) == tolower((unsigned char)s[si])
                      : pc == s[si]);
    res = eq && oracle_at(p, s, nocase, npi, si + 1, memo);
  }
  m = res ? 1 : 0;
  return res;
}

static bool oracle_match(const std::string &p, const std::string &s,
                         int nocase) {
  std::vector<int8_t> memo((p.size() + 1) * (s.size() + 1), -1);
  return oracle_at(p, s, nocase, 0, 0, memo);
}

static int failures = 0;

static void check(int got, int expected, const std::string &pat,
                  const std::string &str, const char *who, int nocase) {
  if (got != expected) {
    std::printf("FAIL [%s] nocase=%d pattern=\"%s\" string=\"%s\": got %d, "
                "expected %d\n",
                who, nocase, pat.c_str(), str.c_str(), got, expected);
    if (++failures > 20) {
      std::printf("too many failures, aborting\n");
      std::exit(1);
    }
  }
}

// Validate matchers on (pat, str). `fast_path` requests that the asterisk fast
// path and the star-only dispatcher are also checked. They are only valid on a
// '?'-after-'*'-free, case-sensitive domain, so fast-path checks are suppressed
// when nocase != 0 (the memchr/memmem skip compares raw bytes).
static void verify(const std::string &pat, const std::string &str, int nocase,
                   bool fast_path) {
  int redis = globmatch::redis_stringmatchlen(pat.data(), pat.size(),
                                              str.data(), str.size(), nocase);

  // Oracle uses standard semantics; skip empty strings (see header note).
  if (!str.empty()) {
    int oracle = oracle_match(pat, str, nocase) ? 1 : 0;
    check(redis, oracle, pat, str, "glob_redis vs oracle", nocase);
  }

  // barch's general matcher must always track the Redis baseline.
  check(globmatch::barch_general(pat.data(), pat.size(), str.data(), str.size(),
                                 nocase),
        redis, pat, str, "glob_barch_general", nocase);

  if (fast_path && nocase == 0) {
    check(globmatch::barch_asterisk(pat.data(), pat.size(), str.data(),
                                    str.size(), nocase),
          redis, pat, str, "glob_barch_asterisk", nocase);
    check(globmatch::barch_stringmatchlen(pat.data(), pat.size(), str.data(),
                                          str.size(), nocase),
          redis, pat, str, "glob_barch", nocase);
  }
}

int main() {
  // ---- Hand-written edge cases (pattern, string, expected match) ----
  // Expected values follow Redis semantics (the spec).
  struct Case { const char *p; const char *s; int expect; };
  static const Case cases[] = {
      {"", "", 1},          {"", "x", 0},
      {"*", "", 0},  // Redis: only the empty pattern matches the empty string
      {"*", "anything", 1}, {"?", "", 0},
      {"?", "a", 1},        {"a", "a", 1},
      {"a", "b", 0},        {"a*", "abc", 1},
      {"a*", "xbc", 0},     {"*c", "abc", 1},
      {"*c", "abx", 0},     {"*bc*", "aXbcY", 1},
      {"*bc*", "aXbY", 0},  {"a?c", "abc", 1},
      {"a?c", "ac", 0},     {"*needle*", "hay needle hay", 1},
      {"*needle*", "haystack", 0}, {"**a**", "zzazz", 1},
      {"\\*", "*", 1},      {"\\*", "x", 0},
      {"a\\?c", "a?c", 1},  {"a\\?c", "abc", 0},
      {"*.txt", "file.txt", 1}, {"*.txt", "file.dat", 0},
      {"*a?b*", "xxayb zz", 1},  // '?' not directly after '*': fast path is fine
      {"*a?b*", "xxab zz", 0},
  };
  for (const auto &c : cases) {
    std::string p = c.p, s = c.s;
    int got = globmatch::redis_stringmatchlen(p.data(), p.size(), s.data(),
                                              s.size(), 0);
    check(got, c.expect, p, s, "edge glob_redis", 0);
    // None of these patterns put '?' directly after '*', so the fast path is
    // valid for all of them.
    verify(p, s, 0, /*fast_path=*/true);
  }

  // Case-insensitive spot checks.
  verify("*ABC*", "xx abc yy", 1, true);
  verify("*ABC*", "xx abc yy", 0, true);

  // ---- Randomized fuzz sweep ----
  std::mt19937 gen(0xC0FFEEu);  // fixed seed: deterministic test
  // Haystack alphabets: a tiny one stresses '*' backtracking; wider ones with
  // longer literal runs exercise the memchr/memmem fast paths.
  const std::string salphas[] = {"ab", "abc", "abcdef "};
  std::uniform_int_distribution<int> coin(0, 1);

  for (int iter = 0; iter < 300000; ++iter) {
    const std::string &salpha = salphas[iter % 3];
    // Half the rounds allow '?' (general-matcher domain); half are '?'-free and
    // additionally validate the asterisk fast path.
    bool fast_path = coin(gen);
    const std::string palpha = fast_path ? "abcdef *\\" : "abcdef *?\\";

    std::uniform_int_distribution<int> plen_d(0, 12);
    std::uniform_int_distribution<int> slen_d(0, 16);
    int pl = plen_d(gen), sl = slen_d(gen);

    std::string pat, str;
    std::uniform_int_distribution<size_t> pc(0, palpha.size() - 1);
    std::uniform_int_distribution<size_t> sc(0, salpha.size() - 1);
    for (int i = 0; i < pl; ++i) pat += palpha[pc(gen)];
    for (int i = 0; i < sl; ++i) str += salpha[sc(gen)];

    // Fast-path rounds run case-sensitively (the only domain the fast path is
    // valid on); general-matcher rounds toggle case sensitivity.
    int nocase = fast_path ? 0 : coin(gen);
    verify(pat, str, nocase, fast_path);
  }

  if (failures == 0) {
    std::printf("all glob matcher tests passed\n");
    return 0;
  }
  std::printf("%d failures\n", failures);
  return 1;
}
