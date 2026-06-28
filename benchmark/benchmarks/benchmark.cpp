#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "counters/bench.h"

// Pick the SIMD backend for the host architecture. Each header is
// self-contained (it also provides the portable scalar/library searchers), so
// exactly one is included per build.
#if defined(__AVX512F__) && defined(__AVX512BW__)
  #include "avx512search.h"
  #define SIMDSEARCH_AVX512 1
  #define SIMD_NAIVE_SEARCH avx512_naive_search
  #define SIMD_NAIVE_SEARCH_ALL avx512_naive_search_all
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #include "neonsearch.h"
  #define SIMDSEARCH_NEON 1
  #define SIMD_NAIVE_SEARCH neon_naive_search
  #define SIMD_NAIVE_SEARCH_ALL neon_naive_search_all
#else
  #error "No supported SIMD backend (need AVX-512 BW or ARM NEON)"
#endif

double pretty_print(const std::string &name, size_t num_values,
                    counters::event_aggregate agg) {
  std::print("{:<50} : ", name);
  std::print(" {:5.3f} ns ", agg.fastest_elapsed_ns() / double(num_values));
  std::print(" {:5.2f} Mv/s ", double(num_values) * 1000.0 / agg.fastest_elapsed_ns());
  if (counters::has_performance_counters()) {
    std::print(" {:5.2f} GHz ", agg.cycles() / double(agg.elapsed_ns()));
    std::print(" {:5.2f} c ", agg.fastest_cycles() / double(num_values));
    std::print(" {:5.2f} i ", agg.fastest_instructions() / double(num_values));
    std::print(" {:5.2f} m ", agg.fastest_branch_misses() / double(num_values));
    std::print(" {:5.2f} i/c ",
               agg.fastest_instructions() / double(agg.fastest_cycles()));

  }
  std::print("\n");
  return double(num_values) / agg.fastest_elapsed_ns();
}

std::string generate_random_string(size_t size = 4096) {
    // Use a high-quality random number generator
    static std::mt19937 gen(std::random_device{}());

    // Printable ASCII characters: space to '~'
    static const std::string chars =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        " ";

    std::uniform_int_distribution<size_t> dist(0, chars.size() - 1);

    std::string result;
    result.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        result += chars[dist(gen)];
    }

    return result;
}


std::vector<std::string> extract_random_substrings(
    const std::string& source,
    size_t count = 200,          // number of substrings to generate
    size_t min_len = 8,
    size_t max_len = 20) {
    if (source.size() < min_len) {
        return {};
    }

    static std::mt19937 gen(std::random_device{}());

    std::uniform_int_distribution<size_t> length_dist(min_len, max_len);

    std::vector<std::string> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        size_t len = length_dist(gen);
        size_t max_start = source.size() - len;

        std::uniform_int_distribution<size_t> start_dist(0, max_start);
        size_t start = start_dist(gen);

        result.emplace_back(source.substr(start, len));
    }

    return result;
}

// std::string::find wrapped in the uniform {found, index} interface so it can
// sit in the algorithm table alongside the others.
static std::pair<bool, size_t> classic_find(const char *text, size_t n,
                                             const char *pattern, size_t m) {
  std::string_view sv(text, n);
  size_t r = sv.find(std::string_view(pattern, m));
  if (r == std::string_view::npos) return {false, 0};
  return {true, r};
}

using search_fn = std::pair<bool, size_t> (*)(const char *, size_t,
                                              const char *, size_t);

// The three std searchers come in a non-amortized form (rebuilt on every call,
// done via the plain function pointer) and an amortized form (one searcher
// pre-built per needle and reused). The amortized kinds index into AmortState.
enum class Kind { Stateless, AmortDef, AmortBM, AmortBMH };

struct Algo {
  const char *name;
  Kind kind;
  search_fn fn;  // used only when kind == Stateless
};

// The single source of truth for which algorithms exist and how they are named.
// Every mode (synthetic / horspool / ashvardanian) iterates this exact list, so
// the algorithm set and the labels are identical across all benchmarks.
static const std::vector<Algo> kAlgos = {
    {"find_classic", Kind::Stateless, classic_find},
#if defined(SIMDSEARCH_AVX512)
    {"find_avx512", Kind::Stateless, avx512_naive_search},
    {"find_avx512_256", Kind::Stateless, avx512_naive_search256},
    {"find_avx512_stringzilla", Kind::Stateless, avx512_stringzilla_find},
#elif defined(SIMDSEARCH_NEON)
    {"find_neon", Kind::Stateless, neon_naive_search},
    {"find_neon64", Kind::Stateless, neon_naive_search64},
    {"find_neon_stringzilla", Kind::Stateless, neon_stringzilla_find},
#endif
    {"find_bmh", Kind::Stateless, bmh_search},
    {"find_bmh16", Kind::Stateless, bmh_search16},
    {"find_strstr", Kind::Stateless, strstr_search},
    {"find_std_default_searcher", Kind::Stateless, std_default_searcher},
    {"find_std_boyer_moore_searcher", Kind::Stateless, std_boyer_moore_searcher},
    {"find_std_boyer_moore_horspool_searcher", Kind::Stateless,
     std_boyer_moore_horspool_searcher},
    {"find_std_default_searcher_amortized", Kind::AmortDef, nullptr},
    {"find_std_boyer_moore_searcher_amortized", Kind::AmortBM, nullptr},
    {"find_std_boyer_moore_horspool_searcher_amortized", Kind::AmortBMH, nullptr},
};

// Pre-built searcher objects, one per needle, for the amortized variants. The
// searchers hold iterators into the needle storage, so the needle vector passed
// to prepare() must outlive this state and must not be modified afterwards.
struct AmortState {
  std::vector<std::default_searcher<const char *>> def;
  std::vector<std::boyer_moore_searcher<const char *>> bm;
  std::vector<std::boyer_moore_horspool_searcher<const char *>> bmh;

  void prepare(const std::vector<std::string> &needles) {
    def.clear();
    bm.clear();
    bmh.clear();
    def.reserve(needles.size());
    bm.reserve(needles.size());
    bmh.reserve(needles.size());
    for (const auto &s : needles) {
      def.emplace_back(s.data(), s.data() + s.size());
      bm.emplace_back(s.data(), s.data() + s.size());
      bmh.emplace_back(s.data(), s.data() + s.size());
    }
  }
};

// First-occurrence search of needle #id within [text, text+n) for any algorithm.
// Stateless kinds go through the function pointer (rebuilding any preprocessing
// each call); amortized kinds reuse the pre-built searcher for that needle.
static inline std::pair<bool, size_t> do_find(const Algo &a,
                                              const AmortState &am, size_t id,
                                              const char *text, size_t n,
                                              const char *pat, size_t m) {
  switch (a.kind) {
    case Kind::Stateless:
      return a.fn(text, n, pat, m);
    case Kind::AmortDef: {
      const char *e = text + n;
      auto it = std::search(text, e, am.def[id]);
      return it == e ? std::pair<bool, size_t>{false, 0}
                     : std::pair<bool, size_t>{true, (size_t)(it - text)};
    }
    case Kind::AmortBM: {
      const char *e = text + n;
      auto it = std::search(text, e, am.bm[id]);
      return it == e ? std::pair<bool, size_t>{false, 0}
                     : std::pair<bool, size_t>{true, (size_t)(it - text)};
    }
    case Kind::AmortBMH: {
      const char *e = text + n;
      auto it = std::search(text, e, am.bmh[id]);
      return it == e ? std::pair<bool, size_t>{false, 0}
                     : std::pair<bool, size_t>{true, (size_t)(it - text)};
    }
  }
  return {false, 0};
}

// True if any selected algorithm uses pre-built (amortized) searchers, so the
// AmortState build cost is only paid when it is actually needed.
static bool needs_amort(const std::vector<const Algo *> &algos) {
  for (const Algo *a : algos)
    if (a->kind != Kind::Stateless) return true;
  return false;
}

// Count all non-overlapping occurrences of needle #id in haystack by repeatedly
// searching the remaining suffix and advancing past each match (StringWars'
// forward find-all loop).
static size_t count_all(const Algo &a, const AmortState &am, size_t id,
                        const std::string &hay, const std::string &needle) {
  size_t pos = 0, cnt = 0, m = needle.size();
  while (pos + m <= hay.size()) {
    auto [f, idx] = do_find(a, am, id, hay.data() + pos, hay.size() - pos,
                            needle.data(), m);
    if (!f) break;
    pos += idx + m;
    ++cnt;
  }
  return cnt;
}

std::string load_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Cannot open file: " << path << "\n";
    exit(1);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Split into whitespace-delimited tokens (space, newline, CR), dropping empties
// — the StringWars "words" tokenization. Tokens are copied into std::string so
// they are NUL-terminated (required by strstr and harmless for the rest).
std::vector<std::string> tokenize_words(const std::string &s) {
  std::vector<std::string> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r')) ++i;
    size_t start = i;
    while (i < n && !(s[i] == ' ' || s[i] == '\n' || s[i] == '\r')) ++i;
    if (i > start) out.emplace_back(s.substr(start, i - start));
  }
  return out;
}

// Synthetic mode: random 1KB text, many short random needles, first-occurrence
// search timed per needle.
void collect_benchmark_results(size_t input_size, size_t number_strings,
                               const std::vector<const Algo *> &algos) {
  std::string source = generate_random_string(input_size);
  auto strings = extract_random_substrings(source, number_strings);

  AmortState am;
  if (needs_amort(algos)) am.prepare(strings);

  // Validate every selected algorithm against std::string::find before timing.
  for (size_t id = 0; id < strings.size(); ++id) {
    const auto &str = strings[id];
    size_t p = source.find(str);
    if (p == std::string::npos) {
      std::cerr << "Error: reference substring not found\n";
      exit(1);
    }
    for (const Algo *a : algos) {
      auto [f, idx] = do_find(*a, am, id, source.data(), source.size(),
                              str.data(), str.size());
      if (!f || idx != p) {
        std::cerr << "Error: " << a->name << " index mismatch (got " << idx
                  << ", expected " << p << ")\n";
        exit(1);
      }
    }
  }

  volatile uint64_t counter = 0;
  for (const Algo *a : algos) {
    auto run = [&]() {
      size_t c = 0;
      for (size_t id = 0; id < strings.size(); ++id) {
        const auto &str = strings[id];
        auto [f, idx] = do_find(*a, am, id, source.data(), source.size(),
                                str.data(), str.size());
        if (f) c += idx;
      }
      counter += c;
    };
    pretty_print(a->name, number_strings, counters::bench(run));
  }
}

// Look up an algorithm by its exact name in kAlgos; nullptr if not found.
static const Algo *find_algo_by_name(const std::string &name) {
  for (const auto &a : kAlgos)
    if (name == a.name) return &a;
  return nullptr;
}

// Split a comma-separated list, dropping empty fields.
static std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string cur;
  while (std::getline(ss, cur, ',')) {
    if (!cur.empty()) out.push_back(cur);
  }
  return out;
}

void list_algos() {
  std::print("available algorithms:\n");
  for (const auto &a : kAlgos) std::print("  {}\n", a.name);
}

// Horspool mode: draw 2000 random substrings of the data file at each requested
// length and time first-occurrence search for each selected algorithm. Patterns
// are cut from the text itself, so each is guaranteed to be found. The table has
// one row per algorithm and one column per length, so reading across a row shows
// how that algorithm's cost evolves with pattern length.
void horspool_benchmark(const std::string &text, const std::string &source_desc,
                        const std::vector<size_t> &requested_lengths,
                        const std::vector<const Algo *> &algos) {
  static std::mt19937 gen(std::random_device{}());
  const size_t patterns_per_len = 2000;
  volatile uint64_t sink = 0;

  // Keep only lengths that fit the text.
  std::vector<size_t> lengths;
  for (size_t L : requested_lengths) {
    if (L < 1) {
      std::print(stderr, "horspool: skipping invalid length 0\n");
    } else if (L > text.size()) {
      std::print(stderr, "horspool: skipping length {} (> text size {})\n", L,
                 text.size());
    } else {
      lengths.push_back(L);
    }
  }
  if (lengths.empty()) {
    std::cerr << "horspool: no usable lengths\n";
    exit(1);
  }

  // ns[algo_index][length_index]
  std::vector<std::vector<double>> ns(
      algos.size(), std::vector<double>(lengths.size(), 0.0));

  for (size_t li = 0; li < lengths.size(); ++li) {
    size_t L = lengths[li];
    std::vector<std::string> pats;
    pats.reserve(patterns_per_len);
    std::uniform_int_distribution<size_t> start_dist(0, text.size() - L);
    for (size_t i = 0; i < patterns_per_len; ++i)
      pats.emplace_back(text.substr(start_dist(gen), L));

    AmortState am;
    if (needs_amort(algos)) am.prepare(pats);

    // Validate every selected algorithm against std::string::find before timing.
    for (size_t id = 0; id < pats.size(); ++id) {
      size_t ref = text.find(pats[id]);
      for (const Algo *a : algos) {
        auto [f, idx] = do_find(*a, am, id, text.data(), text.size(),
                                pats[id].data(), pats[id].size());
        if (!f || idx != ref) {
          std::cerr << "Error: horspool mismatch in " << a->name << " at length "
                    << L << " (got " << idx << ", expected " << ref << ")\n";
          exit(1);
        }
      }
    }

    for (size_t ai = 0; ai < algos.size(); ++ai) {
      const Algo &a = *algos[ai];
      auto run = [&]() {
        size_t c = 0;
        for (size_t id = 0; id < pats.size(); ++id) {
          auto [f, idx] = do_find(a, am, id, text.data(), text.size(),
                                  pats[id].data(), pats[id].size());
          if (f) c += idx;
        }
        sink += c;
      };
      auto agg = counters::bench(run);
      ns[ai][li] = agg.fastest_elapsed_ns() / double(pats.size());
    }
    std::print(stderr, "horspool: length {} done\n", L);
  }

  std::print("# horspool mode\n");
  std::print("source: {}\n", source_desc);
  std::print("text size: {} bytes, {} random patterns per length\n", text.size(),
             patterns_per_len);
  std::print("values are ns per first-occurrence search (lower is better)\n");
  std::print("rows = algorithm, columns = pattern length\n\n");

  std::print("{:<48}", "algo");
  for (size_t L : lengths) std::print(" {:>10}", L);
  std::print("\n");
  for (size_t ai = 0; ai < algos.size(); ++ai) {
    std::print("{:<48}", algos[ai]->name);
    for (size_t li = 0; li < lengths.size(); ++li)
      std::print(" {:>10.1f}", ns[ai][li]);
    std::print("\n");
  }
}

// Ashvardanian mode: emulates StringWars' substring-search benchmark. The whole
// file is the haystack; its word tokens are the needles (every needle is thus
// present). Each timed pass scans the full haystack for every needle, locating
// all non-overlapping occurrences. Throughput is reported as GB/s of haystack
// scanned, crediting one full haystack length per needle (the StringWars
// convention), plus ns per needle.
void ashvardanian_benchmark(const std::string &hay,
                            const std::vector<const Algo *> &algos) {
  const size_t max_needles = 1000;  // keep one pass quick; StringWars cycles all
  auto needles = tokenize_words(hay);
  if (needles.size() > max_needles) needles.resize(max_needles);
  if (needles.empty()) {
    std::cerr << "No word tokens found in input.\n";
    exit(1);
  }

  AmortState am;
  if (needs_amort(algos)) am.prepare(needles);

  std::print("# ashvardanian mode (StringWars-style forward find-all)\n");
  std::print("haystack: {} bytes, {} word needles (capped at {})\n", hay.size(),
             needles.size(), max_needles);

  // Reference match count via std::string::find (kAlgos[0]), always available.
  size_t ref_total = 0;
  for (size_t id = 0; id < needles.size(); ++id)
    ref_total += count_all(kAlgos[0], am, id, hay, needles[id]);
  std::print("total non-overlapping matches across needles: {}\n\n", ref_total);

  std::print("{:<48} {:>10} {:>14} {:>12}\n", "algo", "GB/s", "ns/needle",
             "matches");
  volatile uint64_t sink = 0;
  for (const Algo *a : algos) {
    // Validate match count against the reference before timing.
    size_t total = 0;
    for (size_t id = 0; id < needles.size(); ++id)
      total += count_all(*a, am, id, hay, needles[id]);
    if (total != ref_total) {
      std::cerr << "Error: ashvardanian match-count mismatch in " << a->name
                << " (" << total << " vs " << ref_total << ")\n";
      exit(1);
    }

    auto run = [&]() {
      size_t c = 0;
      for (size_t id = 0; id < needles.size(); ++id) {
        const auto &nd = needles[id];
        size_t pos = 0, m = nd.size();
        while (pos + m <= hay.size()) {
          auto [f, idx] = do_find(*a, am, id, hay.data() + pos,
                                  hay.size() - pos, nd.data(), m);
          if (!f) break;
          pos += idx + m;
          c += pos;
        }
      }
      sink += c;
    };
    auto agg = counters::bench(run);
    double ns_total = agg.fastest_elapsed_ns();
    double gbps = double(needles.size()) * double(hay.size()) / ns_total;
    double ns_per_needle = ns_total / double(needles.size());
    std::print("{:<48} {:>10.3f} {:>14.1f} {:>12}\n", a->name, gbps,
               ns_per_needle, ref_total);
  }
}

// Build the worst-case needle of length L for a given shape, over a haystack of
// all 'a'. Every shape contains exactly one byte that is absent from the
// haystack, so the needle never matches and the search always runs to
// completion. What differs is WHERE that byte sits, which decides which filters
// it defeats:
//   tail : (L-1)*'a' + 'b'          — odd byte last; beats prefix/first-byte
//                                      filters but StringZilla anchors on it.
//   aba  : (L-2)*'a' + 'b' + 'a'    — odd byte at L-2; beats a first+last
//                                      (2-anchor) filter, but StringZilla's
//                                      middle anchor + anomaly search finds it.
//   mid  : 'a'*L with [L/2] = 'b'   — odd byte mid; StringZilla anchors on it.
//   high : 'a'*L with [L/2] = 0xFF  — odd byte is >= 191, which the anomaly
//                                      selector tries to avoid; in practice it
//                                      still anchors on it, so StringZilla
//                                      survives. (Kept to show that even a high
//                                      byte does not hide from the selector.)
//   ab   : alternating 'abab...' with the middle byte flipped into a 3-run.
//                                      Paired with an 'abab...' haystack: every
//                                      byte value ('a','b') is common, so NO
//                                      anchor is selective. Each anchor matches
//                                      ~half the positions, so the filter can
//                                      no longer reject in bulk and StringZilla
//                                      itself is forced into O(n*m). This is the
//                                      shape that actually defeats StringZilla.
static std::string make_worstcase_needle(const std::string &shape, size_t L) {
  std::string nd(L, 'a');
  if (shape == "tail") nd[L - 1] = 'b';
  else if (shape == "aba") nd[L - 2] = 'b';
  else if (shape == "mid") nd[L / 2] = 'b';
  else if (shape == "high") nd[L / 2] = (char)0xFF;
  else if (shape == "ab") {
    for (size_t i = 0; i < L; ++i) nd[i] = (i % 2 == 0) ? 'a' : 'b';
    // Flip the middle byte to its neighbours' value: that makes a 3-byte run,
    // so the needle is non-alternating and never occurs in 'abab...'.
    nd[L / 2] = (L / 2 % 2 == 0) ? 'b' : 'a';
  }
  else if (shape == "block") { /* all 'a' — no distinctive byte at all */ }
  else { std::cerr << "unknown --needle shape: " << shape << "\n"; exit(1); }
  return nd;
}

// The haystack for a given (shape, needle length L):
//   ab    : 'abab...'  — periodic two-symbol text.
//   block : 'a'*(L-1) + 'b' repeated — runs of 'a' are only L-1 long, so the
//           all-'a' needle of length L never completes. Crucially the needle
//           has NO distinctive byte, so StringZilla's anchors are all 'a' (all
//           common here): the filter passes nearly everywhere and each
//           candidate is verified ~L/2 bytes deep before hitting a 'b'. This is
//           what forces StringZilla itself into O(n*L).
//   others: all 'a'.
static std::string make_worstcase_haystack(const std::string &shape, size_t n,
                                            size_t L) {
  std::string hay(n, 'a');
  if (shape == "ab")
    for (size_t i = 0; i < n; ++i) hay[i] = (i % 2 == 0) ? 'a' : 'b';
  else if (shape == "block")
    for (size_t i = 0; i < n; ++i) hay[i] = (i % L == L - 1) ? 'b' : 'a';
  return hay;
}

// Worst-case mode: the haystack is N copies of 'a' and each needle (see
// make_worstcase_needle) shares that all-'a' content except for a single odd
// byte, so the search runs to completion without matching. Depending on where
// the odd byte sits, the all-'a' prefix forces prefix/first-byte filters (the
// naive AVX-512 kernels, find_classic) and Boyer-Moore-Horspool (whose only
// bad-character shift here is 1) into O(n*m) verification. StringZilla survives
// every shape whose odd byte its anomaly selector can anchor on; the "high"
// shape is the one that evades it. One full-haystack search is timed per cell.
void worstcase_benchmark(size_t haystack_size, const std::string &needle_shape,
                         const std::vector<size_t> &requested_lengths,
                         const std::vector<const Algo *> &algos) {
  volatile uint64_t sink = 0;

  size_t min_len = (needle_shape == "tail") ? 2 : 3;
  std::vector<size_t> lengths;
  for (size_t L : requested_lengths) {
    if (L < min_len) {
      std::print(stderr, "worstcase: skipping length {} (shape '{}' needs >= "
                 "{})\n", L, needle_shape, min_len);
    } else if (L > haystack_size) {
      std::print(stderr, "worstcase: skipping length {} (> haystack {})\n", L,
                 haystack_size);
    } else {
      lengths.push_back(L);
    }
  }
  if (lengths.empty()) {
    std::cerr << "worstcase: no usable lengths\n";
    exit(1);
  }

  // Needle and haystack are both built per length (the 'block' haystack depends
  // on L; the rest ignore it).
  std::vector<std::string> needles, haystacks;
  needles.reserve(lengths.size());
  haystacks.reserve(lengths.size());
  for (size_t L : lengths) {
    needles.push_back(make_worstcase_needle(needle_shape, L));
    haystacks.push_back(make_worstcase_haystack(needle_shape, haystack_size, L));
  }

  // Show which three bytes StringZilla's anomaly selector anchors on for each
  // length: if any anchor is the odd byte, the filter rejects and SZ survives;
  // if all three are 'a', the filter passes everywhere and SZ is defeated.
  for (size_t li = 0; li < lengths.size(); ++li) {
    size_t a1, a2, a3;
    sz_locate_needle_anomalies(needles[li].data(), needles[li].size(), a1, a2,
                               a3);
    auto pb = [&](size_t off) {
      unsigned c = (unsigned char)needles[li][off];
      return std::format("{}:{:#x}", off, c);
    };
    std::print(stderr, "worstcase[{}] L={} SZ anchors = {} {} {}\n",
               needle_shape, lengths[li], pb(a1), pb(a2), pb(a3));
  }

  AmortState am;
  if (needs_amort(algos)) am.prepare(needles);

  // The needle must be absent: every algorithm must report not-found.
  for (size_t li = 0; li < lengths.size(); ++li) {
    const std::string &hay = haystacks[li];
    if (hay.find(needles[li]) != std::string::npos) {
      std::cerr << "worstcase: needle unexpectedly present\n";
      exit(1);
    }
    for (const Algo *a : algos) {
      auto [f, idx] = do_find(*a, am, li, hay.data(), hay.size(),
                              needles[li].data(), needles[li].size());
      if (f) {
        std::cerr << "Error: worstcase false match in " << a->name
                  << " at length " << lengths[li] << "\n";
        exit(1);
      }
    }
  }

  // ns[algo_index][length_index] — nanoseconds for one full-haystack search.
  std::vector<std::vector<double>> ns(
      algos.size(), std::vector<double>(lengths.size(), 0.0));
  for (size_t li = 0; li < lengths.size(); ++li) {
    const std::string &hay = haystacks[li];
    for (size_t ai = 0; ai < algos.size(); ++ai) {
      const Algo &a = *algos[ai];
      auto run = [&]() {
        auto [f, idx] = do_find(a, am, li, hay.data(), hay.size(),
                                needles[li].data(), needles[li].size());
        sink += f ? idx : needles[li].size();
      };
      ns[ai][li] = counters::bench(run).fastest_elapsed_ns();
    }
    std::print(stderr, "worstcase: length {} done\n", lengths[li]);
  }

  const char *hay_desc = needle_shape == "ab"      ? "'abab...'"
                         : needle_shape == "block" ? "'a'*(L-1)+'b' repeated"
                                                   : "all 'a'";
  std::print("# worstcase mode\n");
  std::print("haystack: {} bytes ({}); needle shape '{}' (never matches)\n",
             haystack_size, hay_desc, needle_shape);
  std::print("values are ns per full-haystack search (lower is better); "
             "GB/s = {} / ns\n", haystack_size);
  std::print("rows = algorithm, columns = needle length L\n\n");

  std::print("{:<48}", "algo");
  for (size_t L : lengths) std::print(" {:>12}", L);
  std::print("\n");
  for (size_t ai = 0; ai < algos.size(); ++ai) {
    std::print("{:<48}", algos[ai]->name);
    for (size_t li = 0; li < lengths.size(); ++li)
      std::print(" {:>12.1f}", ns[ai][li]);
    std::print("\n");
  }
}

// Random text over a small alphabet ('A'..'A'+alphabet-1), so short needles
// occur densely (the regime where enumerating matches per block can pay off).
static std::string generate_small_alphabet_string(size_t size, size_t alphabet) {
  static std::mt19937 gen(0x5A1FE5EDull);  // fixed seed: reproducible
  std::uniform_int_distribution<int> dist(0, (int)alphabet - 1);
  std::string r(size, 'A');
  for (auto &c : r) c = (char)('A' + dist(gen));
  return r;
}

// Enumerate every (overlapping) occurrence by calling the first-match kernel in
// a loop, advancing one byte past each match. This is the baseline: it pays the
// kernel's per-call setup once per match found. Uses whichever naive kernel the
// active backend provides (AVX-512 or NEON).
static size_t findall_loop_count(const char *t, size_t n, const char *p,
                                 size_t m) {
  size_t pos = 0, cnt = 0;
  while (pos + m <= n) {
    auto [f, idx] = SIMD_NAIVE_SEARCH(t + pos, n - pos, p, m);
    if (!f) break;
    ++cnt;
    pos += idx + 1;
  }
  return cnt;
}

// Enumerate every occurrence with the block-enumerating variant: one scan, all
// matches per block visited before moving on.
static size_t findall_block_count(const char *t, size_t n, const char *p,
                                  size_t m) {
  size_t cnt = 0;
  SIMD_NAIVE_SEARCH_ALL(t, n, p, m, [&](size_t) { ++cnt; });
  return cnt;
}

// findall mode: for each needle length, enumerate ALL (overlapping) occurrences
// of needles drawn from the text, two ways -- the first-match kernel in a loop
// vs. the block-enumerating avx512_naive_search_all -- and compare. Reports the
// average match count (density) and ns per full-text find-all for each, so the
// crossover with match density is visible.
void findall_benchmark(const std::string &text, const std::string &source_desc,
                       const std::vector<size_t> &requested_lengths) {
  static std::mt19937 gen(0xF1A11A11ull);  // fixed seed: reproducible needles
  const size_t needles_per_len = 100;
  volatile uint64_t sink = 0;

  std::vector<size_t> lengths;
  for (size_t L : requested_lengths) {
    if (L >= 1 && L <= text.size()) lengths.push_back(L);
    else std::print(stderr, "findall: skipping length {}\n", L);
  }
  if (lengths.empty()) { std::cerr << "findall: no usable lengths\n"; exit(1); }

  std::print("# findall mode (enumerate ALL occurrences, overlapping)\n");
  std::print("source: {}\n", source_desc);
  std::print("text size: {} bytes, {} needles per length\n", text.size(),
             needles_per_len);
  std::print("loop  = avx512_naive_search called in a loop (restart per match)\n");
  std::print("block = avx512_naive_search_all (enumerate matches per block)\n\n");
  std::print("{:>6} {:>13} {:>13} {:>13} {:>9}\n", "len", "avg_matches",
             "loop_ns", "block_ns", "speedup");

  for (size_t L : lengths) {
    std::vector<std::string> pats;
    pats.reserve(needles_per_len);
    std::uniform_int_distribution<size_t> start_dist(0, text.size() - L);
    for (size_t i = 0; i < needles_per_len; ++i)
      pats.push_back(text.substr(start_dist(gen), L));

    // Validate both strategies against a std::string::find reference.
    size_t total_matches = 0;
    for (const auto &p : pats) {
      size_t cl = findall_loop_count(text.data(), text.size(), p.data(), p.size());
      size_t cb = findall_block_count(text.data(), text.size(), p.data(), p.size());
      size_t cr = 0;
      for (size_t pos = text.find(p, 0); pos != std::string::npos;
           pos = text.find(p, pos + 1))
        ++cr;
      if (cl != cr || cb != cr) {
        std::cerr << "findall mismatch at len " << L << ": loop=" << cl
                  << " block=" << cb << " ref=" << cr << "\n";
        exit(1);
      }
      total_matches += cr;
    }

    auto run_loop = [&]() {
      size_t c = 0;
      for (const auto &p : pats)
        c += findall_loop_count(text.data(), text.size(), p.data(), p.size());
      sink += c;
    };
    auto run_block = [&]() {
      size_t c = 0;
      for (const auto &p : pats)
        c += findall_block_count(text.data(), text.size(), p.data(), p.size());
      sink += c;
    };
    double loop_ns =
        counters::bench(run_loop).fastest_elapsed_ns() / double(needles_per_len);
    double block_ns =
        counters::bench(run_block).fastest_elapsed_ns() / double(needles_per_len);
    std::print("{:>6} {:>13.1f} {:>13.1f} {:>13.1f} {:>8.2f}x\n", L,
               double(total_matches) / double(needles_per_len), loop_ns,
               block_ns, loop_ns / block_ns);
  }
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "";

  if (mode == "synthetic" || mode == "horspool" || mode == "ashvardanian" ||
      mode == "worstcase" || mode == "findall") {
    std::string path = "./data/43-0.txt";
    bool path_given = false;
    std::vector<size_t> lengths;       // horspool / worstcase only
    std::vector<const Algo *> algos;   // all modes; empty => all
    size_t worstcase_size = 1u << 16;  // worstcase only: haystack bytes
    std::string needle_shape = "tail"; // worstcase only: tail|aba|mid|high

    // Helper: value of an option given either as "--opt val" or "--opt=val".
    auto take_value = [&](const std::string &arg, int &k) -> std::string {
      auto eq = arg.find('=');
      if (eq != std::string::npos) return arg.substr(eq + 1);
      if (k + 1 < argc) return argv[++k];
      return "";
    };

    for (int k = 2; k < argc; ++k) {
      std::string arg = argv[k];
      if (arg == "--list") {
        list_algos();
        return 0;
      } else if (arg == "--algos" || arg.rfind("--algos=", 0) == 0) {
        for (const auto &nm : split_csv(take_value(arg, k))) {
          const Algo *a = find_algo_by_name(nm);
          if (!a) {
            std::cerr << "unknown algorithm: " << nm << " (use '" << mode
                      << " --list')\n";
            return 1;
          }
          algos.push_back(a);
        }
      } else if (arg == "--lengths" || arg.rfind("--lengths=", 0) == 0) {
        for (const auto &t : split_csv(take_value(arg, k)))
          lengths.push_back(std::stoul(t));
      } else if (arg == "--size" || arg.rfind("--size=", 0) == 0) {
        worstcase_size = std::stoul(take_value(arg, k));
      } else if (arg == "--needle" || arg.rfind("--needle=", 0) == 0) {
        needle_shape = take_value(arg, k);
      } else if (arg.rfind("--", 0) == 0) {
        std::cerr << "unknown option: " << arg << "\n";
        return 1;
      } else {
        path = arg;  // positional datafile (horspool & ashvardanian)
        path_given = true;
      }
    }

    if (algos.empty())
      for (const auto &a : kAlgos) algos.push_back(&a);
    if (!lengths.empty() && mode != "horspool" && mode != "worstcase" &&
        mode != "findall")
      std::print(stderr,
                 "ignoring --lengths (only applies to horspool/worstcase/"
                 "findall, not {})\n",
                 mode);

    if (mode == "synthetic") {
      collect_benchmark_results(1024, 100000, algos);
    } else if (mode == "horspool") {
      if (lengths.empty())
        for (size_t L = 2; L <= 20; ++L) lengths.push_back(L);
      if (path_given) {
        horspool_benchmark(load_file(path), path, lengths, algos);
      } else {
        // No source file given: generate a random text big enough for the
        // longest requested pattern (and a reasonable haystack to scan).
        size_t max_len = 0;
        for (size_t L : lengths) max_len = std::max(max_len, L);
        size_t gen_size = std::max<size_t>(1u << 17, max_len * 2 + 64);
        horspool_benchmark(generate_random_string(gen_size),
                           std::format("random ({} bytes)", gen_size), lengths,
                           algos);
      }
    } else if (mode == "ashvardanian") {
      ashvardanian_benchmark(load_file(path), algos);
    } else if (mode == "worstcase") {
      if (lengths.empty())
        for (size_t L : {4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u})
          lengths.push_back(L);
      worstcase_benchmark(worstcase_size, needle_shape, lengths, algos);
    } else {  // findall
      if (lengths.empty())
        for (size_t L : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u}) lengths.push_back(L);
      if (path_given)
        findall_benchmark(load_file(path), path, lengths);
      else
        findall_benchmark(
            generate_small_alphabet_string(worstcase_size, 4),
            std::format("random 4-symbol ({} bytes)", worstcase_size), lengths);
    }
    return 0;
  }

  std::print("usage: {} <mode> [datafile] [options]\n", argv[0]);
  std::print("  modes:\n");
  std::print("    synthetic     random 1KB text, 100k short needles (original)\n");
  std::print("    horspool      random substrings of a source text, "
             "first-occurrence ns matrix\n");
  std::print("    ashvardanian  StringWars-style forward find-all over the "
             "whole datafile, GB/s\n");
  std::print("    worstcase     haystack 'aaaa...', needle 'aa...ab' "
             "(never matches), ns-per-search matrix\n");
  std::print("    findall       enumerate ALL matches: first-match-in-a-loop "
             "vs block-enumerate\n");
  std::print("  datafile is optional for horspool (random text if omitted); "
             "ashvardanian defaults to ./data/43-0.txt\n");
  std::print("\n  options (all modes):\n");
  std::print("    --algos x,y       algorithms to test (default all), by name\n");
  std::print("    --list            list available algorithm names and exit\n");
  std::print("\n  horspool / worstcase:\n");
  std::print("    --lengths a,b,c   pattern lengths to test "
             "(horspool default 2..20, worstcase 2..512)\n");
  std::print("\n  worstcase only:\n");
  std::print("    --size N          haystack size in bytes (default 65536)\n");
  std::print("    --needle shape    tail|aba|mid|high|ab|block (default tail); "
             "'block' (all-'a' needle) defeats StringZilla too\n");
  return 1;
}
