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
#include "neonsearch.h"

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
    {"find_neon", Kind::Stateless, neon_naive_search},
    {"find_neon64", Kind::Stateless, neon_naive_search64},
    {"find_neon_stringzilla", Kind::Stateless, neon_stringzilla_find},
    {"find_bmh", Kind::Stateless, bmh_search},
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
void collect_benchmark_results(size_t input_size, size_t number_strings) {
  std::string source = generate_random_string(input_size);
  auto strings = extract_random_substrings(source, number_strings);

  AmortState am;
  am.prepare(strings);

  // Validate every algorithm against std::string::find before timing.
  for (size_t id = 0; id < strings.size(); ++id) {
    const auto &str = strings[id];
    size_t p = source.find(str);
    if (p == std::string::npos) {
      std::cerr << "Error: reference substring not found\n";
      exit(1);
    }
    for (const auto &a : kAlgos) {
      auto [f, idx] = do_find(a, am, id, source.data(), source.size(),
                              str.data(), str.size());
      if (!f || idx != p) {
        std::cerr << "Error: " << a.name << " index mismatch (got " << idx
                  << ", expected " << p << ")\n";
        exit(1);
      }
    }
  }

  volatile uint64_t counter = 0;
  for (const auto &a : kAlgos) {
    auto run = [&]() {
      size_t c = 0;
      for (size_t id = 0; id < strings.size(); ++id) {
        const auto &str = strings[id];
        auto [f, idx] = do_find(a, am, id, source.data(), source.size(),
                                str.data(), str.size());
        if (f) c += idx;
      }
      counter += c;
    };
    pretty_print(a.name, number_strings, counters::bench(run));
  }
}

// Horspool mode: draw random substrings of the data file at each length from 2
// to 20 and time first-occurrence search for every algorithm. Patterns are cut
// from the text itself, so each is guaranteed to be found. The table is printed
// with one row per algorithm and one column per length, so reading across a row
// shows how that algorithm's cost evolves with pattern length.
void horspool_benchmark(const std::string &text) {
  static std::mt19937 gen(std::random_device{}());
  const size_t patterns_per_len = 2000;
  const size_t min_len = 2, max_len = 20;
  volatile uint64_t sink = 0;

  std::vector<size_t> lengths;
  for (size_t L = min_len; L <= max_len && text.size() >= L; ++L)
    lengths.push_back(L);

  // ns[algo_index][length_index]
  std::vector<std::vector<double>> ns(
      kAlgos.size(), std::vector<double>(lengths.size(), 0.0));

  for (size_t li = 0; li < lengths.size(); ++li) {
    size_t L = lengths[li];
    std::vector<std::string> pats;
    pats.reserve(patterns_per_len);
    std::uniform_int_distribution<size_t> start_dist(0, text.size() - L);
    for (size_t i = 0; i < patterns_per_len; ++i)
      pats.emplace_back(text.substr(start_dist(gen), L));

    AmortState am;
    am.prepare(pats);

    // Validate every algorithm against std::string::find before timing.
    for (size_t id = 0; id < pats.size(); ++id) {
      size_t ref = text.find(pats[id]);
      for (const auto &a : kAlgos) {
        auto [f, idx] = do_find(a, am, id, text.data(), text.size(),
                                pats[id].data(), pats[id].size());
        if (!f || idx != ref) {
          std::cerr << "Error: horspool mismatch in " << a.name << " at length "
                    << L << " (got " << idx << ", expected " << ref << ")\n";
          exit(1);
        }
      }
    }

    for (size_t ai = 0; ai < kAlgos.size(); ++ai) {
      const Algo &a = kAlgos[ai];
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
  std::print("text size: {} bytes, {} random patterns per length\n", text.size(),
             patterns_per_len);
  std::print("values are ns per first-occurrence search (lower is better)\n");
  std::print("rows = algorithm, columns = pattern length\n\n");

  std::print("{:<48}", "algo");
  for (size_t L : lengths) std::print(" {:>8}", L);
  std::print("\n");
  for (size_t ai = 0; ai < kAlgos.size(); ++ai) {
    std::print("{:<48}", kAlgos[ai].name);
    for (size_t li = 0; li < lengths.size(); ++li)
      std::print(" {:>8.1f}", ns[ai][li]);
    std::print("\n");
  }
}

// Ashvardanian mode: emulates StringWars' substring-search benchmark. The whole
// file is the haystack; its word tokens are the needles (every needle is thus
// present). Each timed pass scans the full haystack for every needle, locating
// all non-overlapping occurrences. Throughput is reported as GB/s of haystack
// scanned, crediting one full haystack length per needle (the StringWars
// convention), plus ns per needle.
void ashvardanian_benchmark(const std::string &hay) {
  const size_t max_needles = 1000;  // keep one pass quick; StringWars cycles all
  auto needles = tokenize_words(hay);
  if (needles.size() > max_needles) needles.resize(max_needles);
  if (needles.empty()) {
    std::cerr << "No word tokens found in input.\n";
    exit(1);
  }

  AmortState am;
  am.prepare(needles);

  std::print("# ashvardanian mode (StringWars-style forward find-all)\n");
  std::print("haystack: {} bytes, {} word needles (capped at {})\n", hay.size(),
             needles.size(), max_needles);

  size_t ref_total = 0;
  for (size_t id = 0; id < needles.size(); ++id)
    ref_total += count_all(kAlgos[0], am, id, hay, needles[id]);
  std::print("total non-overlapping matches across needles: {}\n\n", ref_total);

  std::print("{:<48} {:>10} {:>14} {:>12}\n", "algo", "GB/s", "ns/needle",
             "matches");
  volatile uint64_t sink = 0;
  for (const auto &a : kAlgos) {
    // Validate match count against the reference before timing.
    size_t total = 0;
    for (size_t id = 0; id < needles.size(); ++id)
      total += count_all(a, am, id, hay, needles[id]);
    if (total != ref_total) {
      std::cerr << "Error: ashvardanian match-count mismatch in " << a.name
                << " (" << total << " vs " << ref_total << ")\n";
      exit(1);
    }

    auto run = [&]() {
      size_t c = 0;
      for (size_t id = 0; id < needles.size(); ++id) {
        const auto &nd = needles[id];
        size_t pos = 0, m = nd.size();
        while (pos + m <= hay.size()) {
          auto [f, idx] = do_find(a, am, id, hay.data() + pos,
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
    std::print("{:<48} {:>10.3f} {:>14.1f} {:>12}\n", a.name, gbps,
               ns_per_needle, ref_total);
  }
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "";
  std::string path = (argc > 2) ? argv[2] : "./data/43-0.txt";

  if (mode == "synthetic") {
    collect_benchmark_results(1024, 100000);
    return 0;
  }
  if (mode == "horspool") {
    horspool_benchmark(load_file(path));
    return 0;
  }
  if (mode == "ashvardanian") {
    ashvardanian_benchmark(load_file(path));
    return 0;
  }

  std::print("usage: {} <mode> [datafile]\n", argv[0]);
  std::print("  modes:\n");
  std::print("    synthetic     random 1KB text, 100k short needles (original)\n");
  std::print("    horspool      random substrings of datafile, length 2..20, "
             "first-occurrence ns matrix\n");
  std::print("    ashvardanian  StringWars-style forward find-all over the "
             "whole datafile, GB/s\n");
  std::print("  datafile defaults to ./data/43-0.txt (horspool & ashvardanian)\n");
  return 1;
}
