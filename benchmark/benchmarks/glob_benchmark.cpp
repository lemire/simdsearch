// Glob ("stringmatchlen") pattern-matching benchmark.
//
// Companion to benchmark.cpp (substring search). Here the workload is whole-
// string glob matching: each pattern (e.g. "*needle*") is matched against the
// whole haystack. This is exactly the shape barch's asterisk fast path targets:
// a leading "*literal" is advanced with memchr / memmem instead of retrying the
// rest of the pattern one byte at a time, as the recursive Redis baseline does.
//
// All generated patterns are "star-only" (no '[...]' classes), so every matcher
// here is applicable and must agree with the Redis baseline.

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "counters/bench.h"
#include "globmatch.h"

using glob_fn = int (*)(const char *, size_t, const char *, size_t, int);

struct GlobAlgo {
  const char *name;
  glob_fn fn;
};

// The single source of truth for which matchers exist and how they are named.
static const std::vector<GlobAlgo> kAlgos = {
    {"glob_redis", globmatch::redis_stringmatchlen},
    {"glob_barch_general", globmatch::barch_general},
    {"glob_barch_asterisk", globmatch::barch_asterisk},
    {"glob_barch", globmatch::barch_stringmatchlen},
};

static const GlobAlgo *find_algo_by_name(const std::string &name) {
  for (const auto &a : kAlgos)
    if (name == a.name) return &a;
  return nullptr;
}

static std::vector<std::string> split_csv(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string cur;
  while (std::getline(ss, cur, ',')) {
    if (!cur.empty()) out.push_back(cur);
  }
  return out;
}

static void list_algos() {
  std::print("available matchers:\n");
  for (const auto &a : kAlgos) std::print("  {}\n", a.name);
}

static std::string load_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Cannot open file: " << path << "\n";
    exit(1);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string generate_random_string(size_t size) {
  static std::mt19937 gen(std::random_device{}());
  static const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ";
  std::uniform_int_distribution<size_t> dist(0, chars.size() - 1);
  std::string result;
  result.reserve(size);
  for (size_t i = 0; i < size; ++i) result += chars[dist(gen)];
  return result;
}

// True if a substring is a usable glob literal: no glob metacharacters (so the
// "*sub*" pattern stays star-only) and no NUL bytes.
static bool clean_literal(const std::string &s) {
  for (char c : s)
    if (c == '*' || c == '?' || c == '[' || c == ']' || c == '\\' || c == '\0')
      return false;
  return true;
}

// Build the pattern set. A fraction `absent_ratio` are random literals (almost
// never present in the text -> force a full scan with no match); the rest are
// "*<literal cut from the text>*" and so are present. Both shapes are wrapped in
// leading/trailing '*'.
static std::vector<std::string> build_patterns(const std::string &text,
                                               size_t count, size_t min_len,
                                               size_t max_len,
                                               double absent_ratio) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<size_t> len_dist(min_len, max_len);
  std::uniform_real_distribution<double> coin(0.0, 1.0);

  std::vector<std::string> pats;
  pats.reserve(count);
  while (pats.size() < count) {
    size_t len = len_dist(gen);
    std::string core;
    if (coin(gen) < absent_ratio || text.size() <= len) {
      core = generate_random_string(len);  // overwhelmingly absent
    } else {
      std::uniform_int_distribution<size_t> start_dist(0, text.size() - len);
      core = text.substr(start_dist(gen), len);  // present
      if (!clean_literal(core)) continue;        // keep the pattern star-only
    }
    pats.push_back("*" + core + "*");
  }
  return pats;
}

int main(int argc, char **argv) {
  std::string path = "./data/43-0.txt";
  size_t count = 5000;
  size_t min_len = 6, max_len = 20;
  double absent_ratio = 0.5;
  // Matching is case-sensitive: barch's memchr/memmem fast path compares raw
  // bytes, so it (and the dispatcher) is only correct for nocase == 0.
  const int nocase = 0;
  std::vector<const GlobAlgo *> algos;

  auto take_value = [&](const std::string &arg, int &k) -> std::string {
    auto eq = arg.find('=');
    if (eq != std::string::npos) return arg.substr(eq + 1);
    if (k + 1 < argc) return argv[++k];
    return "";
  };

  for (int k = 1; k < argc; ++k) {
    std::string arg = argv[k];
    if (arg == "--list") {
      list_algos();
      return 0;
    } else if (arg == "--algos" || arg.rfind("--algos=", 0) == 0) {
      for (const auto &nm : split_csv(take_value(arg, k))) {
        const GlobAlgo *a = find_algo_by_name(nm);
        if (!a) {
          std::cerr << "unknown matcher: " << nm << " (use --list)\n";
          return 1;
        }
        algos.push_back(a);
      }
    } else if (arg == "--count" || arg.rfind("--count=", 0) == 0) {
      count = std::stoul(take_value(arg, k));
    } else if (arg == "--minlen" || arg.rfind("--minlen=", 0) == 0) {
      min_len = std::stoul(take_value(arg, k));
    } else if (arg == "--maxlen" || arg.rfind("--maxlen=", 0) == 0) {
      max_len = std::stoul(take_value(arg, k));
    } else if (arg == "--absent-ratio" || arg.rfind("--absent-ratio=", 0) == 0) {
      absent_ratio = std::stod(take_value(arg, k));
    } else if (arg == "--help" || arg == "-h") {
      std::print("usage: {} [datafile] [options]\n", argv[0]);
      std::print("  options:\n");
      std::print("    --algos x,y       matchers to test (default all), by name\n");
      std::print("    --list            list available matcher names and exit\n");
      std::print("    --count N         number of patterns (default 5000)\n");
      std::print("    --minlen N        min literal length (default 6)\n");
      std::print("    --maxlen N        max literal length (default 20)\n");
      std::print("    --absent-ratio f  fraction of (absent) random patterns "
                 "(default 0.5)\n");
      return 0;
    } else if (arg.rfind("--", 0) == 0) {
      std::cerr << "unknown option: " << arg << "\n";
      return 1;
    } else {
      path = arg;  // positional datafile
    }
  }

  if (min_len < 1 || max_len < min_len) {
    std::cerr << "invalid lengths (need 1 <= minlen <= maxlen)\n";
    return 1;
  }
  if (algos.empty())
    for (const auto &a : kAlgos) algos.push_back(&a);

  std::string hay = load_file(path);
  auto pats = build_patterns(hay, count, min_len, max_len, absent_ratio);

  // Reference match count + per-matcher validation against the Redis baseline.
  std::vector<int> ref(pats.size());
  size_t ref_matches = 0;
  for (size_t i = 0; i < pats.size(); ++i) {
    ref[i] = globmatch::redis_stringmatchlen(pats[i].data(), pats[i].size(),
                                             hay.data(), hay.size(), nocase);
    ref_matches += (size_t)ref[i];
  }
  for (const GlobAlgo *a : algos) {
    for (size_t i = 0; i < pats.size(); ++i) {
      int r = a->fn(pats[i].data(), pats[i].size(), hay.data(), hay.size(),
                    nocase);
      if (r != ref[i]) {
        std::cerr << "Error: " << a->name << " disagrees with glob_redis on "
                  << "pattern \"" << pats[i] << "\" (got " << r << ", expected "
                  << ref[i] << ")\n";
        return 1;
      }
    }
  }

  std::print("# glob match mode\n");
  std::print("haystack: {} ({} bytes)\n", path, hay.size());
  std::print("{} patterns \"*literal*\", literal length {}..{}, nocase={}\n",
             pats.size(), min_len, max_len, nocase);
  std::print("matches: {} / {} ({:.1f}%)\n\n", ref_matches, pats.size(),
             100.0 * double(ref_matches) / double(pats.size()));

  std::print("{:<24} {:>12} {:>12} {:>12} {:>12}\n", "matcher", "ns/pattern",
             "Mpat/s", "GB/s", "matches");

  volatile uint64_t sink = 0;
  for (const GlobAlgo *a : algos) {
    auto run = [&]() {
      size_t c = 0;
      for (size_t i = 0; i < pats.size(); ++i)
        c += (size_t)a->fn(pats[i].data(), pats[i].size(), hay.data(),
                           hay.size(), nocase);
      sink += c;
    };
    auto agg = counters::bench(run);
    double ns_total = agg.fastest_elapsed_ns();
    double ns_per = ns_total / double(pats.size());
    double mpat = double(pats.size()) * 1000.0 / ns_total;
    double gbps = double(pats.size()) * double(hay.size()) / ns_total;
    std::print("{:<24} {:>12.1f} {:>12.3f} {:>12.3f} {:>12}\n", a->name, ns_per,
               mpat, gbps, ref_matches);
  }
  return 0;
}
