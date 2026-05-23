#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <print>
#include <random>
#include <string>
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

void collect_benchmark_results(size_t input_size, size_t number_strings) {
  std::string source = generate_random_string(input_size);
  auto strings = extract_random_substrings(source, number_strings);

  for (const auto &str : strings) {
    auto p = source.find(str);
    auto [found, idx] = neon_naive_search(source.data(), source.size(), str.data(), str.size());
    if (p == std::string::npos || !found || p != idx) {
      std::cerr << "Error: substring not found or index mismatch\n";
      std::cerr << "Source: " << source << "\n";
      std::cerr << "Substring: " << str << "\n";
      std::cerr << "std::string::find index: " << p << "\n";
      std::cerr << "neon_naive_search index: " << idx << "\n";
      exit(1);
    }
    auto [found_sz, idx_sz] = neon_stringzilla_find(source.data(), source.size(), str.data(), str.size());
    if (!found_sz || p != idx_sz) {
      std::cerr << "Error: neon_stringzilla_find index mismatch\n";
      std::cerr << "Source: " << source << "\n";
      std::cerr << "Substring: " << str << "\n";
      std::cerr << "std::string::find index: " << p << "\n";
      std::cerr << "neon_stringzilla_find index: " << idx_sz << "\n";
      exit(1);
    }
    auto [found_bmh, idx_bmh] = bmh_search(source.data(), source.size(), str.data(), str.size());
    if (!found_bmh || p != idx_bmh) {
      std::cerr << "Error: bmh_search index mismatch\n";
      std::cerr << "Source: " << source << "\n";
      std::cerr << "Substring: " << str << "\n";
      std::cerr << "std::string::find index: " << p << "\n";
      std::cerr << "bmh_search index: " << idx_bmh << "\n";
      exit(1);
    }
    auto [found64, idx64] = neon_naive_search64(source.data(), source.size(), str.data(), str.size());
    if (!found64 || p != idx64) {
      std::cerr << "Error: neon_naive_search64 index mismatch\n";
      std::cerr << "Source: " << source << "\n";
      std::cerr << "Substring: " << str << "\n";
      std::cerr << "std::string::find index: " << p << "\n";
      std::cerr << "neon_naive_search64 index: " << idx64 << "\n";
      exit(1);
    }

  }
  volatile uint64_t counter = 0;

  auto find_classic = [&strings, &counter, &source]() {
    size_t c = 0;
    for (const auto &str : strings) {
      auto p = source.find(str);
      if(p != std::string::npos) {
        c += p;
      }
    }
    counter += c;
  };
  pretty_print("find_classic", number_strings, counters::bench(find_classic));
  auto find_neon = [&strings, &counter, &source]() {
    size_t c = 0;
    for (const auto &str : strings) {
      auto [found, idx] = neon_naive_search(source.data(), source.size(), str.data(), str.size());
      if(found) {
        c += idx;
      }
    }
    counter += c;
  };
  pretty_print("find_neon", number_strings, counters::bench(find_neon));
  auto find_neon64 = [&strings, &counter, &source]() {
    size_t c = 0;
    for (const auto &str : strings) {
      auto [found, idx] = neon_naive_search64(source.data(), source.size(), str.data(), str.size());
      if(found) {
        c += idx;
      }
    }
    counter += c;
  };
  pretty_print("find_neon64", number_strings, counters::bench(find_neon64));
  auto find_neon_sz = [&strings, &counter, &source]() {
    size_t c = 0;
    for (const auto &str : strings) {
      auto [found, idx] = neon_stringzilla_find(source.data(), source.size(), str.data(), str.size());
      if(found) {
        c += idx;
      }
    }
    counter += c;
  };
  pretty_print("find_neon_stringzilla", number_strings, counters::bench(find_neon_sz));
  auto find_bmh = [&strings, &counter, &source]() {
    size_t c = 0;
    for (const auto &str : strings) {
      auto [found, idx] = bmh_search(source.data(), source.size(), str.data(), str.size());
      if(found) {
        c += idx;
      }
    }
    counter += c;
  };
  pretty_print("find_bmh", number_strings, counters::bench(find_bmh));
}

int main(int argc, char **argv) { collect_benchmark_results(1024, 100000); }
