// The cost of a mispredicted branch, and of NeedleHammer's second scan, on
// this processor: the constants of NeedleHammer's switch threshold.
//
// 1. The same loop over 1 MiB of random bits and over constant bits; the extra
//    cycles divided by the extra mispredictions.
// 2. Guardzilla and NeedleHammer's second scan alone over 1 MiB of English
//    text with a missing needle (no candidates): the extra cycles per 64-byte
//    block of the second scan.
//
//   branch_penalty CORPUS     prints CSV: what,value
#include <simdsearch/needlehammer.h>

#include <counters/event_counter.h>

#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

__attribute__((noinline)) static unsigned long branchy(const unsigned char* b, size_t n) {
    unsigned long s = 0;
    for (size_t i = 0; i < n; ++i) {
        if (b[i]) {
            s += i * 3;
            asm volatile("" : "+r"(s));
        } else {
            s ^= i;
            asm volatile("" : "+r"(s));
        }
    }
    return s;
}

template <class F>
static counters::event_count best_of(F f, int reps) {
    counters::event_collector c;
    counters::event_count best;
    best.elapsed = std::chrono::duration<double>(1e300);
    for (int r = 0; r < reps; ++r) {
        c.start();
        f();
        const counters::event_count& x = c.end();
        if (x.elapsed < best.elapsed) best = x;
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2 || !counters::has_performance_counters()) {
        std::fprintf(stderr, "usage: %s CORPUS (needs access to performance counters)\n", argv[0]);
        return 1;
    }
    const size_t n = 1 << 20;
    std::vector<unsigned char> rnd(n), one(n, 1);
    std::mt19937 g(1);
    for (auto& x : rnd) x = g() & 1;
    unsigned long sink = 0;
    const auto c1 = best_of([&] { sink += branchy(one.data(), n); }, 50);
    const auto c2 = best_of([&] { sink += branchy(rnd.data(), n); }, 50);
    std::printf("what,value\n");
    std::printf("misprediction_cycles,%.1f\n", (c2.cycles() - c1.cycles()) / (c2.branch_misses() - c1.branch_misses()));

    std::ifstream in(argv[1], std::ios::binary);
    std::string h((std::istreambuf_iterator<char>(in)), {});
    char* buf = (char*)std::aligned_alloc(64, (h.size() / 64 + 2) * 64);
    std::memcpy(buf, h.data(), h.size());
    const std::string needle = "zq#!@ no such needle in the text";
    const auto* t = reinterpret_cast<const uint8_t*>(buf);
    const auto* p = reinterpret_cast<const uint8_t*>(needle.data());
    const size_t m = needle.size(), last = h.size() - m, blocks = (last - 63) / 64 + 1;
    size_t o[3];
    guardzilla::detail::sz_anchors(p, m, o);
    const auto gz = best_of([&] { sink += guardzilla::find(buf, h.size(), needle.data(), m).second; }, 50);
    const auto second = best_of([&] {
        size_t work = 0;
        for (size_t i = 0; i + 63 <= last; i += 64 * NEEDLEHAMMER_WINDOW) {
            const size_t end = last - 63 - i >= 64 * NEEDLEHAMMER_WINDOW ? i + 64 * NEEDLEHAMMER_WINDOW : last - 62;
            sink += needlehammer::detail::compress_chunk(t, h.size(), p, m, o, i, end, work);
        }
    }, 50);
    std::printf("first_scan_cycles_per_block,%.2f\n", gz.cycles() / blocks);
    std::printf("second_scan_cycles_per_block,%.2f\n", second.cycles() / blocks);
    std::printf("second_scan_extra_cycles_per_block,%.2f\n", (second.cycles() - gz.cycles()) / blocks);
    std::printf("threshold,%.3f\n", (second.cycles() - gz.cycles()) / blocks /
                                        ((c2.cycles() - c1.cycles()) / (c2.branch_misses() - c1.branch_misses())));
    std::free(buf);
    return sink == 42;
}
