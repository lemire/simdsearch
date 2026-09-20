// Correctness tests for needle_hammer against std::string::find.
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include "simdsearch/needle_hammer.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)
  #define NH_GUARDED avx512_needle_hammer
  #define NH_UNGUARDED avx512_needle_hammer_unguarded
#else
  #define NH_GUARDED neon_needle_hammer
  #define NH_UNGUARDED neon_needle_hammer_unguarded
#endif

static int fails = 0;
static void check(const std::string& t, const std::string& p, const char* what) {
    size_t ref = t.find(p);
    std::pair<bool, size_t> want{ref != std::string::npos, ref == std::string::npos ? 0 : ref};
    auto g = NH_GUARDED(t.data(), t.size(), p.data(), p.size());
    auto u = NH_UNGUARDED(t.data(), t.size(), p.data(), p.size());
    if (g != want || u != want) {
        if (fails < 20)
            std::fprintf(stderr, "FAIL %s: n=%zu m=%zu want (%d,%zu) guarded (%d,%zu) unguarded (%d,%zu)\n",
                         what, t.size(), p.size(), want.first, want.second, g.first, g.second, u.first, u.second);
        ++fails;
    }
}

int main() {
    std::mt19937 g(12345);
    // --- explicit cases ---
    { std::string t(5000, 'x'); check(t, "", "empty needle"); }
    { std::string t = "abcdefgh"; check(t, t, "n == m"); check(t, "abcdefghi", "m > n"); }
    for (size_t n : {1u, 2u, 3u, 4u, 63u, 64u, 65u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 4096u, 4097u}) {
        for (size_t m : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 33u, 36u, 37u, 64u, 65u, 128u, 129u, 200u}) {
            if (m > n) continue;
            std::string t(n, 'a'); for (auto& c : t) c = 'a' + g() % 3;
            // match at the last position
            std::string p = t.substr(n - m, m); check(t, p, "match at end");
            // match at the first position
            p = t.substr(0, m); check(t, p, "match at start");
            // no match: a byte outside the alphabet
            p = t.substr(n - m, m); p[m / 2] = 'z'; check(t, p, "absent (odd byte)");
            // tail / mid / quarter anomaly in an all-'a' haystack (small n path too)
            std::string h(n, 'a'); std::string q(m, 'a');
            if (m >= 2) { q[m - 1] = 'b'; check(h, q, "tail absent"); q[m - 1] = 'a'; }
            if (m >= 3) { q[m / 2] = 'b'; check(h, q, "mid absent"); q[m / 2] = 'a'; }
            if (m >= 8) { q[m / 8] = 'b'; check(h, q, "q1 absent"); q[m / 8] = 'a'; }
            check(h, q, "all-a present");
        }
    }
    // match straddling a 256-byte block boundary and a 64-byte line, at every alignment
    for (size_t off = 0; off < 64; ++off) {
        std::string t(8192, 'a'); for (auto& c : t) c = 'a' + g() % 20;
        for (size_t m : {5u, 9u, 40u, 70u, 300u}) {
            for (size_t pos : std::vector<size_t>{250, 255, 256, 260, 511, 4095, 4096, 8192 - m}) {
                std::string p = t.substr(pos, m);
                // make it the FIRST occurrence by using a rare byte
                p[m / 3] = 'Z'; std::string t2 = t; t2[pos + m / 3] = 'Z';
                std::string tt = std::string(off, 'q') + t2;
                check(tt, p, "straddle");
            }
        }
    }
    // periodic needles (two-way memory path) and block-shaped haystacks (guard fires)
    for (size_t L : {8u, 16u, 37u, 64u, 128u, 129u, 256u, 1024u}) {
        std::string hay(65536, 'a'); for (size_t i = 0; i < hay.size(); ++i) if (i % L == L - 1) hay[i] = 'b';
        std::string nd(L, 'a'); check(hay, nd, "block absent");
        // present near the end: a run of L 'a's
        std::string hay2 = hay; for (size_t i = 60000; i < 60000 + L; ++i) hay2[i] = 'a'; check(hay2, nd, "block present late");
        // periodic needle with period 3
        std::string per; while (per.size() < L) per += "abc"; per.resize(L);
        std::string h3; while (h3.size() < 65536) h3 += "abcabx"; check(h3, per, "periodic absent");
        std::string h4 = h3; for (size_t i = 0; i < L; ++i) h4[40000 + i] = per[i]; check(h4, per, "periodic present");
    }
    // escalation: log-like haystack where two anchors admit many survivors
    {
        std::string line = "2026-09-17 12:34:56 INFO  server-01 request id=0000000 path=/api/v1/items status=200\n";
        std::string hay; int k = 0;
        while (hay.size() < 1 << 20) { std::string l = line; for (int d = 0; d < 7; ++d) l[35 + d] = '0' + (k / (int)std::pow(10, 6 - d)) % 10; ++k; hay += l; }
        for (size_t L : {9u, 16u, 40u, 64u, 100u, 128u}) {
            for (int rep = 0; rep < 200; ++rep) {
                size_t pos = g() % (hay.size() - L); std::string p = hay.substr(pos, L); check(hay, p, "log present");
                p[L / 2] = '#'; check(hay, p, "log absent");
            }
        }
    }
    // random fuzz over small alphabets, small and medium haystacks
    for (int r = 0; r < 300000; ++r) {
        int sigma = 1 + g() % 4;
        size_t n = 1 + g() % 600; if (g() % 8 == 0) n = 1 + g() % 6000;
        size_t m = 1 + g() % (1 + g() % 90);
        std::string t(n, 'a'); for (auto& c : t) c = 'a' + g() % sigma;
        std::string p(m, 'a');
        if (g() % 3 == 0 && m <= n) { size_t s = g() % (n - m + 1); p = t.substr(s, m); if (g() % 2) p[g() % m] = 'a' + g() % sigma; }
        else for (auto& c : p) c = 'a' + g() % sigma;
        if (g() % 4 == 0) { size_t per = 1 + g() % 4; for (size_t i = per; i < m; ++i) p[i] = p[i - per]; }
        check(t, p, "fuzz");
    }
    // fuzz on wide alphabets with long needles (rare-byte anchor, escalation, guard)
    for (int r = 0; r < 20000; ++r) {
        int sigma = 2 + g() % 60;
        size_t n = 1024 + g() % 20000;
        size_t m = 5 + g() % 300;
        std::string t(n, 'a'); for (auto& c : t) c = 'a' + g() % sigma;
        std::string p; if (g() % 2) { size_t s = g() % (n - m + 1); p = t.substr(s, m); if (g() % 3 == 0) p[g() % m] = '!'; }
        else { p.assign(m, 'a'); for (auto& c : p) c = 'a' + g() % sigma; }
        check(t, p, "fuzz wide");
    }
    std::printf("needle_hammer tests: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}
