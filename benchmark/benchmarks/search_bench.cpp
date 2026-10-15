// The benchmark: one searcher per executable. The searcher is chosen when the
// program is compiled (-DSEARCHER=<id>, see benchmark/CMakeLists.txt), never
// at run time: with several searchers in one program, the measured speed of
// one of them moved by up to 1.45x depending on what else was compiled in.
//
//   search_bench_<id> corpora FILE...    needles present and absent, per corpus
//   search_bench_<id> adversarial        the five adversarial families
//
// Output is CSV on stdout, one row per measurement:
//   mode,input,case,m,searcher,ns_per_search,bytes_per_search,
//   cycles_per_search,instructions_per_search,branches_per_search,
//   branch_misses_per_search
// ns_per_search is the best of `--rounds` timed passes over every needle,
// divided by the number of needles; bytes_per_search is the average number of
// haystack bytes a search had to consider (up to the end of the first match,
// or the whole haystack). The last four columns are the hardware counters of
// that fastest pass, per search (the counters library: Linux perf events or
// Apple's counters), or nan where the system does not give access to them.
// Every answer is checked against std::string_view::find before anything is
// timed.
//
// Options: --offset K   place the haystack K bytes past a 64-byte boundary
//                       (default 0)
//          --rounds R   timed passes (default 10)
//          --lengths L  comma-separated needle lengths (default 8,32,256,2048)
//          --header     print the CSV header
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <counters/event_counter.h>

using R = std::pair<bool, size_t>;
static constexpr R kNotApplicable{false, SIZE_MAX};

// ---------------------------------------------------------------------------
// The searchers. Each defines kName, and prepare/search/release: prepare runs
// once per needle outside the timed loop (only the prebuilt searchers use it).
// ---------------------------------------------------------------------------
#if !defined(SEARCHER)
#error "compile with -DSEARCHER=<id>; see benchmark/CMakeLists.txt"
#endif
#define SEARCHER_IS(x) (SEARCHER == SEARCHER_##x)
#define SEARCHER_guardzilla 1
#define SEARCHER_stringzilla 3
#define SEARCHER_memmem 4
#define SEARCHER_string_view 5
#define SEARCHER_boyer_moore_horspool 6
#define SEARCHER_rust_memmem 7
#define SEARCHER_rust_finder 8
#define SEARCHER_rust_str 9
#define SEARCHER_twoway 10
#define SEARCHER_needlehammer 11

#if SEARCHER_IS(guardzilla) || SEARCHER_IS(twoway)
#include <simdsearch/simdsearch.h>
#endif
#if SEARCHER_IS(needlehammer)
#include <simdsearch/needlehammer.h>
#endif
#if SEARCHER_IS(stringzilla)
#include <stringzilla/stringzilla.h>
#endif
#if SEARCHER_IS(boyer_moore_horspool)
#include <functional>
#endif
#if SEARCHER_IS(rust_memmem) || SEARCHER_IS(rust_finder) || SEARCHER_IS(rust_str)
extern "C" {
ptrdiff_t simdsearch_rust_memchr_find(const char*, size_t, const char*, size_t);
void* simdsearch_rust_finder_new(const char*, size_t);
ptrdiff_t simdsearch_rust_finder_find(const void*, const char*, size_t);
void simdsearch_rust_finder_free(void*);
ptrdiff_t simdsearch_rust_std_find(const char*, size_t, const char*, size_t);
}
static R from_rust(ptrdiff_t r) { return r == -2 ? kNotApplicable : r < 0 ? R{false, 0} : R{true, (size_t)r}; }
#endif

namespace searcher {
#if SEARCHER_IS(guardzilla)
#if GUARDZILLA_ANCHORS != 3 || GUARDZILLA_ALIGN != 1 || GUARDZILLA_BUDGET != 4
static const std::string kName = "Guardzilla (anchors " + std::to_string(GUARDZILLA_ANCHORS) + ", align " +
                                 std::to_string(GUARDZILLA_ALIGN) + ", budget " + std::to_string(GUARDZILLA_BUDGET) + ")";
#else
static const std::string kName = "Guardzilla";
#endif
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) { return guardzilla::find(t, n, p, m); }
static void release(void*) {}
#elif SEARCHER_IS(needlehammer)
#if NEEDLEHAMMER_WINDOW != 64 || NEEDLEHAMMER_BAND != 16
static const std::string kName = "NeedleHammer (window " + std::to_string(NEEDLEHAMMER_WINDOW) + ", band " +
                                 std::to_string(NEEDLEHAMMER_BAND) + ")";
#else
static const std::string kName = "NeedleHammer";
#endif
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) { return needlehammer::find(t, n, p, m); }
static void release(void*) {}
#elif SEARCHER_IS(twoway)
static const std::string kName = "two-way";
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) {
    if (m == 0) return {true, 0};
    if (n < m) return {false, 0};
    const size_t r = guardzilla::detail::twoway((const uint8_t*)t, n, (const uint8_t*)p, m, 0);
    return r < n ? R{true, r} : R{false, 0};
}
static void release(void*) {}
#elif SEARCHER_IS(stringzilla)
static const std::string kName = "StringZilla";
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) {
    if (m == 0) return {true, 0};
    const char* r = sz_find(t, n, p, m);
    return r ? R{true, (size_t)(r - t)} : R{false, 0};
}
static void release(void*) {}
#elif SEARCHER_IS(memmem)
static const std::string kName = "C library memmem";
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) {
    const void* r = memmem(t, n, p, m);
    return r ? R{true, (size_t)((const char*)r - t)} : R{false, 0};
}
static void release(void*) {}
#elif SEARCHER_IS(string_view)
static const std::string kName = "std::string_view::find";
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) {
    const size_t r = std::string_view(t, n).find(std::string_view(p, m));
    return r == std::string_view::npos ? R{false, 0} : R{true, r};
}
static void release(void*) {}
#elif SEARCHER_IS(boyer_moore_horspool)
static const std::string kName = "std::boyer_moore_horspool_searcher (prebuilt)";
using bmh = std::boyer_moore_horspool_searcher<const char*>;
static void* prepare(const char* p, size_t m) { return new bmh(p, p + m); }
static R search(void* st, const char* t, size_t n, const char*, size_t) {
    // not found: the empty range at the end (needles here are never empty)
    const auto r = (*static_cast<bmh*>(st))(t, t + n);
    return r.first == t + n ? R{false, 0} : R{true, (size_t)(r.first - t)};
}
static void release(void* st) { delete static_cast<bmh*>(st); }
#elif SEARCHER_IS(rust_memmem)
static const std::string kName = "Rust memchr::memmem::find";
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) { return from_rust(simdsearch_rust_memchr_find(t, n, p, m)); }
static void release(void*) {}
#elif SEARCHER_IS(rust_finder)
static const std::string kName = "Rust memchr::memmem::Finder (prebuilt)";
static void* prepare(const char* p, size_t m) { return simdsearch_rust_finder_new(p, m); }
static R search(void* st, const char* t, size_t n, const char*, size_t) { return from_rust(simdsearch_rust_finder_find(st, t, n)); }
static void release(void* st) { simdsearch_rust_finder_free(st); }
#elif SEARCHER_IS(rust_str)
static const std::string kName = "Rust str::find";
static void* prepare(const char*, size_t) { return nullptr; }
static R search(void*, const char* t, size_t n, const char* p, size_t m) { return from_rust(simdsearch_rust_std_find(t, n, p, m)); }
static void release(void*) {}
#else
#error "unknown SEARCHER"
#endif
}  // namespace searcher

// ---------------------------------------------------------------------------

static size_t g_offset = 0;
static int g_rounds = 10;
static std::vector<size_t> g_lengths = {8, 32, 256, 2048};

// The haystack, copied to g_offset bytes past a 64-byte boundary.
struct haystack {
    char* store = nullptr;
    const char* data = nullptr;
    size_t size = 0;
    explicit haystack(const std::string& h) : size(h.size()) {
        store = (char*)std::aligned_alloc(64, ((h.size() + g_offset) / 64 + 2) * 64);
        std::memcpy(store + g_offset, h.data(), h.size());
        data = store + g_offset;
    }
    ~haystack() { std::free(store); }
};

static bool valid_utf8(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        const size_t k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!k || i + k > s.size()) return false;
        for (size_t j = 1; j < k; ++j)
            if (((unsigned char)s[i + j] >> 6) != 2) return false;
        i += k;
    }
    return true;
}
static std::string reverse_code_points(const std::string& s) {
    std::string out;
    size_t i = s.size();
    while (i > 0) {
        size_t j = i - 1;
        while (j > 0 && ((unsigned char)s[j] >> 6) == 2) --j;
        out.append(s, j, i - j);
        i = j;
    }
    return out;
}

// Checks every needle, then times `rounds` passes over all of them. Returns
// false when the searcher does not take this input (Rust's str::find on bytes
// that are not UTF-8).
struct result {
    double ns, bytes, cycles, instructions, branches, branch_misses;  // per search
};

static bool measure(const haystack& H, const std::string& reference, const std::vector<std::string>& needles,
                    result* out, const char* what) {
    std::vector<void*> st;
    for (auto& nd : needles) st.push_back(searcher::prepare(nd.data(), nd.size()));
    double scanned = 0;
    bool ok = true;
    for (size_t k = 0; k < needles.size() && ok; ++k) {
        const size_t want = std::string_view(reference).find(needles[k]);
        scanned += want == std::string_view::npos ? (double)H.size : (double)(want + needles[k].size());
        const R r = searcher::search(st[k], H.data, H.size, needles[k].data(), needles[k].size());
        if (r == kNotApplicable) { ok = false; break; }
        const size_t got = r.first ? r.second : std::string_view::npos;
        if (got != want) {
            std::fprintf(stderr, "MISMATCH %s: %s m=%zu got %zu want %zu\n", searcher::kName.c_str(), what,
                         needles[k].size(), got, want);
            std::exit(2);
        }
    }
    if (ok) {
        // Called through a volatile pointer, and with the haystack pointer
        // hidden from the optimizer: an inlined searcher declared pure (memmem)
        // was otherwise hoisted out of the timed loop.
        R (*volatile fn)(void*, const char*, size_t, const char*, size_t) = searcher::search;
        static counters::event_collector collector;
        static const bool have_counters = collector.has_events();
        counters::event_count best;
        best.elapsed = std::chrono::duration<double>(1e300);
        size_t sink = 0;
        for (int r = 0; r < g_rounds; ++r) {
            collector.start();
            for (size_t k = 0; k < needles.size(); ++k) {
                const char* h = H.data;
                asm volatile("" : "+r"(h) : : "memory");
                const R x = fn(st[k], h, H.size, needles[k].data(), needles[k].size());
                sink += x.second + x.first;
            }
            const counters::event_count& c = collector.end();
            if (c.elapsed < best.elapsed) best = c;
        }
        if (sink == 42) std::fputs("", stderr);
        const double k = (double)needles.size(), nan = std::nan("");
        *out = {best.elapsed_ns() / k,
                scanned / k,
                have_counters ? best.cycles() / k : nan,
                have_counters ? best.instructions() / k : nan,
                have_counters ? best.branches() / k : nan,
                have_counters ? best.branch_misses() / k : nan};
    }
    for (void* s : st) searcher::release(s);
    return ok;
}

static void print_result(const result& x) {
    std::printf("%.1f,%.0f,%.1f,%.1f,%.1f,%.2f\n", x.ns, x.bytes, x.cycles, x.instructions, x.branches, x.branch_misses);
}

// Needles copied from random positions of the corpus (present: the search
// stops at the first occurrence) or with their code points reversed and kept
// only if absent (the search scans the whole haystack). When the corpus is
// valid UTF-8, so is every needle.
static void run_corpora(int argc, char** argv, int first) {
    for (int a = first; a < argc; ++a) {
        std::ifstream in(argv[a], std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[a]); std::exit(1); }
        std::string h((std::istreambuf_iterator<char>(in)), {});
        // A corpus cut to size in the middle of a UTF-8 sequence loses that
        // last partial character, so that it is valid UTF-8 (Rust's str::find
        // takes nothing else) for every searcher alike.
        for (size_t cut = 1; cut <= 3 && !valid_utf8(h) && h.size() > cut; ++cut)
            if (valid_utf8(std::string_view(h).substr(0, h.size() - cut))) h.resize(h.size() - cut);
        const bool utf8 = valid_utf8(h);
        const haystack H(h);
        std::string name = argv[a];
        name = name.substr(name.find_last_of('/') + 1);
        name = name.substr(0, name.find('.'));
        for (size_t m : g_lengths) {
            if (m > h.size()) continue;
            for (int present = 1; present >= 0; --present) {
                std::mt19937_64 rng(0xC0FFEE ^ m ^ (present ? 0 : 0xABCD));
                std::vector<std::string> needles;
                const int want = present ? 200 : 20;
                for (int tries = 0; (int)needles.size() < want && tries < 200000; ++tries) {
                    std::string nd = h.substr(std::uniform_int_distribution<size_t>(0, h.size() - m)(rng), m);
                    if (utf8 && !valid_utf8(nd)) continue;
                    if (!present) {
                        nd = reverse_code_points(nd);
                        if (std::string_view(h).find(nd) != std::string_view::npos) continue;
                    }
                    needles.push_back(std::move(nd));
                }
                if ((int)needles.size() < want) continue;   // e.g. every 8-letter DNA string occurs
                result x;
                if (measure(H, h, needles, &x, name.c_str())) {
                    std::printf("corpora,%s,%s,%zu,%s,", name.c_str(), present ? "present" : "absent", m,
                                searcher::kName.c_str());
                    print_result(x);
                }
                std::fflush(stdout);
            }
        }
    }
}

// The adversarial families, 1 MiB, needle absent:
//   tail   a^n against a^{m-1}b         mid   a^n against a^{m/2} b a^{m/2-1}
//   ab     (ab)^{n/2} against (ab)^{m/2} with the middle byte flipped
//   block  (a^{m-1}b)^{n/m} against a^m: every window matches m-1 bytes
//   near   copies of (a^255 b)^{m/256} with the second-to-last byte changed
static void run_adversarial() {
    const size_t n = 1 << 20;
    for (const char* fam : {"tail", "mid", "ab", "block", "near"}) {
        for (size_t m : {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
            std::string h(n, 'a'), p;
            const std::string f = fam;
            if (f == "tail") { p.assign(m - 1, 'a'); p += 'b'; }
            else if (f == "mid") { p.assign(m, 'a'); p[m / 2] = 'b'; }
            else if (f == "ab") {
                for (size_t i = 0; i < n; ++i) h[i] = (i & 1) ? 'b' : 'a';
                for (size_t i = 0; i < m; ++i) p += (i & 1) ? 'b' : 'a';
                p[m / 2] = p[m / 2] == 'a' ? 'b' : 'a';
            } else if (f == "block") {
                for (size_t i = 0; i < n; ++i) h[i] = (i % m == m - 1) ? 'b' : 'a';
                p.assign(m, 'a');
            } else {
                const size_t per = m < 256 ? (m < 64 ? m : 64) : 256;
                for (size_t i = 0; i < m; ++i) p += (i % per == per - 1) ? 'b' : 'a';
                std::string c = p;
                c[m - 2] = 'c';
                for (size_t i = 0; i < n; ++i) h[i] = c[i % m];
            }
            const haystack H(h);
            result x;
            if (measure(H, h, {p}, &x, fam)) {
                std::printf("adversarial,%s,absent,%zu,%s,", fam, m, searcher::kName.c_str());
                print_result(x);
            }
            std::fflush(stdout);
        }
    }
}

int main(int argc, char** argv) {
    int a = 1;
    std::string mode;
    for (; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--offset" && a + 1 < argc) g_offset = (size_t)std::atoi(argv[++a]) % 64;
        else if (s == "--rounds" && a + 1 < argc) g_rounds = std::atoi(argv[++a]);
        else if (s == "--lengths" && a + 1 < argc) {
            g_lengths.clear();
            std::string l = argv[++a];
            for (size_t pos = 0; pos <= l.size();) {
                const size_t c = l.find(',', pos);
                g_lengths.push_back((size_t)std::stoul(l.substr(pos, c - pos)));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        } else if (s == "--header") std::printf("mode,input,case,m,searcher,ns_per_search,bytes_per_search,cycles_per_search,"
                                           "instructions_per_search,branches_per_search,branch_misses_per_search\n");
        else { mode = s; ++a; break; }
    }
    if (mode == "corpora") run_corpora(argc, argv, a);
    else if (mode == "adversarial") run_adversarial();
    else {
        std::fprintf(stderr, "usage: %s [--offset K] [--rounds R] [--lengths L] [--header] corpora FILE... | adversarial\n", argv[0]);
        return 1;
    }
    return 0;
}
