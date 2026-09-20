// The public API as a user sees it: <simdsearch/simdsearch.h> included at
// C++17, exact-size heap buffers (so a read past either buffer is a real
// out-of-bounds read under a sanitizer, not a read into std::string's slack),
// every byte value including NUL, and a second translation unit that includes
// the same header, so that the headers are proven safe to include more than
// once per program.
#if defined(SIMDSEARCH_SINGLE_HEADER)
#include "simdsearch.h"            // singleheader/, the amalgamated build
#else
#include <simdsearch/simdsearch.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>

size_t find_from_second_tu(std::string_view haystack, std::string_view needle);

int main() {
    int fails = 0;
    auto expect = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
    };

    // The documented conventions.
    expect(simdsearch::find("hello", 5, "", 0) == std::pair<bool, size_t>{true, 0}, "empty needle matches at 0");
    expect(simdsearch::find("", 0, "", 0) == std::pair<bool, size_t>{true, 0}, "empty in empty");
    expect(simdsearch::find("", 0, "a", 1) == std::pair<bool, size_t>{false, 0}, "needle longer than haystack");
    expect(simdsearch::find(std::string_view("the quick brown fox"), "brown") == 10, "string_view overload");
    expect(simdsearch::find(std::string_view("the quick brown fox"), "green") == std::string_view::npos, "npos when absent");
    expect(find_from_second_tu("abcabcabd", "abd") == 6, "second translation unit");
    {
        // NUL bytes are ordinary bytes.
        const char t[] = {'a', 0, 'b', 0, 0, 'c'};
        const char p[] = {0, 0, 'c'};
        expect(simdsearch::find(t, sizeof t, p, sizeof p) == std::pair<bool, size_t>{true, 3}, "NUL bytes");
    }

    // Exact-size buffers, full byte range, misaligned bases, against
    // string_view::find.
    std::mt19937_64 g(2026);
    for (int r = 0; r < 60000; ++r) {
        const size_t n = g() % 2500, m = g() % 80;
        const size_t toff = g() % 64, poff = g() % 64;
        const bool binary = g() % 2;
        char* tb = (char*)std::malloc(n + toff + 1); char* text = tb + toff;
        char* pb = (char*)std::malloc(m + poff + 1); char* pat = pb + poff;
        for (size_t i = 0; i < n; ++i) text[i] = (char)(binary ? g() % 256 : 'a' + g() % 3);
        if (m <= n && g() % 2) {
            const size_t s = g() % (n - m + 1);
            std::memcpy(pat, text + s, m);
            if (m && g() % 3 == 0) pat[g() % m] ^= (char)(1 + g() % 255);
        } else {
            for (size_t i = 0; i < m; ++i) pat[i] = (char)(binary ? g() % 256 : 'a' + g() % 3);
        }
        const size_t ref = std::string_view(text, n).find(std::string_view(pat, m));
        const size_t got = simdsearch::find(std::string_view(text, n), std::string_view(pat, m));
        const auto got2 = simdsearch::find_unguarded(text, n, pat, m);
        if (got != ref || got2 != std::pair<bool, size_t>{ref != std::string_view::npos, ref == std::string_view::npos ? 0 : ref}) {
            if (fails < 10) std::fprintf(stderr, "FAIL: n=%zu m=%zu ref=%zu got=%zu\n", n, m, ref, got);
            ++fails;
        }
        std::free(tb); std::free(pb);
    }
    std::printf("api tests: %s (%d failures)\n", fails ? "FAILED" : "ok", fails);
    return fails ? 1 : 0;
}
