// Guardzilla against std::string_view::find: random text over small and full
// alphabets, needles drawn from the text (some with one byte changed) or at
// random, every haystack alignment, block boundaries, the adversarial families
// that defeat StringZilla's filter, and the guard's hand-over to two-way.
// Builds and passes for every GUARDZILLA_ANCHORS / _ALIGN / _BUDGET setting.
#include <simdsearch/simdsearch.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

static int fails = 0;
static size_t cases = 0;

static void check(const char* text, size_t n, const char* pat, size_t m, const char* what) {
    ++cases;
    const size_t ref = std::string_view(text, n).find(std::string_view(pat, m));
    const auto g = guardzilla::find(text, n, pat, m);
    const size_t got = g.first ? g.second : std::string_view::npos;
    if (got != ref) {
        if (fails < 20) std::fprintf(stderr, "FAIL %s: n=%zu m=%zu ref=%zu got=%zu\n", what, n, m, ref, got);
        ++fails;
    }
}

int main(int argc, char** argv) {
    // --quick: fewer and smaller cases, for running under an emulator (CI).
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    std::mt19937_64 g(20261013);
    // Random cases, exact-size buffers at random alignments.
    for (int r = 0; r < (quick ? 10000 : 200000); ++r) {
        const int alpha = (int)(g() % 5 == 0 ? 256 : 1 + g() % 6);
        const size_t n = g() % (r % 10 == 0 ? 20000 : 1200);
        const size_t m = 1 + g() % (r % 7 == 0 ? 3000 : 200);
        const size_t toff = g() % 64;
        std::vector<char> tb(n + toff + 1), pb(m + 1);
        char* text = tb.data() + toff;
        for (size_t i = 0; i < n; ++i) text[i] = (char)(alpha == 256 ? g() : 'a' + g() % alpha);
        if (m <= n && g() % 2) {
            std::memcpy(pb.data(), text + g() % (n - m + 1), m);
            if (g() % 3 == 0) pb[g() % m] ^= (char)(1 + g() % 255);
        } else {
            for (size_t i = 0; i < m; ++i) pb[i] = (char)(alpha == 256 ? g() : 'a' + g() % alpha);
        }
        check(text, n, pb.data(), m, "random");
    }
    // Every alignment of the haystack and every block offset of the match.
    {
        std::string base(4096 + 64, 'x');
        for (size_t i = 0; i < base.size(); ++i) base[i] = (char)('a' + (i * 7 + i / 13) % 23);
        for (size_t off = 0; off < 64; ++off)
            for (size_t m : {2, 5, 63, 64, 65, 200}) {
                const char* text = base.data() + off;
                const size_t n = 4096;
                for (size_t at = 0; at + m <= n; at += 61) {
                    const std::string pat(text + at, m);
                    check(text, n, pat.data(), m, "alignment");
                }
            }
    }
    // Adversarial families, where the guard hands over to two-way.
    for (size_t m : {16, 64, 65, 128, 1000, quick ? 2000 : 4096}) {
        const size_t n = quick ? 1 << 13 : 1 << 18;
        std::string h(n, 'a'), p(m, 'a');
        p[m - 1] = 'b';
        check(h.data(), n, p.data(), m, "tail");
        p = std::string(m, 'a'); p[m / 2] = 'b';
        check(h.data(), n, p.data(), m, "mid");
        for (size_t i = 0; i < n; ++i) h[i] = (i % m == m - 1) ? 'b' : 'a';
        p = std::string(m, 'a');
        check(h.data(), n, p.data(), m, "block");
        h[n - 3 * m] = 'a';   // a^{m} now occurs late: found after the hand-over
        check(h.data(), n, p.data(), m, "block-found");
    }
    std::printf("guardzilla tests (anchors %d, align %d, budget %d): %zu cases, %s\n", GUARDZILLA_ANCHORS,
                GUARDZILLA_ALIGN, GUARDZILLA_BUDGET, cases, fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
