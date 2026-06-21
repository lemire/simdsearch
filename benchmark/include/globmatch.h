#ifndef GLOBMATCH_H
#define GLOBMATCH_H

// Glob-style ("stringmatchlen") pattern matchers, for benchmarking the SIMD /
// memchr-accelerated variant against the plain recursive baseline.
//
// Three matchers are provided, all with the same signature and semantics
// (return 1 on match, 0 otherwise):
//
//   redis_stringmatchlen      - the canonical Redis util.c matcher. The
//                               recursive '*' handling tries the rest of the
//                               pattern at every text position, i.e. O(n*m) for
//                               a pattern such as "*needle*". This is the
//                               baseline.
//   barch_stringmatchlen_impl - barch's faithful copy of that same general
//                               matcher (handles '[...]' classes). It tracks
//                               the redis baseline; included so the speedup of
//                               the fast path below is measured against an
//                               equivalent implementation in the same codebase.
//   barch_asterisk_impl       - barch's optimized fast path for the very common
//                               "only '*', '?' and literals" pattern shape. A
//                               leading "*literal..." is advanced with memchr /
//                               memmem, which skip across the haystack instead
//                               of retrying the suffix one byte at a time. (libc
//                               memchr/memmem are themselves SIMD-accelerated on
//                               modern targets.) Correct only for case-sensitive
//                               matching (nocase == 0) and only for patterns
//                               with no '[...]' character classes; it also has
//                               an upstream bug for a '?' placed immediately
//                               after a '*' (the source keeps the "TODO ?BUG?"
//                               markers). The barch dispatcher and the
//                               glob_benchmark stay within this safe domain.
//   barch_stringmatchlen      - barch's public dispatcher: routes star-only
//                               patterns to the asterisk fast path and the rest
//                               to the general matcher.
//
// The asterisk fast path is ported faithfully from Chris Pretorius's barch
// project (https://github.com/tjizep/barch, src/glob.cpp), which in turn derives
// the general matcher from Redis. The original Redis license follows.
//
// All matchers require the pattern to be NUL-terminated (the asterisk fast path
// may read pattern[1] when patternLen == 1; a trailing '\0' makes that read safe
// and never matches '*'). std::string / string literals satisfy this.
//
/*
 * Copyright (c) 2009-2012, Redis Ltd.
 * Copyright (c) 2012, Twitter, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cctype>
#include <cstddef>
#include <cstring>

namespace globmatch {

// ---------------------------------------------------------------------------
// Upstream Redis baseline (redis/src/util.c stringmatchlen), reproduced as-is.
// Recursive on '*'; no haystack-skipping. This is the reference behaviour every
// other matcher here is validated against.
// ---------------------------------------------------------------------------
inline int redis_stringmatchlen_impl(const char *pattern, int patternLen,
                                     const char *string, int stringLen,
                                     int nocase, int *skipLongerMatches,
                                     int nesting) {
    /* Protection against abusive patterns. */
    if (nesting > 1000) return 0;

    while (patternLen && stringLen) {
        switch (pattern[0]) {
            case '*':
                while (patternLen && pattern[1] == '*') {
                    pattern++;
                    patternLen--;
                }
                if (patternLen == 1) return 1; /* match */
                while (stringLen) {
                    if (redis_stringmatchlen_impl(pattern + 1, patternLen - 1,
                                                  string, stringLen, nocase,
                                                  skipLongerMatches, nesting + 1))
                        return 1; /* match */
                    if (*skipLongerMatches) return 0; /* no match */
                    string++;
                    stringLen--;
                }
                /* There was no match for the rest of the pattern starting
                 * from anywhere in the rest of the string. If there were
                 * any '*' earlier in the pattern, we can terminate the
                 * search early without trying to match them to longer
                 * substrings, because a longer match for the earlier part
                 * of the pattern would require the rest of the pattern to
                 * match starting later in the string, which we have just
                 * shown impossible. */
                *skipLongerMatches = 1;
                return 0; /* no match */
                break;
            case '?':
                string++;
                stringLen--;
                break;
            case '[': {
                int not_op, match;

                pattern++;
                patternLen--;
                not_op = pattern[0] == '^';
                if (not_op) {
                    pattern++;
                    patternLen--;
                }
                match = 0;
                while (1) {
                    if (pattern[0] == '\\' && patternLen >= 2) {
                        pattern++;
                        patternLen--;
                        if (pattern[0] == string[0]) match = 1;
                    } else if (pattern[0] == ']') {
                        break;
                    } else if (patternLen == 0) {
                        pattern--;
                        patternLen++;
                        break;
                    } else if (patternLen >= 3 && pattern[1] == '-') {
                        int start = pattern[0];
                        int end = pattern[2];
                        int c = string[0];
                        if (start > end) {
                            int t = start;
                            start = end;
                            end = t;
                        }
                        if (nocase) {
                            start = tolower(start);
                            end = tolower(end);
                            c = tolower(c);
                        }
                        pattern += 2;
                        patternLen -= 2;
                        if (c >= start && c <= end) match = 1;
                    } else {
                        if (!nocase) {
                            if (pattern[0] == string[0]) match = 1;
                        } else {
                            if (tolower((int)pattern[0]) ==
                                tolower((int)string[0]))
                                match = 1;
                        }
                    }
                    pattern++;
                    patternLen--;
                }
                if (not_op) match = !match;
                if (!match) return 0; /* no match */
                string++;
                stringLen--;
                break;
            }
            case '\\':
                if (patternLen >= 2) {
                    pattern++;
                    patternLen--;
                }
                /* fall through */
            default:
                if (!nocase) {
                    if (pattern[0] != string[0]) return 0; /* no match */
                } else {
                    if (tolower((int)pattern[0]) != tolower((int)string[0]))
                        return 0; /* no match */
                }
                string++;
                stringLen--;
                break;
        }
        pattern++;
        patternLen--;
        if (stringLen == 0) {
            while (*pattern == '*') {
                pattern++;
                patternLen--;
            }
            break;
        }
    }
    if (patternLen == 0 && stringLen == 0) return 1;
    return 0;
}

inline int redis_stringmatchlen(const char *pattern, size_t patternLen,
                                const char *string, size_t stringLen,
                                int nocase) {
    int skipLongerMatches = 0;
    return redis_stringmatchlen_impl(pattern, (int)patternLen, string,
                                     (int)stringLen, nocase,
                                     &skipLongerMatches, 0);
}

// ---------------------------------------------------------------------------
// barch (https://github.com/tjizep/barch, src/glob.cpp).
// ---------------------------------------------------------------------------

// if there's only literals in the next N pattern bytes (a very common case)
template <int N>
inline bool no_token(const char *pattern) {
    for (int i = 0; i < N; i++) {
        if (pattern[i] == '\\' || pattern[i] == '?' || pattern[i] == '*') {
            return false;
        }
    }
    return true;
}

// barch's optimized matcher for the "only '*', '?' and literals" pattern shape.
// The win is in the '*' case: a leading "*literal..." is located with memmem /
// memchr, which skip across the haystack, rather than retrying the rest of the
// pattern one byte at a time as the recursive baseline does. Not valid for
// patterns containing '[...]' character classes.
inline int barch_asterisk_impl(const char *pattern, int patternLen,
                               const char *string, int stringLen, int nocase,
                               int *skipLongerMatches, int nesting) {
    /* Protection against abusive patterns. */
    if (nesting > 1000) return 0;

    while (patternLen && stringLen) {
        switch (pattern[0]) {
            case '*':
                while (patternLen && pattern[1] == '*') {
                    pattern++;
                    patternLen--;
                }
                if (patternLen == 1) return 1; /* match */
                if (nesting == 0 && patternLen > 4) {
                    auto asterisk =
                        (const char *)memchr(pattern + 1, '*', patternLen - 1);
                    while (pattern[1] == '?' && patternLen > 4) {
                        ++pattern;
                        --patternLen;
                    }
                    auto slash = memchr(pattern + 1, '\\', patternLen - 1);
                    if (!slash) {
                        if (patternLen > 4 && no_token<4>(pattern + 1)) {
                            // hope the 4 chars are enough to find a unique
                            // sequence far away; we would prefer to choose the
                            // least frequent chars in the pattern
                            auto str = (const char *)memmem(string, stringLen,
                                                            pattern + 1, 4);
                            if (!str) {
                                return 0;
                            }
                            stringLen -= (str - string);
                            string = str;
                        } else if (pattern[1] != '?' &&
                                   (!asterisk          // no further asterisks
                                    || (asterisk - pattern) > 3)) {  // or far away
                        _memchr_section:
                            // weakness: pattern[1] may be a very frequent char
                            auto str = (const char *)memchr(string, pattern[1],
                                                            stringLen);
                            if (!str) {
                                return 0;
                            }
                            stringLen -= (str - string);
                            string = str;
                            if (stringLen > 3 && pattern[2] != '?' &&
                                pattern[2] != string[1]) {
                                // we can try again
                                string++;
                                stringLen--;
                                goto _memchr_section;  // adds a few percent
                            }
                        }
                    }
                }
                while (stringLen) {
                    if (barch_asterisk_impl(pattern + 1, patternLen - 1, string,
                                            stringLen, nocase, skipLongerMatches,
                                            nesting + 1))
                        return 1; /* match */
                    if (*skipLongerMatches) return 0; /* no match */
                    string++;
                    stringLen--;
                }
                *skipLongerMatches = 1;
                return 0; /* no match */
                break;

            case '?':
                string++;
                stringLen--;
                break;
            case '\\':
                if (patternLen >= 2) {
                    pattern++;
                    patternLen--;
                }
                /* fall through */
            default:
                if (!nocase) {
                    if (pattern[0] != string[0]) return 0; /* no match */
                } else {
                    if (tolower((int)pattern[0]) != tolower((int)string[0]))
                        return 0; /* no match */
                }
                string++;
                stringLen--;
                break;
        }
        pattern++;
        patternLen--;
        if (stringLen == 0) {
            while (*pattern == '*') {
                pattern++;
                patternLen--;
            }
            break;
        }
    }
    if (patternLen == 0 && stringLen == 0) return 1;
    return 0;
}

// barch's faithful copy of the general (Redis-style) matcher. Equivalent to the
// redis baseline above; kept separate so the fast path is compared against an
// equivalent implementation inside the same project.
inline int barch_stringmatchlen_impl(const char *pattern, int patternLen,
                                     const char *string, int stringLen,
                                     int nocase, int *skipLongerMatches,
                                     int nesting) {
    return redis_stringmatchlen_impl(pattern, patternLen, string, stringLen,
                                     nocase, skipLongerMatches, nesting);
}

// true if the pattern has no '[...]' character class (so the asterisk fast path
// is applicable).
inline bool star_only(const char *pattern, size_t patternLen) {
    for (size_t i = 0; i < patternLen; i++) {
        switch (pattern[i]) {
            case '*':
            case '?':
            case '\\':
                continue;
            case '[':
                return false;
            default:
                continue;
        }
    }
    return true;
}

// Direct entry point for the asterisk fast path (star-only patterns only).
inline int barch_asterisk(const char *pattern, size_t patternLen,
                          const char *string, size_t stringLen, int nocase) {
    int skipLongerMatches = 0;
    return barch_asterisk_impl(pattern, (int)patternLen, string, (int)stringLen,
                               nocase, &skipLongerMatches, 0);
}

// Direct entry point for barch's general matcher.
inline int barch_general(const char *pattern, size_t patternLen,
                         const char *string, size_t stringLen, int nocase) {
    int skipLongerMatches = 0;
    return barch_stringmatchlen_impl(pattern, (int)patternLen, string,
                                     (int)stringLen, nocase,
                                     &skipLongerMatches, 0);
}

// barch's public dispatcher: star-only patterns take the fast path.
inline int barch_stringmatchlen(const char *pattern, size_t patternLen,
                                const char *string, size_t stringLen,
                                int nocase) {
    int skipLongerMatches = 0;
    if (star_only(pattern, patternLen))
        return barch_asterisk_impl(pattern, (int)patternLen, string,
                                   (int)stringLen, nocase, &skipLongerMatches,
                                   0);
    return barch_stringmatchlen_impl(pattern, (int)patternLen, string,
                                     (int)stringLen, nocase, &skipLongerMatches,
                                     0);
}

}  // namespace globmatch

#endif  // GLOBMATCH_H
