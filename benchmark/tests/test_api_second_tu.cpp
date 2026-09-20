// A second translation unit including the public header: the program links
// only if every function the headers define is inline or static.
#if defined(SIMDSEARCH_SINGLE_HEADER)
#include "simdsearch.h"            // singleheader/, the amalgamated build
#else
#include <simdsearch/simdsearch.h>
#endif
#include <string_view>

size_t find_from_second_tu(std::string_view haystack, std::string_view needle) {
    return simdsearch::find(haystack, needle);
}
