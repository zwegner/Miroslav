#include <stdlib.h>
#include <sys/mman.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "miroslav.h"

// hacky macro for easier argument chomping
#define EAT_ARG() do { argc--; argv++; } while (0);

template <typename MatchVerifier>
void run(int argc, char **argv, NFAEdgeList &edges, MatchHandlerPrintLine &mh) {
    Miroslav<MatchHandlerPrintLine, MatchVerifier> m(edges, mh); 

    while (argc > 0) {
        File f(argv[0]);
        EAT_ARG();

        m.run(f);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [-c] pattern path...\n";
        std::cerr << "       " << argv[0] << " [-c] -f pattern-file path...\n";
        exit(1);
    }
    EAT_ARG();

    // Option parsing
    bool print_count = false;
    if (!strcmp(argv[0], "-c")) {
        print_count = true;
        EAT_ARG();
    }

    // Parse the input regex
    std::string pattern(argv[0]);
    char sep = '|';
    if (!strcmp(argv[0], "-f")) {
        EAT_ARG();
        std::ifstream f(argv[0]);
        EAT_ARG();
        std::stringstream buffer;
        buffer << f.rdbuf();
        pattern = buffer.str();
        sep = '\n';
    } else {
        pattern = std::string(argv[0]);
        EAT_ARG();
    }

    NFAEdgeList edges;

    typedef std::pair<uint8_t, uint32_t> edge_key;
    std::map<edge_key, uint32_t> edge_map{};

    uint32_t last_state = START_STATE;
    uint32_t state = 2;
    bool can_be_duplicate = true;
    for (uint32_t i = 0; i < pattern.length(); i++) {
        auto c = pattern[i];

        if (c == sep) {
            last_state = START_STATE;
            can_be_duplicate = true;
        } else {
            // Add a new character to the pattern. Before we do, see if we can
            // branch off a previous state that matches the substring
            // up until now.
            if (can_be_duplicate) {
                edge_key k(c, last_state);
                if (edge_map.count(k)) {
                    last_state = edge_map[k];
                    continue;
                } else
                    can_be_duplicate = false;
            }
            if (i < pattern.length() - 1 && pattern[i + 1] != sep) {
                edges.push_back(std::make_tuple(c, last_state, state));
                edge_map[edge_key(c, last_state)] = state;
                last_state = state;
                state += 1;
            } else
                edges.push_back(std::make_tuple(c, last_state, END_STATE));
        }
    }

    bool print_path = (argc > 1);
    MatchHandlerPrintLine mh(print_path, !print_count, print_count);

    if (state <= 32)
        run<BasicMatchVerifier32>(argc, argv, edges, mh);
    else if (state <= 64)
        run<BasicMatchVerifier64>(argc, argv, edges, mh);
    else if (state <= 128)
        run<BasicMatchVerifier128>(argc, argv, edges, mh);
    // SIMD verifier: only allow 2^n-1 states since we need an extra for a sentinel
    else if (state <= 255)
        run<SIMDMatchVerifier8>(argc, argv, edges, mh);
    else if (state <= 65535)
        run<SIMDMatchVerifier16>(argc, argv, edges, mh);
    else {
        std::cerr << "error: max states exceeded: " << state << "\n";
        exit(1);
    }
}
