#include <stdlib.h>
#include <sys/mman.h>

#include <iostream>

#include "miroslav.h"

// hacky macro for easier argument chomping
#define EAT_ARG() do { argc--; argv++; } while (0);

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [-c] pattern file...\n";
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
    EAT_ARG();
    std::vector<std::tuple<uint8_t, uint32_t, uint32_t>> edges;
    uint32_t last_state = START_STATE;
    uint32_t state = 2;
    for (uint32_t i = 0; i < pattern.length(); i++) {
        auto c = pattern[i];

        if (c == '|')
            last_state = START_STATE;
        else if (i < pattern.length() - 1 && pattern[i + 1] != '|') {
            edges.push_back(std::make_tuple(c, last_state, state));
            last_state = state;
            state += 1;
        } else
            edges.push_back(std::make_tuple(c, last_state, END_STATE));
    }

    if (state > MAX_STATES) {
        std::cerr << "error: max states exceeded: " << state << "\n";
        exit(1);
    }

    bool print_path = (argc > 1);

    MatchHandlerPrintLine mh(print_path, !print_count, print_count);

    while (argc > 0) {
        File f(argv[0]);
        EAT_ARG();

        Miroslav<MatchHandlerPrintLine> m(edges, mh); 
        m.run(f);
    }
}
