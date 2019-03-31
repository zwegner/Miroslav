#include <stdlib.h>
#include <sys/mman.h>

#include <iostream>

#include "miroslav.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " pattern file\n";
        exit(1);
    }

    // Parse the input regex
    std::string pattern(argv[1]);
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

    MatchHandlerPrintLine mh(true, false, true);

    File f(argv[2]);

    Miroslav<MatchHandlerPrintLine> m(edges, mh); 
    m.run(f);
}
