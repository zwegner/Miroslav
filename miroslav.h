#pragma once

#include <stdint.h>

#include <iostream>
#include <tuple>
#include <vector>

#include <immintrin.h>

#define UNUSED __attribute__((unused))

static inline uint32_t bsf64(uint64_t x) {
	return __builtin_ctzll(x);
}

#define START_STATE (0)
#define END_STATE (1)

// For the backwards verification run. smask should have MAX_STATES bits.
// The backwards state table will be (max/8) * max * 256 bytes.
// So 8 states is 2K, 16 => 8K, 32 => 32K, 64 => 128K, 128 => 512K

#define MAX_STATES (32)
typedef uint32_t smask;

////////////////////////////////////////////////////////////////////////////////
// Basic mmap wrapper for file handling
////////////////////////////////////////////////////////////////////////////////
struct File {
    const char *path;
    const uint8_t *data;
    size_t size;

    File() : path(NULL), data(NULL), size(0) { }

    File(const char *path) : path(path) {
        FILE *f = fopen(path, "r");

        fseek(f, 0, SEEK_END);
        this->size = ftell(f);

        this->data = (uint8_t *)mmap(NULL, this->size, PROT_READ, MAP_FILE|MAP_PRIVATE, fileno(f), 0);
        if (this->data == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }

        fclose(f);
    }

    ~File() {
        if (munmap((void *)this->data, this->size) == -1) {
            perror("munmap");
            exit(1);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// Vector definition classes
// These contain the basic properties of the underlying SIMD implementation,
// as well as functions for any architecture-dependent operations.
////////////////////////////////////////////////////////////////////////////////

// AVX2: we have 32-byte vectors, but can only shuffle within 16-byte halves of
// these vectors, which requires special handling
struct VecInfoAVX2 {
    static const uint32_t VL = 32;
    typedef __m256i V;
    typedef uint32_t vmask;
    typedef uint64_t double_vmask;

    // Lookup mask: since avx2 shuffles only work in two 16-byte chunks, we have
    // to use just four bits of each character as a table index, instead of the
    // five that we'd prefer
    static const uint32_t LMASK = 16 - 1;

    static inline V broadcast(uint8_t value) {
        return _mm256_set1_epi8(value);
    }

    static inline V permute(V &table, V &index) {
        return _mm256_shuffle_epi8(table, index);
    }

    static inline V vec_and(V &a, V &b) {
        return _mm256_and_si256(a, b);
    }

    static inline V vec_lanes_shl_1(V &top, V &bottom) {
        // Jumble the masks around, including the last state mask, so we can
        // compare consecutive bytes. The permute takes [last_H, last_L] and
        // [to_H, to_L] and gives us [to_L, last_H]. This is like the "to"
        // vector shifted back in time 16 bytes, so we can use it with valign
        // back 15 bytes, giving us a net shift of just 1 byte.
        V shl_16 = _mm256_permute2f128_si256(top, bottom, 0x03);
        return _mm256_alignr_epi8(top, shl_16, 15);
    }

    static inline vmask test_high_bit(V &a) {
        return _mm256_movemask_epi8(a);
    }

    static inline vmask test_low_bit(V &a) {
        // Movemask tests the high bit, so the input has to be shifted up
        return _mm256_movemask_epi8(_mm256_slli_epi32(a, 7));
    }

    static inline vmask test_nz(V &a) {
        static V vzero = broadcast(0);
        return ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, vzero));
    }

    static void prepare_state_table(uint8_t state_bytes[VL]) {
        // HACK because AVX2 sucks and can only do 16-byte shuffles
        for (uint32_t i = 0; i < 16; i++)
            state_bytes[i + 16] = state_bytes[i];
    }
};

////////////////////////////////////////////////////////////////////////////////
// Match verifier
// Since the vectorized matcher can give false positives, we have to run through
// each potential match backwards from the end character.
////////////////////////////////////////////////////////////////////////////////

template<bool linewise_matches>
class BasicMatchVerifier {
    // This is a table of [state][input_character] -> prev_states, where
    smask back_edges[MAX_STATES][256];

public:
    BasicMatchVerifier(std::vector<std::tuple<uint8_t, uint32_t, uint32_t>> &edges) {
        for (uint32_t i = 0; i < MAX_STATES; i++)
            for (uint32_t j = 0; j < 256; j++)
                back_edges[i][j] = 0;

        // Initialize from/to state masks tables
        for (auto edge : edges) {
            uint8_t c;
            uint32_t from, to;
            std::tie(c, from, to) = edge;

            // Fill in backwards edges in the NFA graph
            back_edges[to][c] |= 1 << from;
        }
    }

    const uint8_t *verify(const uint8_t *data, const uint8_t *end) {
        smask states = 1 << END_STATE;

        while (states && end >= data) {
            uint8_t c = *end;

            if (linewise_matches && c == '\n')
                break;

            // Iterate through all current states
            smask next_states = 0;
            while (states) {
                uint32_t s = bsf64(states);
                next_states |= back_edges[s][c];
                states &= states - 1;
            }

            if (next_states & (1 << START_STATE))
                return end;

            states = next_states;
            end--;
        }
        return NULL;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Match handlers
// These classes take verified matches and perform an action (count it, print
// it out, etc.)
////////////////////////////////////////////////////////////////////////////////

// Count all occurrences of a match, ignoring newlines
class MatchHandlerBasicCounter {
    uint32_t _match_count;

public:
    typedef uint32_t return_type;

    MatchHandlerBasicCounter() { }

    void start() {
        _match_count = 0;
    }

    void handle_match(UNUSED File &f, UNUSED const uint8_t *start, UNUSED const uint8_t *end) {
        _match_count++;
    }

    return_type finish(UNUSED File &f) {
        return _match_count;
    }
};

// Basic grep-style handling: print out the full line, ignore other matches
// on the same line. We support counting matching lines and printing the path.
class MatchHandlerPrintLine {
    uint32_t _match_count;
    const uint8_t *_last_match_line;
    bool _print_path;
    bool _print_matches;
    bool _print_count;

public:
    typedef void return_type;

    MatchHandlerPrintLine(bool print_path, bool print_matches, bool print_count) {
        _print_path = print_path;
        _print_matches = print_matches;
        _print_count = print_count;
    }

    void start() {
        _match_count = 0;
        _last_match_line = NULL;
    }

    void handle_match(File &f, const uint8_t *start, const uint8_t *end) {
        // We only care about one match per line for printing/counting
        // purposes. If this match is on the same line as the last, just skip it.
        if (start < _last_match_line)
            return;

        const uint8_t *last_nl = start;
        while (last_nl >= f.data && *last_nl != '\n')
            last_nl--;

        const uint8_t *next_nl = end;
        while (next_nl < f.data + f.size && *next_nl != '\n')
            next_nl++;

        if (_print_matches) {
            if (_print_path)
                std::cout << f.path << ":";
            std::string m(last_nl + 1, next_nl);
            std::cout << m << "\n";
        }

        _match_count++;
        _last_match_line = next_nl;
    }

    return_type finish(File &f) {
        if (_print_count) {
            if (_print_path)
                std::cout << f.path << ":";
            std::cout << _match_count << "\n";
        }
    }
};

////////////////////////////////////////////////////////////////////////////////
// Miroslav: the core SIMD string matching algorithm
////////////////////////////////////////////////////////////////////////////////
template<typename VI, typename MatchVerifier, typename MatchHandler>
class _Miroslav {
    // Hacky substitute for "using"
    typedef typename VI::V V;
    typedef typename VI::vmask vmask;
    typedef typename VI::double_vmask double_vmask;
    static const uint32_t VL = VI::VL;
    static const uint32_t LMASK = VI::LMASK;

    static inline uint8_t state_mask(uint32_t state) {
        if (state == START_STATE)
            return 1 << 0;
        if (state == END_STATE)
            return 1 << 7;
        return 1 << (state % 6 + 1);
    }

    static inline uint8_t char_mask(uint8_t c) {
        return c & LMASK;
    }

    MatchVerifier match_verifier;
    MatchHandler &match_handler;

    V from_states;
    V to_states;
    V v_char_mask;

    // For branchless testing of whether a pattern has a 1-character match.
    // Cuts down on false matches when the start and end characters of a
    // pattern hash to the same character class, which would cause every
    // occurrence of a character in that class to generate a match
    vmask has_1_char_match;

public:
    _Miroslav(std::vector<std::tuple<uint8_t, uint32_t, uint32_t>> &edges,
            MatchHandler &handler) : match_verifier(edges), match_handler(handler) {
        uint8_t from_state_bytes[VL] = {0};
        uint8_t to_state_bytes[VL] = {0};

        has_1_char_match = 0;

        // Initialize from/to state masks tables
        for (auto edge : edges) {
            uint8_t c;
            uint32_t from, to;
            std::tie(c, from, to) = edge;

            from_state_bytes[char_mask(c)] |= state_mask(from);
            to_state_bytes[char_mask(c)] |= state_mask(to);

            // Check for 1-character matches and update the mask
            if (from == START_STATE && to == END_STATE)
                has_1_char_match = (vmask)-1;
        }

        VI::prepare_state_table(from_state_bytes);
        VI::prepare_state_table(to_state_bytes);

        from_states = *(V*)from_state_bytes;
        to_states = *(V*)to_state_bytes;

        v_char_mask = VI::broadcast(LMASK);
    }

    typename MatchHandler::return_type run(File &f) {
        match_handler.start();

        double_vmask carry, last_carry = 0;
        V last_to = VI::broadcast(state_mask(START_STATE));

        const uint8_t *chunk;
        for (chunk = f.data; chunk + VL <= f.data + f.size; chunk += VL) {
            V masked_input = VI::vec_and(*(V*)chunk, v_char_mask);

            V from = VI::permute(from_states, masked_input);
            V to = VI::permute(to_states, masked_input);

            // Get a vector of the to states, but shifted back in the data
            // stream by 1 byte. We fill the empty space in the first lane with
            // the last lane from last_to (which is initialized to the starting
            // state).
            V shl_to_1 = VI::vec_lanes_shl_1(to, last_to);

            V trans = VI::vec_and(shl_to_1, from);

            // Find which input bytes lead to the start and end states.
            vmask start_m = VI::test_low_bit(from);
            vmask end_m = VI::test_high_bit(to);
            vmask trans_m = VI::test_nz(trans);

            vmask covered = end_m & trans_m;
            carry = last_carry + ((double_vmask)start_m << 1) + trans_m;
            vmask matches = ~carry & covered;
            // Check for 1-char matches
            matches |= has_1_char_match & start_m & end_m;

            last_to = to;
            last_carry = carry >> VL;

            // Look through the bitset of all potential matches, and run a
            // backwards verification step to weed out false positives. Any
            // real matches we pass off to the match handler.
            while (matches) {
                const uint8_t *end = chunk + bsf64(matches);
                matches &= matches - 1;

                const uint8_t *start = match_verifier.verify(f.data, end);
                if (start)
                    match_handler.handle_match(f, start, end);
            }
        }

        // Run the slow backwards NFA on all the remainder bytes that don't fit
        // in a vector register. We could probably read past the original
        // buffer or do an unaligned read or something, but oh well.
        for (; chunk < f.data + f.size; chunk++) {
            const uint8_t *start = match_verifier.verify(f.data, chunk);
            if (start)
                match_handler.handle_match(f, start, chunk);
        }

        return match_handler.finish(f);
    }
};

// Template alias using VEC_INFO class defined in the Makefile
template <typename MatchHandler>
using Miroslav = _Miroslav<VEC_INFO, BasicMatchVerifier<true>, MatchHandler>;
