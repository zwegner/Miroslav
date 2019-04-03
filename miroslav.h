#pragma once

#include <assert.h>
#include <stdint.h>

#include <iostream>
#include <tuple>
#include <vector>

#include <immintrin.h>

#define UNUSED __attribute__((unused))

static inline uint32_t bsf64(uint64_t x) {
	return __builtin_ctzll(x);
}

static inline uint32_t bsf32(uint32_t x) {
	return __builtin_ctzl(x);
}

#define START_STATE (0)
#define END_STATE (1)

// Wacky macro to make tuple unpacking a little less annoying
#define FOR_EACH_EDGE(c, from, to, edges) \
    for (auto edge_i = edges.begin(); \
            edge_i != edges.end() ? (std::tie(c, from, to) = *edge_i), 1 : 0; \
            edge_i++)

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

    static inline V vec_shr(V &a, uint32_t shift) {
        return _mm256_srli_epi32(a, shift);
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

typedef std::vector<std::tuple<uint8_t, uint32_t, uint32_t>> NFAEdgeList;

////////////////////////////////////////////////////////////////////////////////
// Match verifier
// Since the vectorized matcher can give false positives, we have to run through
// each potential match backwards from the end character.
////////////////////////////////////////////////////////////////////////////////

// Basic bitset NFA simulator
template<typename StateInfo>
class BasicMatchVerifier {
    typedef typename StateInfo::smask smask;
    static const uint32_t MAX_STATES = StateInfo::MAX_STATES;

    // This is a table of [input_byte][state] -> prev_states
    smask back_edges[256][MAX_STATES];
    // All states that lead to a successful match from each input byte
    smask match_mask[256];
    // Shortcut mask: for each input byte, keep a mask of states that have a
    // predecessor state so we don't try any unnecessary lookups
    smask next_mask[256];

public:
    BasicMatchVerifier(NFAEdgeList &edges) {
        for (uint32_t i = 0; i < 256; i++) {
            for (uint32_t j = 0; j < MAX_STATES; j++)
                back_edges[i][j] = 0;
            next_mask[i] = 0;
            match_mask[i] = 0;
        }

        // Initialize state mask tables
        uint8_t c;
        uint32_t from, to;
        FOR_EACH_EDGE(c, from, to, edges) {
            assert(from < MAX_STATES);
            assert(to < MAX_STATES);
            next_mask[c] |= (smask)1 << to;

            if (from == START_STATE)
                match_mask[c] |= (smask)1 << to;
            else
                back_edges[c][to] |= (smask)1 << from;
        }
    }

    const uint8_t *verify(const uint8_t *data, const uint8_t *end) {
        smask states = 1 << END_STATE;

        do {
            uint8_t c = *end;

            if (match_mask[c] & states)
                return end;

            states &= next_mask[c];

            // Iterate through all current states and look up their next states
            smask next_states = 0;
            while (states) {
                uint32_t s = StateInfo::bsf(states);
                next_states |= back_edges[c][s];
                states &= states - 1;
            }
            states = next_states;
        } while (states && --end >= data);

        return NULL;
    }
};

// Structs with definitions for the BasicMatchVerifier.
// smask should have MAX_STATES bits.
// The backwards state table will be (max/8) * max * 256 bytes.
// So 8 states is 2K, 16 => 8K, 32 => 32K, 64 => 128K, 128 => 512K

struct StateInfo32 {
    static const uint32_t MAX_STATES = 32;
    typedef uint32_t smask;
    static inline uint32_t bsf(smask x) { return bsf32(x); }
};
struct StateInfo64 {
    static const uint32_t MAX_STATES = 64;
    typedef uint64_t smask;
    static inline uint32_t bsf(smask x) { return bsf64(x); }
};
struct StateInfo128 {
    static const uint32_t MAX_STATES = 128;
    typedef __uint128_t smask;
    static inline uint32_t bsf(smask x) {
        if ((uint64_t)x)
            return bsf64(((uint64_t)x));
        return bsf64(x >> 64) + 64;
    }
};

typedef BasicMatchVerifier<StateInfo32> BasicMatchVerifier32;
typedef BasicMatchVerifier<StateInfo64> BasicMatchVerifier64;
typedef BasicMatchVerifier<StateInfo128> BasicMatchVerifier128;

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
template<typename VI, typename MatchVerifier, typename MatchHandler,
    const uint32_t N_BYTES, const uint32_t N_SHIFTS, const uint32_t SHIFTS[]>
class _Miroslav {
    // Hacky substitute for "using"
    typedef typename VI::V V;
    typedef typename VI::vmask vmask;
    typedef typename VI::double_vmask double_vmask;
    static const uint32_t VL = VI::VL;
    static const uint32_t LMASK = VI::LMASK;

    static inline uint8_t state_mask(uint32_t state, uint32_t byte) {
        // If we only have one byte of state bits, start and end states
        // get the bottom and top bits, and all the other states rotate
        // through the other 6 bits
        if (N_BYTES == 1) {
            if (state == START_STATE)
                return 1 << 0;
            if (state == END_STATE)
                return 1 << 7;
            return 1 << (state % 6 + 1);
        }
        // For two or more state bytes, we use the top bit of the first
        // two bytes for the start and end states (for vpmovmskb convenience)
        // and all the other states rotate through the remaining bits
        else {
            if (state == START_STATE)
                return byte == 0 ? 1 << 7 : 0;
            if (state == END_STATE)
                return byte == 1 ? 1 << 7 : 0;
            state = state % (N_BYTES * 8 - 2);
            // Skip over bit 7 (start) and 15 (end)
            state += (state >= 7);
            state += (state >= 15);
            // Return the proper bit within this byte if the high bits of the
            // state id match the byte number
            return ((state >> 3) == byte) << (state & 7);
        }
    }

    static inline uint8_t char_mask(uint8_t c, uint32_t shift) {
        return (c >> shift) & LMASK;
    }

    MatchVerifier match_verifier;
    MatchHandler &match_handler;

    V from_states[N_BYTES][N_SHIFTS];
    V to_states[N_BYTES][N_SHIFTS];
    V v_char_mask;

    // For branchless testing of whether a pattern has a 1-character match.
    // Cuts down on false matches when the start and end characters of a
    // pattern hash to the same character class, which would cause every
    // occurrence of a character in that class to generate a match
    vmask has_1_char_match;

public:
    _Miroslav(NFAEdgeList &edges, MatchHandler &handler) :
            match_verifier(edges), match_handler(handler) {
        uint8_t from_state_bytes[N_BYTES][N_SHIFTS][VL] = {{{0}}};
        uint8_t to_state_bytes[N_BYTES][N_SHIFTS][VL] = {{{0}}};

        has_1_char_match = 0;

        // Initialize from/to state masks tables
        uint8_t c;
        uint32_t from, to;
        FOR_EACH_EDGE(c, from, to, edges) {
            for (uint32_t b = 0; b < N_BYTES; b++) {
                uint8_t fm = state_mask(from, b);
                uint8_t tm = state_mask(to, b);
                for (uint32_t s = 0; s < N_SHIFTS; s++) {
                    from_state_bytes[b][s][char_mask(c, SHIFTS[s])] |= fm;
                    to_state_bytes[b][s][char_mask(c, SHIFTS[s])] |= tm;
                }
            }

            // Check for 1-character matches and update the mask
            if (from == START_STATE && to == END_STATE)
                has_1_char_match = (vmask)-1;
        }

        for (uint32_t b = 0; b < N_BYTES; b++) {
            for (uint32_t s = 0; s < N_SHIFTS; s++) {
                VI::prepare_state_table(from_state_bytes[b][s]);
                VI::prepare_state_table(to_state_bytes[b][s]);
                from_states[b][s] = *(V*)from_state_bytes[b][s];
                to_states[b][s] = *(V*)to_state_bytes[b][s];
            }
        }

        v_char_mask = VI::broadcast(LMASK);
    }

    typename MatchHandler::return_type run(File &f) {
        match_handler.start();

        double_vmask carry, last_carry = 0;

        // Fill a vector for each byte of state mask with the starting state. This
        // vector tracks the state between iterations of the main loop. Only the
        // last byte of each of these vectors is ever used.
        V last_to[N_BYTES];
        for (uint32_t b = 0; b < N_BYTES; b++)
            last_to[b] = VI::broadcast(state_mask(START_STATE, b));

        const uint8_t *chunk;
        for (chunk = f.data; chunk + VL <= f.data + f.size; chunk += VL) {
            V input = *(V*)chunk;

            vmask start_m, end_m, seq_m = 0;

            for (uint32_t b = 0; b < N_BYTES; b++) {
                V from, to;
                // For each of the shifts defined in the template arguments, do
                // a vector table lookup by shifting each byte by the shift, masking,
                // and doing a permute on the appropriate table vector. We AND the
                // results together for each of these shifts, which should narrow down
                // the number of false positives from different input bytes
                // mapping to the same character class.
                for (uint32_t s = 0; s < N_SHIFTS; s++) {
                    V masked_input = input;
                    if (SHIFTS[s])
                        masked_input = VI::vec_shr(masked_input, SHIFTS[s]);
                    masked_input = VI::vec_and(masked_input, v_char_mask);

                    V f = VI::permute(from_states[b][s], masked_input);
                    V t = VI::permute(to_states[b][s], masked_input);
                    if (s == 0) {
                        from = f;
                        to = t;
                    } else {
                        from = VI::vec_and(from, f);
                        to = VI::vec_and(to, t);
                    }
                }

                // Get a vector of the to states, but shifted back in the data
                // stream by 1 byte. We fill the empty space in the first lane with
                // the last lane from last_to (which is initialized to the starting
                // state).
                V shl_to_1 = VI::vec_lanes_shl_1(to, last_to[b]);
                last_to[b] = to;

                // Test which input bytes can lead from the start state, and lead to
                // the end state. We handle the N_BYTES=1 and N cases differently,
                // because for 1, both bits are in the same byte, but otherwise, it's
                // a bit cheaper to have the start and end state bits in the top bit
                // of the first two state bytes, since vpmovmskb only looks at the
                // high bits of each byte in a vector. All this code should be 
                // unrolled/branchless, with start_m and end_m being set exactly once.
                if (N_BYTES == 1) {
                    start_m = VI::test_low_bit(from);
                    end_m = VI::test_high_bit(to);
                } else {
                    if (b == 0)
                        start_m = VI::test_high_bit(from);
                    else if (b == 1)
                        end_m = VI::test_high_bit(to);
                }

                // Now find all input bytes that can come from some state that the
                // previous input byte could lead to.
                V seq = VI::vec_and(shl_to_1, from);
                seq_m |= VI::test_nz(seq);
            }

            // Test for potential matches. We use the ripple of carries through the
            // "seq" mask to find sequences of input bytes that lead from the start
            // state to the end state, while passing through valid state transitions.
            // To be precise, since carries will ripple past the end mask, we find
            // bits in the end mask that are cleared by a carry. This is slightly
            // complicated by a couple factors: first, we have to keep track of
            // carries across iterations (which is why we use the "double_vmask"
            // type and shift carry right by VL each iteration), and second, extra
            // bits in the start mask can make us think a carry didn't happen, so
            // we clear out bits from start_m before testing the carries with
            // the end mask.
            carry = last_carry + ((double_vmask)start_m << 1) + seq_m;
            vmask covered = end_m & seq_m;
            vmask matches = ~(carry & ~start_m) & covered;

            // Check for 1-char matches, if they're possible
            matches |= has_1_char_match & start_m & end_m;

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
static constexpr uint32_t SHIFTS[] = {0, 3};
static constexpr uint32_t N_SHIFTS = sizeof(SHIFTS) / sizeof(SHIFTS[0]);
template <typename MatchHandler, typename MatchVerifier>
using Miroslav = _Miroslav<VEC_INFO, MatchVerifier, MatchHandler, 1, N_SHIFTS, SHIFTS>;
